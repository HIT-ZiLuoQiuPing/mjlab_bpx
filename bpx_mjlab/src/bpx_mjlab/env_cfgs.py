from copy import deepcopy

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RayCastSensorCfg,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)

from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG

from bpx_mjlab.bpx.bpx_constants import (
    BPX_ACTION_SCALE,
    BPX_SIM2REAL_JOINT_ORDER,
    FOOT_GEOMS,
    FOOT_SITES,
    get_bpx_robot_cfg,
)


_LEG_PREFIXES = ("fl", "fr", "hl", "hr")
CALF_GEOMS = tuple(
    f"{leg}_calf_link_collision_{idx}"
    for leg in _LEG_PREFIXES
    for idx in (0, 1)
)
THIGH_GEOMS = tuple(
    f"{leg}_thigh_link_collision_{idx}"
    for leg in _LEG_PREFIXES
    for idx in (0, 1)
)
TORSO_GEOMS = ("torso_collision_0", "torso_collision_1", "torso_collision_2")
DANGEROUS_GROUND_GEOMS = (*TORSO_GEOMS, *THIGH_GEOMS)


def _bpx_action_scale_for_joint(joint_name: str) -> float:
    if joint_name.endswith("_hip_roll_joint"):
        return BPX_ACTION_SCALE[".*_hip_roll_joint"]
    if joint_name.endswith("_hip_pitch_joint"):
        return BPX_ACTION_SCALE[".*_hip_pitch_joint"]
    if joint_name.endswith("_knee_joint"):
        return BPX_ACTION_SCALE[".*_knee_joint"]
    raise ValueError(f"Unsupported BPX action joint: {joint_name}")


def _make_bpx_simreal_actions() -> dict[str, JointPositionActionCfg]:
    return {
        f"joint_pos_{idx:02d}_{joint_name}": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(joint_name,),
            scale=_bpx_action_scale_for_joint(joint_name),
            use_default_offset=True,
        )
        for idx, joint_name in enumerate(BPX_SIM2REAL_JOINT_ORDER)
    }


def _configure_bpx_simreal_actor_terms(actor_terms: dict) -> None:
    joint_asset_cfg = SceneEntityCfg(
        "robot",
        joint_names=BPX_SIM2REAL_JOINT_ORDER,
        preserve_order=True,
    )

    actor_terms["base_ang_vel"].scale = 0.25
    actor_terms["command"].scale = (2.0, 2.0, 0.25)
    actor_terms["joint_vel"].scale = 0.05

    actor_terms["joint_pos"].params["asset_cfg"] = deepcopy(joint_asset_cfg)
    actor_terms["joint_vel"].params["asset_cfg"] = deepcopy(joint_asset_cfg)

    for term in actor_terms.values():
        term.clip = (-100.0, 100.0)


def _safe_set_asset_names(term, field_name: str, names: tuple[str, ...]) -> bool:
    """
    兼容不同 mjlab 版本：
    有的 term.params 里是 asset_cfg.body_names/site_names/geom_names；
    有的版本可能直接是 body_name/body_names 等。
    这里能设就设，不能设就跳过，避免导入阶段直接 KeyError。
    """
    params = getattr(term, "params", None)
    if not isinstance(params, dict):
        return False

    asset_cfg = params.get("asset_cfg", None)
    if asset_cfg is not None and hasattr(asset_cfg, field_name):
        setattr(asset_cfg, field_name, names)
        return True

    if field_name in params:
        params[field_name] = names
        return True

    singular_map = {
        "body_names": "body_name",
        "site_names": "site_name",
        "geom_names": "geom_name",
    }
    singular = singular_map.get(field_name)
    if singular is not None and singular in params:
        params[singular] = names[0] if len(names) == 1 else names
        return True

    return False


def _safe_pop_term(term_dict, key: str) -> None:
    if term_dict is not None and key in term_dict:
        term_dict.pop(key, None)


def _bpx_track_forward_velocity(
    env,
    command_name: str,
    std: float,
) -> torch.Tensor:
    asset = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    error = command[:, 0] - asset.data.root_link_lin_vel_b[:, 0]
    return torch.exp(-(error.square()) / std**2)


def _bpx_track_lateral_velocity(
    env,
    command_name: str,
    std: float,
) -> torch.Tensor:
    asset = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    error = command[:, 1] - asset.data.root_link_lin_vel_b[:, 1]
    return torch.exp(-(error.square()) / std**2)


def _bpx_track_yaw_velocity(
    env,
    command_name: str,
    std: float,
) -> torch.Tensor:
    asset = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    error = command[:, 2] - asset.data.root_link_ang_vel_b[:, 2]
    return torch.exp(-(error.square()) / std**2)


def _bpx_forward_lateral_drift(
    env,
    command_name: str,
    min_forward_command: float = 0.2,
    lateral_command_threshold: float = 0.05,
    yaw_command_threshold: float = 0.05,
) -> torch.Tensor:
    asset = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    straight = (
        (command[:, 0] > min_forward_command)
        & (torch.abs(command[:, 1]) < lateral_command_threshold)
        & (torch.abs(command[:, 2]) < yaw_command_threshold)
    )
    lateral_velocity = asset.data.root_link_lin_vel_b[:, 1]
    return lateral_velocity.square() * straight.float()


def _bpx_forward_yaw_drift(
    env,
    command_name: str,
    min_forward_command: float = 0.2,
    lateral_command_threshold: float = 0.05,
    yaw_command_threshold: float = 0.05,
) -> torch.Tensor:
    asset = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    straight = (
        (command[:, 0] > min_forward_command)
        & (torch.abs(command[:, 1]) < lateral_command_threshold)
        & (torch.abs(command[:, 2]) < yaw_command_threshold)
    )
    yaw_velocity = asset.data.root_link_ang_vel_b[:, 2]
    return yaw_velocity.square() * straight.float()


def _bpx_terrain_levels_vel(
    env,
    env_ids: torch.Tensor,
    command_name: str,
    promotion_distance_ratio: float = 0.75,
    demotion_command_ratio: float = 0.5,
) -> dict[str, torch.Tensor]:
    asset = env.scene["robot"]
    terrain = env.scene.terrain
    assert terrain is not None
    terrain_generator = terrain.cfg.terrain_generator
    assert terrain_generator is not None

    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."

    distance = torch.norm(
        asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
        dim=1,
    )
    promotion_distance = terrain_generator.size[0] * promotion_distance_ratio

    # 只有真正跑满 episode 的轨迹才允许升级，避免高等级地形过早堆上来。
    timed_out = env.termination_manager.get_term("time_out")[env_ids]
    move_up = (distance > promotion_distance) & timed_out

    move_down = (
        distance
        < torch.norm(command[env_ids, :2], dim=1)
        * env.max_episode_length_s
        * demotion_command_ratio
    )
    move_down &= ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)

    levels = terrain.terrain_levels.float()
    result: dict[str, torch.Tensor] = {
        "mean": torch.mean(levels),
        "max": torch.max(levels),
        "promotion_distance": torch.tensor(promotion_distance, device=env.device),
        "move_up_rate": torch.mean(move_up.float()),
        "move_down_rate": torch.mean(move_down.float()),
    }

    sub_terrain_names = list(terrain_generator.sub_terrains.keys())
    terrain_origins = terrain.terrain_origins
    assert terrain_origins is not None
    num_cols = terrain_origins.shape[1]
    if num_cols == len(sub_terrain_names):
        types = terrain.terrain_types
        for i, name in enumerate(sub_terrain_names):
            mask = types == i
            if mask.any():
                result[name] = torch.mean(levels[mask])

    return result


def bpx_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_velocity_env_cfg()

    # 仿真接触参数先保守一点，避免接触过多时报错。
    cfg.sim.njmax = 300
    cfg.sim.nconmax = 128
    cfg.sim.contact_sensor_maxmatch = 64
    cfg.sim.mujoco.ccd_iterations = 50

    # 换成 BPX 机器人。
    cfg.scene.entities = {
        "robot": get_bpx_robot_cfg(),
    }

    # Viewer 跟随 BPX 主躯干。
    cfg.viewer.body_name = "torso"
    cfg.viewer.distance = 2.0
    cfg.viewer.elevation = -10.0

    # 平地训练。
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # 平地训练不需要 raycast 扫描。
    # 默认 velocity task 里可能有 terrain_scan / foot_height_scan。
    # BPX 当前没有给这些 raycast sensor 配 frame，保留会导致：
    # RuntimeError: stack expects a non-empty TensorList
    cfg.scene.sensors = tuple(
        s for s in (cfg.scene.sensors or ())
        if s.name not in ("terrain_scan", "foot_height_scan")
    )

    # 同时删除依赖这些 raycast sensor 的 observation term。
    for obs_group in cfg.observations.values():
        terms = getattr(obs_group, "terms", None)
        if not isinstance(terms, dict):
            continue

        for term_name, term_cfg in list(terms.items()):
            params = getattr(term_cfg, "params", {})
            sensor_name = params.get("sensor_name") if isinstance(params, dict) else None

            if (
                term_name in ("height_scan", "foot_height", "foot_height_scan")
                or sensor_name in ("terrain_scan", "foot_height_scan")
            ):
                terms.pop(term_name, None)

    cfg.curriculum.pop("terrain_levels", None)

    # 删除仍然引用 raycast sensor 的 reward term。
    # 只把 weight 设成 0 不够，因为 RewardManager 会先初始化 term。
    for reward_name, term_cfg in list(cfg.rewards.items()):
        params = getattr(term_cfg, "params", {})
        if not isinstance(params, dict):
            continue

        should_remove = False

        # 常见显式字段。
        for key in (
            "height_sensor_name",
            "sensor_name",
            "raycast_sensor_name",
            "terrain_sensor_name",
        ):
            if params.get(key) in ("terrain_scan", "foot_height_scan"):
                should_remove = True

        # 兜底：params 里任何值直接等于这两个名字，也删。
        for value in params.values():
            if value in ("terrain_scan", "foot_height_scan"):
                should_remove = True

        if should_remove:
            cfg.rewards.pop(reward_name, None)


    # 四个脚尖接触传感器。
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=FOOT_GEOMS,
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    # 非脚部碰地传感器：用于判断摔倒/非法接触。
    nonfoot_ground_cfg = ContactSensorCfg(
        name="nonfoot_ground_touch",
        primary=ContactMatch(
            mode="geom",
            entity="robot",
            pattern=r".*_collision_\d+$",
            exclude=FOOT_GEOMS,
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        feet_ground_cfg,
        nonfoot_ground_cfg,
    )

    # 动作顺序严格对齐部署侧 type-major policy contract。
    cfg.actions = _make_bpx_simreal_actions()

    # 尝试把默认 Go1/G1 的 body/site/geom 名字替换成 BPX。
    # 这里全部 safe，不存在就跳过，避免任务注册失败。
    if "foot_height" in cfg.observations.get("critic", {}).terms:
        _safe_set_asset_names(
            cfg.observations["critic"].terms["foot_height"],
            "site_names",
            FOOT_SITES,
        )

    if "foot_friction" in cfg.events:
        _safe_set_asset_names(
            cfg.events["foot_friction"],
            "geom_names",
            FOOT_GEOMS,
        )

    if "base_com" in cfg.events:
        _safe_set_asset_names(
            cfg.events["base_com"],
            "body_names",
            ("torso",),
        )

    if "pose" in cfg.rewards:
        cfg.rewards["pose"].params["std_standing"] = {
            ".*_hip_roll_joint": 0.05,
            ".*_hip_pitch_joint": 0.10,
            ".*_knee_joint": 0.10,
        }
        cfg.rewards["pose"].params["std_walking"] = {
            ".*_hip_roll_joint": 0.30,
            ".*_hip_pitch_joint": 0.30,
            ".*_knee_joint": 0.60,
        }
        cfg.rewards["pose"].params["std_running"] = {
            ".*_hip_roll_joint": 0.30,
            ".*_hip_pitch_joint": 0.30,
            ".*_knee_joint": 0.60,
        }

    if "upright" in cfg.rewards:
        _safe_set_asset_names(
            cfg.rewards["upright"],
            "body_names",
            ("torso",),
        )

    if "body_ang_vel" in cfg.rewards:
        _safe_set_asset_names(
            cfg.rewards["body_ang_vel"],
            "body_names",
            ("torso",),
        )

    for reward_name in ("foot_clearance", "foot_swing_height", "foot_slip"):
        if reward_name in cfg.rewards:
            _safe_set_asset_names(
                cfg.rewards[reward_name],
                "site_names",
                FOOT_SITES,
            )

    # 初期先关掉一些容易干扰的奖励，等能跑起来再慢慢加。
    for reward_name in ("body_ang_vel", "angular_momentum", "air_time"):
        if reward_name in cfg.rewards:
            cfg.rewards[reward_name].weight = 0.0

    # 非脚部碰地就终止。
    cfg.terminations["illegal_contact"] = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_name": nonfoot_ground_cfg.name},
    )

    # 速度命令可视化高度。
    cmd = cfg.commands["twist"]
    assert isinstance(cmd, UniformVelocityCommandCfg)
    cmd.viz.z_offset = 0.5

    if play:
        cfg.episode_length_s = int(1e9)

        if "actor" in cfg.observations:
            cfg.observations["actor"].enable_corruption = False

        cfg.events.pop("push_robot", None)

    return cfg


def bpx_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_velocity_env_cfg()

    cfg.sim.njmax = 1500
    cfg.sim.nconmax = 192
    cfg.sim.contact_sensor_maxmatch = 128
    cfg.sim.mujoco.ccd_iterations = 50

    cfg.scene.entities = {
        "robot": get_bpx_robot_cfg(),
    }
    cfg.viewer.body_name = "torso"
    cfg.viewer.distance = 2.5
    cfg.viewer.elevation = -10.0

    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "generator"
    terrain_generator = deepcopy(ROUGH_TERRAINS_CFG)
    terrain_generator.curriculum = True
    terrain_proportions = {
        "flat": 0.12,
        "pyramid_stairs": 0.20,
        "pyramid_stairs_inv": 0.08,
        "hf_pyramid_slope": 0.25,
        "hf_pyramid_slope_inv": 0.20,
        "random_rough": 0.08,
        "wave_terrain": 0.07,
    }
    for terrain_name, proportion in terrain_proportions.items():
        if terrain_name in terrain_generator.sub_terrains:
            terrain_generator.sub_terrains[terrain_name].proportion = proportion
    for terrain_name in ("pyramid_stairs", "pyramid_stairs_inv"):
        if terrain_name in terrain_generator.sub_terrains:
            stairs_cfg = terrain_generator.sub_terrains[terrain_name]
            stairs_cfg.step_width = 0.35
            stairs_cfg.step_height_range = (0.04, 0.16)
    for terrain_name in ("hf_pyramid_slope", "hf_pyramid_slope_inv"):
        if terrain_name in terrain_generator.sub_terrains:
            terrain_generator.sub_terrains[terrain_name].slope_range = (0.0, 0.85)
    cfg.scene.terrain.terrain_generator = terrain_generator
    cfg.scene.terrain.max_init_terrain_level = 1
    cfg.scene.extent = 3.0

    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan":
            assert isinstance(sensor, RayCastSensorCfg)
            assert isinstance(sensor.frame, ObjRef)
            sensor.frame.name = "torso"
            sensor.debug_vis = False
        elif sensor.name == "foot_height_scan":
            assert isinstance(sensor, TerrainHeightSensorCfg)
            sensor.frame = tuple(
                ObjRef(type="site", name=site_name, entity="robot")
                for site_name in FOOT_SITES
            )
            sensor.pattern = RingPatternCfg.single_ring(radius=0.035, num_samples=4)
            sensor.debug_vis = False

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=FOOT_GEOMS,
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    calf_ground_cfg = ContactSensorCfg(
        name="calf_ground_touch",
        primary=ContactMatch(
            mode="geom",
            entity="robot",
            pattern=CALF_GEOMS,
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    dangerous_ground_cfg = ContactSensorCfg(
        name="dangerous_ground_touch",
        primary=ContactMatch(
            mode="geom",
            entity="robot",
            pattern=DANGEROUS_GROUND_GEOMS,
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        feet_ground_cfg,
        calf_ground_cfg,
        dangerous_ground_cfg,
    )

    cfg.actions = _make_bpx_simreal_actions()

    if "foot_friction" in cfg.events:
        _safe_set_asset_names(
            cfg.events["foot_friction"],
            "geom_names",
            FOOT_GEOMS,
        )
    if "base_com" in cfg.events:
        _safe_set_asset_names(
            cfg.events["base_com"],
            "body_names",
            ("torso",),
        )

    if "pose" in cfg.rewards:
        cfg.rewards["pose"].params["std_standing"] = {
            ".*_hip_roll_joint": 0.05,
            ".*_hip_pitch_joint": 0.10,
            ".*_knee_joint": 0.10,
        }
        cfg.rewards["pose"].params["std_walking"] = {
            ".*_hip_roll_joint": 0.30,
            ".*_hip_pitch_joint": 0.30,
            ".*_knee_joint": 0.60,
        }
        cfg.rewards["pose"].params["std_running"] = {
            ".*_hip_roll_joint": 0.35,
            ".*_hip_pitch_joint": 0.35,
            ".*_knee_joint": 0.70,
        }
    if "upright" in cfg.rewards:
        _safe_set_asset_names(cfg.rewards["upright"], "body_names", ("torso",))
        cfg.rewards["upright"].params["terrain_sensor_names"] = ("terrain_scan",)
    if "body_ang_vel" in cfg.rewards:
        _safe_set_asset_names(cfg.rewards["body_ang_vel"], "body_names", ("torso",))
    for reward_name in ("foot_clearance", "foot_slip"):
        if reward_name in cfg.rewards:
            _safe_set_asset_names(cfg.rewards[reward_name], "site_names", FOOT_SITES)

    if "track_linear_velocity" in cfg.rewards:
        cfg.rewards["track_linear_velocity"].weight = 3.5
        cfg.rewards["track_linear_velocity"].params["std"] = 0.35
    if "track_angular_velocity" in cfg.rewards:
        cfg.rewards["track_angular_velocity"].weight = 3.2
        cfg.rewards["track_angular_velocity"].params["std"] = 0.40
    cfg.rewards["track_forward_velocity_fine"] = RewardTermCfg(
        func=_bpx_track_forward_velocity,
        weight=1.4,
        params={"command_name": "twist", "std": 0.25},
    )
    cfg.rewards["track_lateral_velocity_fine"] = RewardTermCfg(
        func=_bpx_track_lateral_velocity,
        weight=1.2,
        params={"command_name": "twist", "std": 0.16},
    )
    cfg.rewards["track_yaw_velocity_fine"] = RewardTermCfg(
        func=_bpx_track_yaw_velocity,
        weight=1.4,
        params={"command_name": "twist", "std": 0.25},
    )
    cfg.rewards["forward_lateral_drift"] = RewardTermCfg(
        func=_bpx_forward_lateral_drift,
        weight=-1.5,
        params={"command_name": "twist"},
    )
    cfg.rewards["forward_yaw_drift"] = RewardTermCfg(
        func=_bpx_forward_yaw_drift,
        weight=-1.2,
        params={"command_name": "twist"},
    )
    if "body_ang_vel" in cfg.rewards:
        cfg.rewards["body_ang_vel"].weight = -0.05
    if "angular_momentum" in cfg.rewards:
        cfg.rewards["angular_momentum"].weight = 0.0
    if "action_rate_l2" in cfg.rewards:
        cfg.rewards["action_rate_l2"].weight = -0.12
    if "air_time" in cfg.rewards:
        cfg.rewards["air_time"].weight = 0.2
    if "foot_clearance" in cfg.rewards:
        cfg.rewards["foot_clearance"].params["target_height"] = 0.12
    if "foot_swing_height" in cfg.rewards:
        cfg.rewards["foot_swing_height"].params["target_height"] = 0.12
    cfg.rewards["calf_ground_touch"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-0.1,
        params={"sensor_name": calf_ground_cfg.name},
    )
    cfg.rewards["termination"] = RewardTermCfg(
        func=mdp.is_terminated,
        weight=-25.0,
    )

    cfg.terminations["illegal_contact"] = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_name": dangerous_ground_cfg.name},
    )

    cmd = cfg.commands["twist"]
    assert isinstance(cmd, UniformVelocityCommandCfg)
    cmd.viz.z_offset = 0.5
    cmd.ranges.lin_vel_x = (-0.20, 0.75)
    cmd.ranges.lin_vel_y = (-0.10, 0.10)
    cmd.ranges.ang_vel_z = (-0.25, 0.25)
    cmd.rel_standing_envs = 0.02
    cmd.rel_forward_envs = 0.75
    cmd.resampling_time_range = (6.0, 10.0)

    cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
        func=_bpx_terrain_levels_vel,
        params={
            "command_name": "twist",
            "promotion_distance_ratio": 0.75,
            "demotion_command_ratio": 0.5,
        },
    )
    cfg.curriculum["command_vel"] = CurriculumTermCfg(
        func=mdp.commands_vel,
        params={
            "command_name": "twist",
            "velocity_stages": [
                {
                    "step": 0,
                    "lin_vel_x": (-0.20, 0.75),
                    "lin_vel_y": (-0.10, 0.10),
                    "ang_vel_z": (-0.25, 0.25),
                },
                {
                    "step": 6000 * 16,
                    "lin_vel_x": (-0.30, 0.95),
                    "lin_vel_y": (-0.18, 0.18),
                    "ang_vel_z": (-0.40, 0.40),
                },
                {
                    "step": 12000 * 16,
                    "lin_vel_x": (-0.35, 1.10),
                    "lin_vel_y": (-0.22, 0.22),
                    "ang_vel_z": (-0.48, 0.48),
                },
                {
                    "step": 18000 * 16,
                    "lin_vel_x": (-0.45, 1.25),
                    "lin_vel_y": (-0.30, 0.30),
                    "ang_vel_z": (-0.60, 0.60),
                },
                {
                    "step": 26000 * 16,
                    "lin_vel_x": (-0.50, 1.40),
                    "lin_vel_y": (-0.30, 0.30),
                    "ang_vel_z": (-0.60, 0.60),
                },
                {
                    "step": 36000 * 16,
                    "lin_vel_x": (-0.55, 1.60),
                    "lin_vel_y": (-0.32, 0.32),
                    "ang_vel_z": (-0.65, 0.65),
                },
                {
                    "step": 45000 * 16,
                    "lin_vel_x": (-0.60, 1.80),
                    "lin_vel_y": (-0.35, 0.35),
                    "ang_vel_z": (-0.70, 0.70),
                },
            ],
        },
    )

    base_actor_terms = cfg.observations["actor"].terms
    base_critic_terms = cfg.observations["critic"].terms
    actor_term_names = (
        "base_ang_vel",
        "projected_gravity",
        "command",
        "joint_pos",
        "joint_vel",
        "actions",
    )
    privileged_term_names = (
        "base_lin_vel",
        "height_scan",
        "foot_height",
        "foot_air_time",
        "foot_contact",
        "foot_contact_forces",
    )

    actor_terms = {
        name: deepcopy(base_actor_terms[name])
        for name in actor_term_names
    }
    _configure_bpx_simreal_actor_terms(actor_terms)
    critic_terms = {
        name: deepcopy(actor_terms[name])
        for name in actor_term_names
    }
    critic_terms.update(
        {
            name: deepcopy(base_critic_terms[name])
            for name in privileged_term_names
        }
    )
    estimator_target_terms = {
        "base_lin_vel": deepcopy(base_critic_terms["base_lin_vel"]),
        "height_scan": deepcopy(base_critic_terms["height_scan"]),
    }

    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=not play,
            nan_policy="sanitize",
        ),
        "actor_history": ObservationGroupCfg(
            terms=deepcopy(actor_terms),
            concatenate_terms=True,
            enable_corruption=not play,
            history_length=5,
            flatten_history_dim=True,
            nan_policy="sanitize",
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
            nan_policy="sanitize",
        ),
        "estimator_target": ObservationGroupCfg(
            terms=estimator_target_terms,
            concatenate_terms=True,
            enable_corruption=False,
            nan_policy="sanitize",
        ),
    }

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.events.pop("push_robot", None)

    return cfg
