from copy import deepcopy

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.reward_manager import RewardTermCfg
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

# 下面几个 reward term 直接在 cfg 里写函数比较麻烦，单独写成函数放这里。

# 函数作用： 根据命令和当前速度计算一个奖励，鼓励机器人更精确地跟踪命令的速度，尤其是在较低速度下。
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

# 鼓励机器人跟踪横向速度命令，减少横向漂移。
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

# 鼓励机器人跟踪旋转速度命令，减少转向误差。
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

# 当机器人有明显的前向速度但命令要求它直行时，惩罚它的横向速度，鼓励它减少漂移。
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

# 当机器人有明显的前向速度但命令要求它直行时，惩罚它的偏航速度，鼓励它减少转向漂移。
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

# 根据机器人与环境原点的距离以及命令的速度，动态调整地形难度等级，鼓励机器人逐渐适应更复杂的地形，同时避免过早或过快地增加难度。
def _bpx_terrain_levels_vel(
    env,
    env_ids: torch.Tensor,
    command_name: str,
    promotion_distance_ratio: float = 0.9, # 当机器人与环境原点的距离超过地形块长度的 90% 时，考虑升级地形难度。
    demotion_command_ratio: float = 0.55, # 当机器人与环境原点的距离小于命令速度的 55% 时，考虑降低地形难度。
    max_level_schedule: tuple[tuple[int, int], ...] | None = None, # 可选的时间表，用于根据训练进度动态调整允许的最大地形难度等级，格式为 ((step1, level1), (step2, level2), ...)，表示在训练步骤 step1 之后允许的最大地形等级为 level1，以此类推。
) -> dict[str, torch.Tensor]:
    asset = env.scene["robot"] # 获取机器人资产，后续会用到它的数据来计算与环境原点的距离。
    terrain = env.scene.terrain # 获取地形信息，后续会用到地形块的长度来计算升级距离，以及当前地形等级来判断是否可以升级或降级。
    assert terrain is not None #，确保地形信息存在。
    terrain_generator = terrain.cfg.terrain_generator # 获取地形生成器配置，后续会用到它的 size 属性来计算升级距离。
    assert terrain_generator is not None

    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."

    distance = torch.norm(
        asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], # 计算机器人当前在水平面上与环境原点的距离，用于判断是否满足升级或降级地形的条件。
        dim=1,
    )
    promotion_distance = terrain_generator.size[0] * promotion_distance_ratio # 计算升级地形的距离阈值，通常是地形块长度的某个比例。
    max_allowed_level = terrain.max_terrain_level - 1 # 初始化允许的最大地形等级为地形配置中定义的最大等级减一（因为等级通常从 0 开始），后续会根据训练进度和 max_level_schedule 进行调整。
    if max_level_schedule is not None:
        for step, level in max_level_schedule:
            if env.common_step_counter >= step: # 根据当前的训练步骤数，找到对应的最大地形等级。
                max_allowed_level = level # 更新允许的最大地形等级，但不超过 terrain.max_terrain_level - 1，确保不会超过地形配置中定义的最大等级。
        max_allowed_level = max(
            0, min(int(max_allowed_level), terrain.max_terrain_level - 1) # 不能小于 0，也不能超过真实最高地形等级。
        )

    # 只有真正跑满 episode 的轨迹才允许升级，避免高等级地形过早堆上来。
    # 升级条件：走得够远 并且完整跑完 episode  并且当前等级还没到允许上限。
    timed_out = env.termination_manager.get_term("time_out")[env_ids] # 只有机器人坚持到了 episode 自然结束，才允许升级。
    current_levels = terrain.terrain_levels[env_ids] # 获取当前环境的地形等级，用于判断是否满足升级或降级的条件。
    move_up = (distance > promotion_distance) & timed_out # 如果机器人走的距离超过升级门槛，并且是正常跑完整局，就允许升级。
    move_up &= current_levels < max_allowed_level # 但如果当前等级已经达到最大允许等级，就不能再升。但如果当前等级已经达到最大允许等级，就不能再升。

    move_down = (
        distance
        < torch.norm(command[env_ids, :2], dim=1)
        * env.max_episode_length_s
        * demotion_command_ratio
    ) # 实际走过的距离 < 命令速度 * 最大 episode 时长 * 降级比例，说明机器人可能根本没动起来，或者被卡住了，这时考虑降级。
    
    move_down &= ~move_up # 如果满足升级条件，就不考虑降级，避免冲突。

    terrain.update_env_origins(env_ids, move_up, move_down) # 根据 move_up 和 move_down 更新这些环境的地形等级和起点。
    if max_level_schedule is not None:
        clamped_levels = torch.clamp(terrain.terrain_levels[env_ids], max=max_allowed_level)
        if torch.any(clamped_levels != terrain.terrain_levels[env_ids]):
            terrain.terrain_levels[env_ids] = clamped_levels
            terrain.env_origins[env_ids] = terrain.terrain_origins[
                terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]
            ] # 对每个 env，根据它自己的地形等级 terrain_levels 和地形类型 terrain_types，去 terrain_origins 这张表里查对应的起点坐标。
            # → 查找新等级、新类型对应的地形块位置,下次重置时把机器人放到那里
            # terrain.env_origins:当前每个 env 实际使用的起点.
            # terrain.terrain_origins:预定义的起点表，按照地形等级和地形类型分类好，每个格子里是对应等级和类型的起点坐标。

    # 计算一些统计信息，供 curriculum manager 观察和决策。
    levels = terrain.terrain_levels.float() # 当前地形等级的统计信息，每个environment的地形等级都可能不同，这些统计信息可以帮助 curriculum manager 了解整体训练进度和难度分布。
    result: dict[str, torch.Tensor] = {
        "mean": torch.mean(levels),
        "max": torch.max(levels),
        "max_allowed_level": torch.tensor(max_allowed_level, device=env.device),
        "promotion_distance": torch.tensor(promotion_distance, device=env.device),
        "move_up_rate": torch.mean(move_up.float()),
        "move_down_rate": torch.mean(move_down.float()),
    }
    # 如果 terrain_generator 里每个子地形都有单独的统计，就顺便算一下各个子地形的平均等级，供 curriculum manager 观察。
    sub_terrain_names = list(terrain_generator.sub_terrains.keys()) # 取出所有子地形名称：比如平地、阶梯、斜坡等不同类型的地形块，这些信息可以帮助 curriculum manager 了解不同类型地形的训练进度和难度分布。
    terrain_origins = terrain.terrain_origins # [地形难度等级数量, 地形类型数量, xyz坐标]，[num_levels, num_terrain_types, 3]
    assert terrain_origins is not None
    num_cols = terrain_origins.shape[1] # 地形类型数量，如果这个数量和 terrain_generator 里定义的子地形数量一致，就说明每个子地形都有对应的统计信息，可以计算各自的平均等级。
    if num_cols == len(sub_terrain_names): # 如果地形类型数量和子地形数量一致，说明每个子地形都有对应的统计信息，可以计算各自的平均等级。
        types = terrain.terrain_types # 取出每个环境当前属于哪种地形类型。
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

    # 动作：12 个关节的位置控制。
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = BPX_ACTION_SCALE

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


# 创建一个适合 BPX 机器人在崎岖地形上训练速度跟踪能力的强化学习环境配置。
def bpx_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_velocity_env_cfg() # 先拿一个默认的机器人速度控制任务模板，然后下面逐步改成 BPX 机器人 + 崎岖地形版本。

    # 仿真器参数：仿真接触参数需要设置得更大一些，以适应崎岖地形可能带来的更多接触点和约束。
    cfg.sim.njmax = 1500 # 设置 MuJoCo 仿真中的最大约束/关节相关数量。
    cfg.sim.nconmax = 192 # 设置 MuJoCo 仿真中的最大接触数量，崎岖地形可能会有更多的接触点。
    cfg.sim.contact_sensor_maxmatch = 128 # 设置接触传感器最多匹配多少个接触对象。
    cfg.sim.mujoco.ccd_iterations = 50 # 设置连续碰撞检测迭代次数。崎岖地形、台阶、快速运动时，碰撞检测更复杂，所以提高迭代次数有助于减少穿模或漏碰。

    # 开始设置场景中的实体，换成 BPX 机器人。这个函数会返回一个配置好的 BPX 机器人资产配置，我们直接用它替换掉默认的机器人配置。
    cfg.scene.entities = {
        "robot": get_bpx_robot_cfg(),
    }
    cfg.viewer.body_name = "torso" # Viewer 跟随 BPX 主躯干，确保在崎岖地形上训练时能更好地观察机器人的整体姿态和运动。
    cfg.viewer.distance = 2.5 # 适当拉远一些观察距离，以便在崎岖地形上更好地观察机器人和周围环境的交互。
    cfg.viewer.elevation = -10.0 # 保持较低的仰角，以更好地观察机器人在崎岖地形上的运动细节和地形特征。


    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "generator"
    terrain_generator = deepcopy(ROUGH_TERRAINS_CFG)
    terrain_generator.curriculum = True
    terrain_proportions = {
        "flat": 0.12,
        "pyramid_stairs": 0.14,
        "pyramid_stairs_inv": 0.08,
        "hf_pyramid_slope": 0.24,
        "hf_pyramid_slope_inv": 0.26,
        "random_rough": 0.08,
        "wave_terrain": 0.08,
    }
    for terrain_name, proportion in terrain_proportions.items():
        if terrain_name in terrain_generator.sub_terrains:
            terrain_generator.sub_terrains[terrain_name].proportion = proportion
    for terrain_name in ("pyramid_stairs", "pyramid_stairs_inv"):
        if terrain_name in terrain_generator.sub_terrains:
            stairs_cfg = terrain_generator.sub_terrains[terrain_name]
            stairs_cfg.step_width = 0.35
            stairs_cfg.step_height_range = (0.0, 0.08)
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

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = BPX_ACTION_SCALE

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
        cfg.rewards["track_angular_velocity"].weight = 3.0
        cfg.rewards["track_angular_velocity"].params["std"] = 0.35
    cfg.rewards["track_forward_velocity_fine"] = RewardTermCfg(
        func=_bpx_track_forward_velocity,
        weight=1.4,
        params={"command_name": "twist", "std": 0.25},
    )
    cfg.rewards["track_lateral_velocity_fine"] = RewardTermCfg(
        func=_bpx_track_lateral_velocity,
        weight=1.7,
        params={"command_name": "twist", "std": 0.12},
    )
    cfg.rewards["track_yaw_velocity_fine"] = RewardTermCfg(
        func=_bpx_track_yaw_velocity,
        weight=1.5,
        params={"command_name": "twist", "std": 0.22},
    )
    cfg.rewards["forward_lateral_drift"] = RewardTermCfg(
        func=_bpx_forward_lateral_drift,
        weight=-3.0,
        params={
            "command_name": "twist",
            "lateral_command_threshold": 0.08,
            "yaw_command_threshold": 0.08,
        },
    )
    cfg.rewards["forward_yaw_drift"] = RewardTermCfg(
        func=_bpx_forward_yaw_drift,
        weight=-1.8,
        params={
            "command_name": "twist",
            "lateral_command_threshold": 0.08,
            "yaw_command_threshold": 0.08,
        },
    )
    if "body_ang_vel" in cfg.rewards:
        cfg.rewards["body_ang_vel"].weight = -0.08
    if "angular_momentum" in cfg.rewards:
        cfg.rewards["angular_momentum"].weight = 0.0
    if "action_rate_l2" in cfg.rewards:
        cfg.rewards["action_rate_l2"].weight = -0.14
    if "air_time" in cfg.rewards:
        cfg.rewards["air_time"].weight = 0.25
    if "foot_clearance" in cfg.rewards:
        cfg.rewards["foot_clearance"].weight = -2.5
        cfg.rewards["foot_clearance"].params["target_height"] = 0.14
    if "foot_swing_height" in cfg.rewards:
        cfg.rewards["foot_swing_height"].weight = -0.35
        cfg.rewards["foot_swing_height"].params["target_height"] = 0.14
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
    cmd.rel_forward_envs = 0.85
    cmd.resampling_time_range = (6.0, 10.0)

    cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
        func=_bpx_terrain_levels_vel,
        params={
            "command_name": "twist",
            "promotion_distance_ratio": 0.9,
            "demotion_command_ratio": 0.55,
            "max_level_schedule": (
                (0, 2),
                (6000 * 16, 3),
                (12000 * 16, 4),
                (20000 * 16, 5),
                (30000 * 16, 6),
                (40000 * 16, 8),
                (46000 * 16, 9),
            ),
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
                    "lin_vel_x": (-0.25, 0.85),
                    "lin_vel_y": (-0.12, 0.12),
                    "ang_vel_z": (-0.30, 0.30),
                },
                {
                    "step": 12000 * 16,
                    "lin_vel_x": (-0.30, 1.05),
                    "lin_vel_y": (-0.16, 0.16),
                    "ang_vel_z": (-0.40, 0.40),
                },
                {
                    "step": 20000 * 16,
                    "lin_vel_x": (-0.45, 1.25),
                    "lin_vel_y": (-0.20, 0.20),
                    "ang_vel_z": (-0.45, 0.45),
                },
                {
                    "step": 30000 * 16,
                    "lin_vel_x": (-0.45, 1.45),
                    "lin_vel_y": (-0.24, 0.24),
                    "ang_vel_z": (-0.50, 0.50),
                },
                {
                    "step": 40000 * 16,
                    "lin_vel_x": (-0.50, 1.65),
                    "lin_vel_y": (-0.28, 0.28),
                    "ang_vel_z": (-0.55, 0.55),
                },
                {
                    "step": 45000 * 16,
                    "lin_vel_x": (-0.55, 1.80),
                    "lin_vel_y": (-0.30, 0.30),
                    "ang_vel_z": (-0.60, 0.60),
                },
            ],
        },
    )

    base_actor_terms = cfg.observations["actor"].terms
    base_critic_terms = cfg.observations["critic"].terms
    actor_term_names = (
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "actions",
        "command",
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
    critic_terms = {
        name: deepcopy(base_critic_terms[name])
        for name in (*actor_term_names, *privileged_term_names)
    }
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
            history_length=15,
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
