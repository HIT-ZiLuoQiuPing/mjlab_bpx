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
    # 可视化视角
    cfg.viewer.body_name = "torso" # Viewer 跟随 BPX 主躯干，确保在崎岖地形上训练时能更好地观察机器人的整体姿态和运动。
    cfg.viewer.distance = 2.5 # 适当拉远一些观察距离，以便在崎岖地形上更好地观察机器人和周围环境的交互。
    cfg.viewer.elevation = -10.0 # 保持较低的仰角，以更好地观察机器人在崎岖地形上的运动细节和地形特征。

    # 地形配置：使用生成器生成崎岖地形，配置不同类型地形的比例，并调整一些特定地形的参数，以提供多样化的训练环境，帮助机器人学习在不同崎岖地形上的速度跟踪能力。
    assert cfg.scene.terrain is not None # 确认地形配置存在。
    cfg.scene.terrain.terrain_type = "generator" # 设置地形类型为生成器。地形不是固定一个平面，而是由程序生成不同类型的地形。
    terrain_generator = deepcopy(ROUGH_TERRAINS_CFG) # 复制一份崎岖地形配置
    terrain_generator.curriculum = True # 启用地形课程学习。
    terrain_proportions = { 
        "flat": 0.12,
        "pyramid_stairs": 0.16,
        "pyramid_stairs_inv": 0.10,
        "hf_pyramid_slope": 0.24,
        "hf_pyramid_slope_inv": 0.23,
        "random_rough": 0.08,
        "wave_terrain": 0.07,
    }
    #底下这些修改一些地形的参数，都是 safe 的，不存在就跳过，不会报错。
    for terrain_name, proportion in terrain_proportions.items():
        if terrain_name in terrain_generator.sub_terrains:
            terrain_generator.sub_terrains[terrain_name].proportion = proportion # 如果地形生成器里确实有这种地形：就把它的生成比例设置成指定值。这样可以避免某些地形名字不存在时报错。
    for terrain_name in ("pyramid_stairs", "pyramid_stairs_inv"):
        if terrain_name in terrain_generator.sub_terrains:
            stairs_cfg = terrain_generator.sub_terrains[terrain_name]
            stairs_cfg.step_width = 0.35
            stairs_cfg.step_height_range = (0.04, 0.16)
    for terrain_name in ("hf_pyramid_slope", "hf_pyramid_slope_inv"):
        if terrain_name in terrain_generator.sub_terrains:
            terrain_generator.sub_terrains[terrain_name].slope_range = (0.0, 0.85)
    cfg.scene.terrain.terrain_generator = terrain_generator # 把配置好的崎岖地形生成器写回环境配置。
    cfg.scene.terrain.max_init_terrain_level = 1 # 刚开始先限制最大地形等级，避免一开始就生成太难的地形块。
    cfg.scene.extent = 3.0 # 环境范围适当扩大一些，以适应崎岖地形可能需要更多的空间来生成不同的地形块。

    # 传感器配置：添加脚部接触传感器和非脚部碰地传感器，以便在崎岖地形上更好地感知与地面的交互，同时删除默认的 raycast 传感器，因为它们可能不适合崎岖地形的训练需求。
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan": # 地形传感器 
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

    # 动作配置：保持使用关节位置控制，但调整动作缩放以适应 BPX 机器人的运动范围和崎岖地形上的控制需求。
    joint_pos_action = cfg.actions["joint_pos"] # 继续使用关节位置控制，因为它通常更适合复杂地形上的精细控制，同时调整动作缩放因子。
    assert isinstance(joint_pos_action, JointPositionActionCfg) # 确认这个动作项确实是关节位置动作配置
    joint_pos_action.scale = BPX_ACTION_SCALE # 设置动作缩放

    # 随机事件配置：事件通常用于 domain randomization，也就是训练时随机改变物理参数，让机器人更鲁棒。
    if "foot_friction" in cfg.events:
        _safe_set_asset_names(
            cfg.events["foot_friction"],
            "geom_names",
            FOOT_GEOMS,
        )# 把默认事件里可能存在的 Go1/G1 的脚部几何体名字替换成 BPX 的脚部几何体名字，确保这个事件能正确地作用在 BPX 机器人的脚部。
    if "base_com" in cfg.events:
        _safe_set_asset_names(
            cfg.events["base_com"],
            "body_names",
            ("torso",),
        )# 把默认事件里可能存在的 Go1/G1 的躯干名字替换成 BPX 的躯干名字，确保这个事件能正确地作用在 BPX 机器人的躯干。

    # 姿态奖励配置：调整姿态奖励的标准差参数，使其适应 BPX 机器人的运动范围和崎岖地形上的控制需求，鼓励机器人在不同运动状态下保持合适的姿态，同时允许一定的灵活性以适应复杂地形。
    if "pose" in cfg.rewards:# 调整姿态奖励的标准差参数，使其适应 BPX 机器人的运动范围和崎岖地形上的控制需求。
        cfg.rewards["pose"].params["std_standing"] = {
            ".*_hip_roll_joint": 0.05,
            ".*_hip_pitch_joint": 0.10,
            ".*_knee_joint": 0.10,
        }
        cfg.rewards["pose"].params["std_walking"] = {
            ".*_hip_roll_joint": 0.30,
            ".*_hip_pitch_joint": 0.30,
            ".*_knee_joint": 0.60, # 站立时膝盖比较笔直，走路时膝盖可以弯曲一些，跑步时膝盖可以弯曲更多，所以 std_running 比 std_walking 要大一些。
        }
        cfg.rewards["pose"].params["std_running"] = {
            ".*_hip_roll_joint": 0.30,
            ".*_hip_pitch_joint": 0.30,
            ".*_knee_joint": 0.60,
        }
    if "upright" in cfg.rewards: # 把默认奖励里可能存在的 Go1/G1 的躯干名字替换成 BPX 的躯干名字，确保这个奖励能正确地作用在 BPX 机器人的躯干。
        _safe_set_asset_names(cfg.rewards["upright"], "body_names", ("torso",))
        cfg.rewards["upright"].params["terrain_sensor_names"] = ("terrain_scan",)
    if "body_ang_vel" in cfg.rewards:
        _safe_set_asset_names(cfg.rewards["body_ang_vel"], "body_names", ("torso",))
    for reward_name in ("foot_clearance", "foot_slip"):
        if reward_name in cfg.rewards:
            _safe_set_asset_names(cfg.rewards[reward_name], "site_names", FOOT_SITES)

    if "track_linear_velocity" in cfg.rewards:
        cfg.rewards["track_linear_velocity"].weight = 2.0
        cfg.rewards["track_linear_velocity"].params["std"] = 0.5
    if "track_angular_velocity" in cfg.rewards:
        cfg.rewards["track_angular_velocity"].weight = 2.0
        cfg.rewards["track_angular_velocity"].params["std"] = 2**0.5 / 2
    if "body_ang_vel" in cfg.rewards:
        cfg.rewards["body_ang_vel"].weight = 0.0
    if "angular_momentum" in cfg.rewards:
        cfg.rewards["angular_momentum"].weight = 0.0
    if "action_rate_l2" in cfg.rewards:
        cfg.rewards["action_rate_l2"].weight = -0.1
    if "air_time" in cfg.rewards:
        cfg.rewards["air_time"].weight = 0.0
    if "foot_clearance" in cfg.rewards:
        cfg.rewards["foot_clearance"].weight = 0.0
        cfg.rewards["foot_clearance"].params["target_height"] = 0.12
    if "foot_swing_height" in cfg.rewards:
        cfg.rewards["foot_swing_height"].weight = 0.0
        cfg.rewards["foot_swing_height"].params["target_height"] = 0.12
    cfg.rewards["calf_ground_touch"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-0.1,
        params={"sensor_name": calf_ground_cfg.name},
    )
    cfg.rewards["termination"] = RewardTermCfg( # 终止惩罚，摔倒了就给个大负奖励。
        func=mdp.is_terminated,
        weight=-25.0,
    )

    # 终止条件配置：添加一个新的终止条件，当机器人与地面发生非法接触时（即非脚部接触），就终止 episode，以鼓励机器人避免摔倒或与地面发生不安全的接触。
    cfg.terminations["illegal_contact"] = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_name": dangerous_ground_cfg.name},
    )

    # 命令空间配置
    cmd = cfg.commands["twist"]
    assert isinstance(cmd, UniformVelocityCommandCfg) # 均匀采样速度命令配置
    cmd.viz.z_offset = 0.5
    cmd.ranges.lin_vel_x = (-0.40, 0.90) # 前向速度范围，保持和 curriculum manager 里设置的初始范围一致，后续会逐步放宽。
    cmd.ranges.lin_vel_y = (-0.30, 0.30) # 侧向速度从早期就保留足够覆盖，否则策略容易把横向误差当成扰动而不是控制目标。
    cmd.ranges.ang_vel_z = (-0.40, 0.40) # yaw 命令从早期就参与训练，避免后期才学习转向导致直行 yaw 偏移修不回来。
    cmd.rel_standing_envs = 0.05 # 约 5% 的环境是站立命令，也就是机器人被要求不动。
    cmd.rel_forward_envs = 0.35  # 降低纯前向命令占比，让策略像 PPO 平地任务一样充分学习横向和 yaw 跟踪。
    cmd.resampling_time_range = (3.0, 8.0) # 与 PPO 平地任务保持一致，避免长时间固定命令把偏航漂移固化成习惯。

    # 地形课程学习配置：设置一个基于机器人与环境原点距离的地形课程学习机制，根据训练进度动态调整允许的最大地形等级，同时确保只有在机器人坚持跑完整局并且走得够远的情况下才允许升级地形，避免过早引入高难度地形，同时也提供了降级机制以防止机器人被卡住或根本没动起来。
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
    # 命令速度课程学习
    cfg.curriculum["command_vel"] = CurriculumTermCfg(
        func=mdp.commands_vel,
        params={
            "command_name": "twist",
            "velocity_stages": [
                {
                    "step": 0,
                    "lin_vel_x": (-0.40, 0.90),
                    "lin_vel_y": (-0.30, 0.30),
                    "ang_vel_z": (-0.40, 0.40),
                },
                {
                    "step": 6000 * 16,
                    "lin_vel_x": (-0.50, 1.00),
                    "lin_vel_y": (-0.35, 0.35),
                    "ang_vel_z": (-0.45, 0.45),
                },
                {
                    "step": 12000 * 16,
                    "lin_vel_x": (-0.60, 1.10),
                    "lin_vel_y": (-0.45, 0.45),
                    "ang_vel_z": (-0.50, 0.50),
                },
                {
                    "step": 20000 * 16,
                    "lin_vel_x": (-0.80, 1.30),
                    "lin_vel_y": (-0.55, 0.55),
                    "ang_vel_z": (-0.55, 0.55),
                },
                {
                    "step": 30000 * 16,
                    "lin_vel_x": (-1.00, 1.50),
                    "lin_vel_y": (-0.70, 0.70),
                    "ang_vel_z": (-0.60, 0.60),
                },
                {
                    "step": 40000 * 16,
                    "lin_vel_x": (-1.00, 1.80),
                    "lin_vel_y": (-0.80, 0.80),
                    "ang_vel_z": (-0.70, 0.70),
                },
                {
                    "step": 46000 * 16,
                    "lin_vel_x": (-1.00, 1.80),
                    "lin_vel_y": (-1.00, 1.00),
                    "ang_vel_z": (-0.70, 0.70),
                },
            ],
        },
    )

    #观察空间配置：根据 BPX 机器人的传感器配置和崎岖地形的训练需求，重新定义 actor 和 critic 的观察空间，确保它们包含足够的信息来支持在崎岖地形上学习速度跟踪能力，同时考虑到 play 模式下的特殊需求（比如去掉一些容易干扰的观测项）。
    base_actor_terms = cfg.observations["actor"].terms # 取出原始 actor 观察项。actor 就是策略网络，负责根据观察输出动作。
    base_critic_terms = cfg.observations["critic"].terms # 取出原始 critic 观察项。critic 就是价值网络，负责根据观察输出状态值或者优势函数。
    #定义actor能看到的内容
    actor_term_names = (
        "base_ang_vel", # 机身角速度。机器人可以感知自己是否在晃、在转。
        "projected_gravity", # 投影到机身坐标系下的重力向量。机器人可以感知自己当前的姿态，比如是站着、趴着还是侧躺。
        "joint_pos", # 关节位置。机器人可以感知自己当前的关节配置，比如腿是弯曲还是伸直。
        "joint_vel", # 关节速度。机器人可以感知自己关节的运动状态，比如腿是在抬起还是放下。
        "actions", # 上一步的动作。机器人可以感知自己上一步的动作输入，帮助它了解自己的运动趋势和惯性。这能帮助策略输出更连续的动作。
        "command", # 当前的速度命令。机器人可以感知自己被要求达到什么样的速度，帮助它根据命令调整自己的运动。
    )
    # 定义 critic 的特权观察
    privileged_term_names = (
        "base_lin_vel", # 机身线速度。真实机器人上这个值可能不容易直接准确获得，所以不给 actor，但训练时可以给 critic。
        "height_scan", # 地形高度扫描。崎岖地形上地面高低不平，知道前方地形的高度信息对评估状态很有帮助，但这个信息可能不太现实，所以只给 critic。
        "foot_height", # 脚部高度。知道脚离地面的高度对评估状态也很有帮助，但这个信息可能不太现实，所以只给 critic。
        "foot_air_time", # 脚部离地时间。知道脚部离地的时间对评估状态也很有帮助，但这个信息可能不太现实，所以只给 critic。
        "foot_contact", # 脚部接触地面。知道脚部是否接触地面对评估状态也很有帮助，但这个信息可能不太现实，所以只给 critic。
        "foot_contact_forces", # 脚部接触力。知道脚部接触地面时的力对评估状态也很有帮助，但这个信息可能不太现实，所以只给 critic。
    )

    actor_terms = { # 构造 actor 观察字典
        name: deepcopy(base_actor_terms[name])
        for name in actor_term_names
    }
    critic_terms = { # 构造 critic 观察字典，包含 actor 的观察项 + 特权观察项，因为 critic 可以看到更多的信息来更准确地评估状态，但 actor 只能看到有限的信息来输出动作。
        name: deepcopy(base_critic_terms[name])
        for name in (*actor_term_names, *privileged_term_names)
    }
    estimator_target_terms = { # 定义估计器目标。这可能用于辅助学习，比如让网络从历史观察中估计真实线速度和地形高度。
        "base_lin_vel": deepcopy(base_critic_terms["base_lin_vel"]), # 真实线速度，critic 可以看到，actor 看不到，估计器目标里也包含，让网络学会从历史观察中估计这个值。
        "height_scan": deepcopy(base_critic_terms["height_scan"]), # 地形高度扫描，critic 可以看到，actor 看不到，估计器目标里也包含，让网络学会从历史观察中估计这个值。
    }

    # 用新的观察配置替换原来的观察配置。
    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms, # 使用前面定义的 actor_terms。
            concatenate_terms=True, # 把多个观察项拼接成一个向量。策略网络通常需要一个扁平向量输入。
            enable_corruption=not play, # 训练时启用观测噪声，测试时关闭
            nan_policy="sanitize", # 如果观察里出现 NaN，就进行清理。
        ),
        "actor_history": ObservationGroupCfg(
            terms=deepcopy(actor_terms),
            concatenate_terms=True,
            enable_corruption=not play,
            history_length=15, # 保留最近 15 帧历史。这很重要，因为单帧观察可能不足以判断速度、运动趋势和接触状态。
            flatten_history_dim=True, #把历史维度压平成一个长向量
            nan_policy="sanitize",
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False, # critic 的观察不加噪声，因为它主要用于评估状态值，过多的噪声可能会干扰训练稳定性。
            nan_policy="sanitize",
        ),
        "estimator_target": ObservationGroupCfg(
            terms=estimator_target_terms, # 估计器目标观察项，主要用于辅助学习，让网络学会从历史观察中估计一些关键的状态信息。
            concatenate_terms=True,
            enable_corruption=False, # 估计器目标的观察不加噪声，因为它们是用来训练网络去估计的，如果这些目标本身就有噪声，可能会干扰训练。
            nan_policy="sanitize",
        ),
    }

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.events.pop("push_robot", None)
        cmd.ranges.lin_vel_x = (-0.55, 1.80)
        cmd.ranges.lin_vel_y = (-0.30, 0.30)
        cmd.ranges.ang_vel_z = (-0.60, 0.60)
        cfg.curriculum.pop("command_vel", None)

    return cfg
