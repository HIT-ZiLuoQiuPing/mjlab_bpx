# BPX DreamWaQ Sim2Real 训练与部署说明

这份文档对应当前仓库的 `Mjlab-Velocity-Rough-BPX` 任务，以及上层部署包：

```text
/home/ubuntu/bpx_simreal_v6/bpx_simreal_v6_crouchfix_xwk
```

SDK 目录通常是：

```text
/home/ubuntu/bpx_sdk_open
```

当前训练端已经按上层部署合同对齐：

- actor 单帧观测：45 维
- DreamWaQ history：5 帧
- encoder 输入：225 维
- TorchScript 部署输入：225 或 270 维
- policy 输出：12 维
- action scale：0.25
- 默认站姿：hip roll `0.0`，hip pitch `0.8`，knee `-1.5`
- 最终 policy/sim2real PD：`kp=50.0`，`kd=0.8`
- 低增益站立/吊起首测 PD：`kp=6.0`，`kd=0.35`
- `leg_symmetry` 奖励已删除

## 1. 环境检查

进入训练工程：

```bash
cd /home/ubuntu/robot_rl
```

建议使用当前机器上的 `bpx-mjlab` 环境：

```bash
/home/ubuntu/miniconda3/envs/bpx-mjlab/bin/python -m mjlab.scripts.list_envs | grep BPX
```

应该能看到：

```text
Mjlab-Velocity-Flat-BPX
Mjlab-Velocity-Rough-BPX
```

如果看不到 BPX 任务，先重新安装本地包：

```bash
/home/ubuntu/miniconda3/envs/bpx-mjlab/bin/pip install -e bpx_mjlab
```

## 2. 训练模型

推荐直接训练 rough WAQ 任务：

```bash
cd /home/ubuntu/robot_rl
MPLCONFIGDIR=/tmp/mpl /home/ubuntu/miniconda3/envs/bpx-mjlab/bin/python -m mjlab.scripts.train Mjlab-Velocity-Rough-BPX
```

默认训练配置在：

```text
bpx_mjlab/src/bpx_mjlab/rl_cfg.py
```

关键默认值：

```text
max_iterations = 50000
num_steps_per_env = 16
save_interval = 200
experiment_name = bpx_waq_rough
run_name = dreamwaq
```

训练日志和 checkpoint 会写到：

```text
logs/rsl_rl/bpx_waq_rough/<日期>_dreamwaq/
```

常见 checkpoint 名字：

```text
model_200.pt
model_400.pt
...
model_50000.pt
```

注意：不要从旧 checkpoint resume。旧模型的 history 是 15 帧或观测/动作顺序不同，和现在的 5 帧 sim2real 合同不兼容。

## 3. Play 检查模型

先用零动作或随机动作检查环境：

```bash
MPLCONFIGDIR=/tmp/mpl /home/ubuntu/miniconda3/envs/bpx-mjlab/bin/python -m mjlab.scripts.play Mjlab-Velocity-Rough-BPX --agent zero --num-envs 1 --viewer native
```

播放训练好的 checkpoint：

```bash
MPLCONFIGDIR=/tmp/mpl /home/ubuntu/miniconda3/envs/bpx-mjlab/bin/python -m mjlab.scripts.play Mjlab-Velocity-Rough-BPX \
  --checkpoint-file logs/rsl_rl/bpx_waq_rough/<run_name>/model_<iter>.pt \
  --num-envs 1 \
  --viewer native
```

如果本机没有图形显示，用 viser：

```bash
MPLCONFIGDIR=/tmp/mpl /home/ubuntu/miniconda3/envs/bpx-mjlab/bin/python -m mjlab.scripts.play Mjlab-Velocity-Rough-BPX \
  --checkpoint-file logs/rsl_rl/bpx_waq_rough/<run_name>/model_<iter>.pt \
  --num-envs 1 \
  --viewer viser
```

Play 阶段重点看：

- 是否能稳定站住
- 小速度命令是否跟随
- 动作是否过大或抖动
- 是否频繁摔倒、侧偏、转圈
- 脚尖接触是否正常，非脚部碰地是否明显

## 4. 导出 sim2real TorchScript

训练 checkpoint 不能直接给上层 UI 用。上层 UI 需要 TorchScript policy。

使用本仓库提供的导出脚本：

```bash
cd /home/ubuntu/robot_rl
MPLCONFIGDIR=/tmp/mpl /home/ubuntu/miniconda3/envs/bpx-mjlab/bin/python bpx_mjlab/scripts/export_simreal_policy.py \
  --checkpoint logs/rsl_rl/bpx_waq_rough/<run_name>/model_<iter>.pt \
  --output exported/bpx_dwaq_v2.pt
```

如果要导出后直接复制到上层部署包：

```bash
cd /home/ubuntu/robot_rl
MPLCONFIGDIR=/tmp/mpl /home/ubuntu/miniconda3/envs/bpx-mjlab/bin/python bpx_mjlab/scripts/export_simreal_policy.py \
  --checkpoint logs/rsl_rl/bpx_waq_rough/<run_name>/model_<iter>.pt \
  --output exported/bpx_dwaq_v2.pt \
  --deploy-dir /home/ubuntu/bpx_simreal_v6/bpx_simreal_v6_crouchfix_xwk
```

这样会生成并复制：

```text
/home/ubuntu/bpx_simreal_v6/bpx_simreal_v6_crouchfix_xwk/policy/bpx_dwaq_v2.pt
```

导出后检查签名：

```bash
cd /home/ubuntu/bpx_simreal_v6/bpx_simreal_v6_crouchfix_xwk
bash scripts/07_check_policy_signature.sh policy/bpx_dwaq_v2.pt
```

期望至少看到：

```text
OK   history_only          input=225 -> output=(12,)
OK   current_plus_history  input=270 -> output=(12,)
```

`legacy_or_wrong input=480` 失败是正常的。

## 5. 模型输入

当前 policy 支持两种上层输入格式。

### history_only

```text
shape = [1, 225]
225 = 5 * 45
```

5 帧按时间顺序排列：

```text
frame_0, frame_1, frame_2, frame_3, frame_4
```

`frame_4` 是当前最新帧。导出 wrapper 会自动从最后一帧取 current obs。

### current_plus_history

```text
shape = [1, 270]
270 = 45 + 225
```

排列为：

```text
current_obs_45 + history_225
```

上层 UI 会自动 dry-run 检测 225 和 270，哪个能跑通就用哪个。

### 单帧 45 维 obs 顺序

每一帧都是：

```text
0:3    base angular velocity * 0.25
3:6    projected gravity
6:9    command [vx, vy, wz] * [2.0, 2.0, 0.25]
9:21   joint_pos - default_joint_pos
21:33  joint_vel * 0.05
33:45  last_action
```

12 个关节顺序固定为：

```text
0  fl_hip_roll_joint
1  fr_hip_roll_joint
2  hl_hip_roll_joint
3  hr_hip_roll_joint
4  fl_hip_pitch_joint
5  fr_hip_pitch_joint
6  hl_hip_pitch_joint
7  hr_hip_pitch_joint
8  fl_knee_joint
9  fr_knee_joint
10 hl_knee_joint
11 hr_knee_joint
```

注意：BPX SDK 的数组顺序通常是按腿分组，不是这个 policy 顺序。上层部署包通过 `configs/real_config_working.yaml` 里的 `calibration.joints.<joint>.sdk_index` 做转换，不要直接把 action[0:12] 当 SDK pos[0:12]。

## 6. 模型输出

policy 输出：

```text
shape = [1, 12]
```

输出顺序和上面的 policy joint order 完全一致。

训练/严格 sim2sim 公式：

```text
sim_action = clip(policy_action, -100.0, 100.0)
q_des_sim = default_joint_pos + sim_action * 0.25
```

默认关节位置：

```text
hip_roll:  0.0
hip_pitch: 0.8
knee:     -1.5
```

真机 UI 里还有额外安全保护：

```text
sent_action = clip(sim_action, -safe_guard_clip, safe_guard_clip)
q_des_sim_sent = default_joint_pos + sent_action * 0.25
```

首测必须开 safe guard，不要一上来严格 replay。

## 7. 与 SDK 和上层部署文件联动

### 7.1 确认部署包

进入上层部署包：

```bash
cd /home/ubuntu/bpx_simreal_v6/bpx_simreal_v6_crouchfix_xwk
```

目录里关键文件：

```text
configs/real_config_working.yaml       真机标定和 policy contract
policy/bpx_dwaq_v2.pt                  UI 默认加载的 policy
bridge_cpp/bpx_joint_bridge.cpp        C++ SDK bridge
app/xwk_joint_lab_ui.py                上层 UI
scripts/01c_check_sdk_layout.sh        检查 SDK 文件
scripts/02_build_bridge_real.sh        编译真机 bridge
scripts/03_run_bridge_real.sh          运行真机 bridge
scripts/04_run_ui.sh                   启动 UI
scripts/07_check_policy_signature.sh   检查 TorchScript 输入输出
scripts/08_check_sim2sim_contract.sh   检查 sim2sim 合同
```

### 7.2 确认 SDK

部署包默认找：

```text
third_party/bpx_sdk_open
```

检查：

```bash
bash scripts/01c_check_sdk_layout.sh
```

如果缺 SDK，而 `/home/ubuntu/bpx_sdk_open` 存在，可以把 SDK 放到：

```text
/home/ubuntu/bpx_simreal_v6/bpx_simreal_v6_crouchfix_xwk/third_party/bpx_sdk_open
```

可以复制，也可以做软链接：

```bash
cd /home/ubuntu/bpx_simreal_v6/bpx_simreal_v6_crouchfix_xwk
mkdir -p third_party
ln -s /home/ubuntu/bpx_sdk_open third_party/bpx_sdk_open
```

如果 `third_party/bpx_sdk_open` 已经存在，就不要重复创建软链接。

### 7.3 复制 policy

推荐用导出脚本自动复制：

```bash
cd /home/ubuntu/robot_rl
MPLCONFIGDIR=/tmp/mpl /home/ubuntu/miniconda3/envs/bpx-mjlab/bin/python bpx_mjlab/scripts/export_simreal_policy.py \
  --checkpoint logs/rsl_rl/bpx_waq_rough/<run_name>/model_<iter>.pt \
  --deploy-dir /home/ubuntu/bpx_simreal_v6/bpx_simreal_v6_crouchfix_xwk
```

复制完成后，上层包里应该有：

```text
policy/bpx_dwaq_v2.pt
```

检查 policy：

```bash
cd /home/ubuntu/bpx_simreal_v6/bpx_simreal_v6_crouchfix_xwk
bash scripts/07_check_policy_signature.sh policy/bpx_dwaq_v2.pt
```

### 7.4 检查 sim2sim 合同

```bash
cd /home/ubuntu/bpx_simreal_v6/bpx_simreal_v6_crouchfix_xwk
bash scripts/08_check_sim2sim_contract.sh
```

需要重点确认：

- joint order OK
- default_joint_pos OK
- action_scale = 0.25
- sim_pd.kp = 50.0
- sim_pd.kd = 0.8
- obs scales OK
- SDK index mapping OK
- URDF limits OK

只要这里有红项，就不要上真机跑 policy。

### 7.5 编译 bridge

```bash
cd /home/ubuntu/bpx_simreal_v6/bpx_simreal_v6_crouchfix_xwk
bash scripts/02_build_bridge_real.sh
```

成功后会生成：

```text
build_bridge_real/bpx_joint_bridge
```

### 7.6 启动真机 bridge

机器人上电、网络连通后运行：

```bash
cd /home/ubuntu/bpx_simreal_v6/bpx_simreal_v6_crouchfix_xwk
bash scripts/03_run_bridge_real.sh 10.21.40.1
```

如果机器人 IP 是 `192.168.0.1`：

```bash
bash scripts/03_run_bridge_real.sh 192.168.0.1
```

脚本参数顺序是：

```text
03_run_bridge_real.sh <robot_ip> <ui_port> <hz> <max_roll> <max_pitch>
```

默认：

```text
port = 8765
hz = 50
max_roll = 0.80
max_pitch = 0.80
watchdog = 0.25s
```

bridge 的作用：

- 从 BPX SDK 读取关节、速度、IMU
- 按 `configs/real_config_working.yaml` 把 raw joint 转成 sim joint
- 接收 UI 发来的 q_des raw 命令
- 执行 watchdog 和姿态保护

### 7.7 启动 UI

另开一个终端：

```bash
cd /home/ubuntu/bpx_simreal_v6/bpx_simreal_v6_crouchfix_xwk
bash scripts/04_run_ui.sh
```

UI 默认连接：

```text
host = 127.0.0.1
port = 8765
```

如果 bridge 和 UI 不在同一台机器，host 填 bridge 所在机器 IP。

### 7.8 UI 真机安全流程

按这个顺序做：

```text
1. Connect
2. 确认 sdk_ok=True
3. 确认机器人吊起或有支撑
4. Start 2s IMU Calibration
5. ARM real joint command
6. Low-Gain Standard Stand
7. 确认 PD stand 稳定
8. Load Policy: policy/bpx_dwaq_v2.pt
9. START POLICY
```

首测建议参数：

```text
Policy action guard: ON
Policy action clip: 0.05 ~ 0.12
Rate limit: ON
Policy Kp: 6.0
Policy Kd: 0.35
IMU filter alpha: 0.15
Command smooth alpha: 0.12 ~ 0.15
vx limit: 0.10
vy limit: 0.05
yaw limit: 0.25
```

这里的 `Policy Kp=6.0, Policy Kd=0.35` 是低增益吊起/首测参数，不是最终可用增益。确认关节映射、IMU 方向、动作符号和 watchdog 都正确之后，再逐步提高到实测可用区间：

```text
Policy Kp: 6 -> 8 -> 10 -> 15 -> 20 -> 30 -> 40 -> 50
Policy Kd: 0.35 -> 0.45 -> 0.6 -> 0.8
Policy action clip: 0.05/0.08/0.12 先保守，再根据吊起响应放开
```

当前训练侧最终合同按 `Kp=50.0, Kd=0.8` 重新训练；如果真机最终只在 `Kp=50` 左右能站稳，部署侧 `sim_pd` 也要同步成 `50.0/0.8`。

如果抖动：

```text
Kp: 6 -> 4 -> 3
Kd: 0.35 -> 0.45
Action clip: 0.12 -> 0.08 -> 0.05
IMU alpha: 0.15 -> 0.08
Command alpha: 0.12 -> 0.08
```

不要直接落地首测。先吊起或支撑，确认方向、关节映射、IMU 和动作符号都正确。

### 7.9 严格 sim2sim 对照

只在吊起或支撑状态下做：

```text
safe_guard OFF
rate_limit OFF
Kp = 50
Kd = 0.8
```

这样更接近训练/仿真合同，但真机风险更高，不适合第一次直接落地。

### 7.10 急停

UI 里可以：

```text
SPACE
EMERGENCY / DAMPING
Stop Policy
```

bridge 也有 watchdog。UI 停止发送后，bridge 会在 watchdog 超时后保护。

## 8. 推荐完整流程

完整流程按这个顺序：

```bash
# 1. 训练
cd /home/ubuntu/robot_rl
MPLCONFIGDIR=/tmp/mpl /home/ubuntu/miniconda3/envs/bpx-mjlab/bin/python -m mjlab.scripts.train Mjlab-Velocity-Rough-BPX

# 2. Play 检查 checkpoint
MPLCONFIGDIR=/tmp/mpl /home/ubuntu/miniconda3/envs/bpx-mjlab/bin/python -m mjlab.scripts.play Mjlab-Velocity-Rough-BPX \
  --checkpoint-file logs/rsl_rl/bpx_waq_rough/<run_name>/model_<iter>.pt \
  --num-envs 1 \
  --viewer native

# 3. 导出并复制到部署包
MPLCONFIGDIR=/tmp/mpl /home/ubuntu/miniconda3/envs/bpx-mjlab/bin/python bpx_mjlab/scripts/export_simreal_policy.py \
  --checkpoint logs/rsl_rl/bpx_waq_rough/<run_name>/model_<iter>.pt \
  --deploy-dir /home/ubuntu/bpx_simreal_v6/bpx_simreal_v6_crouchfix_xwk

# 4. 检查部署 policy 和 contract
cd /home/ubuntu/bpx_simreal_v6/bpx_simreal_v6_crouchfix_xwk
bash scripts/07_check_policy_signature.sh policy/bpx_dwaq_v2.pt
bash scripts/08_check_sim2sim_contract.sh

# 5. 编译并启动 bridge
bash scripts/02_build_bridge_real.sh
bash scripts/03_run_bridge_real.sh <robot_ip>

# 6. 另开终端启动 UI
cd /home/ubuntu/bpx_simreal_v6/bpx_simreal_v6_crouchfix_xwk
bash scripts/04_run_ui.sh
```

然后在 UI 里按安全流程启动。

## 9. 常见问题

### policy 加载失败

先跑：

```bash
bash scripts/07_check_policy_signature.sh policy/bpx_dwaq_v2.pt
```

如果 225/270 都失败，通常是拿了训练 checkpoint `model_*.pt`，而不是导出的 TorchScript。

### 动作方向不对

不要先改 policy。先检查：

```bash
bash scripts/08_check_sim2sim_contract.sh
```

再检查 `configs/real_config_working.yaml` 里的：

```text
calibration.joints.<joint>.scale
calibration.joints.<joint>.offset
calibration.joints.<joint>.sdk_index
```

### 站姿不对

确认部署配置里的 default joint pos 是：

```text
hip_roll = 0.0
hip_pitch = 0.8
knee = -1.5
```

训练端和部署端必须一致。

### 观测维度不对

当前合同是：

```text
actor_obs_single_frame_dim = 45
dwaq_history_length = 5
encoder_input_dim = 225
supported_policy_input_dims = 225 / 270
```

如果看到 675、720、480 之类，是旧模型或旧部署合同。
