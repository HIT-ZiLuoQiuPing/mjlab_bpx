# BPX Model Artifacts

This repository intentionally tracks a small set of deployment artifacts even though
`*.pt` files and `logs/` are ignored by default.

## Real-Tested Model

- Path: `logs/rsl_rl/bpx_waq_rough/2026-06-18_18-43-25_dreamwaq/model_58799.pt`
- Status: deployed on the real robot by the user
- SHA256: `f1520aeca99d7f49a57197fd75fab602d66648d48c0862fc6527913ba5524377`

## Metric-Best Candidate

- Path: `logs/rsl_rl/bpx_waq_rough/2026-06-18_18-43-25_dreamwaq/model_57200.pt`
- Selection basis: best checkpoint by a simple score over mean reward, mean
  episode length, terrain level, velocity tracking errors, upright reward, and
  termination rates from the TensorBoard scalars.
- Key metrics:
  - `Train/mean_reward`: `122.3699`
  - `Train/mean_episode_length`: `979.4200`
  - `Curriculum/terrain_levels/mean`: `4.2246`
  - `Episode_Metrics/bpx_err_vx`: `0.0960`
  - `Episode_Metrics/bpx_err_vy`: `0.0641`
  - `Episode_Metrics/bpx_err_wz`: `0.1668`
  - `Episode_Termination/fell_over`: `0.0000`
  - `Episode_Termination/illegal_contact`: `0.0000`
  - `Episode_Termination/out_of_terrain_bounds`: `0.0000`
- SHA256: `5716d8767d1945e36cf598488f70e5cf51fd1fead1800cc4b44eb087ccf2d9bd`

## Exported TorchScript Policies

- `exported/bpx_dwaq_20260615_34000.pt`
  - SHA256: `78699bcf8acb113266d683202619a0ba42ecd8799a017c01f337dc451c85a89b`
- `exported/bpx_dwaq_20260616_190738_model_32600.pt`
  - SHA256: `9360c10c1003d63c0ba43572ac1d9312466891ea893335539111c32aff127480`
- `exported/bpx_dwaq_v3.pt`
  - SHA256: `317ddd339e8643d391eab432d787e52963a01d17a61f51147ddd6ea6d97ad590`
- `exported/waq.pt`
  - SHA256: `271bb14f71cf64e7c2e970d4051905ab3f0ec2525fb83abadd81602e9634de10`

The full TensorBoard event file for `2026-06-18_18-43-25_dreamwaq` is not tracked
because it is large and not needed for deployment.
