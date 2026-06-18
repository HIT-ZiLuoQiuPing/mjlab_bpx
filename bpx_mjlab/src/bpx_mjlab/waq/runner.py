from __future__ import annotations

import pathlib
import statistics
import time

from mjlab.rl.runner import MjlabOnPolicyRunner
import torch
from rsl_rl.env import VecEnv
from rsl_rl.utils.logger import Logger


class BpxCompactLogger(Logger):
    """Keep TensorBoard logs complete while making stdout BPX-focused."""

    _CONSOLE_EXTRA_KEYS = (
        "Curriculum/terrain_levels/mean",
        "Curriculum/terrain_levels/max",
        "Curriculum/terrain_levels/move_up_rate",
        "Curriculum/terrain_levels/move_down_rate",
        "Curriculum/command_vel/lin_vel_x_max",
        "Curriculum/command_vel/lin_vel_y_max",
        "Curriculum/command_vel/ang_vel_z_max",
        "Episode_Metrics/bpx_cmd_vx",
        "Episode_Metrics/bpx_cmd_vy",
        "Episode_Metrics/bpx_cmd_wz",
        "Episode_Metrics/bpx_vel_vx",
        "Episode_Metrics/bpx_vel_vy",
        "Episode_Metrics/bpx_vel_wz",
        "Episode_Metrics/bpx_err_vx",
        "Episode_Metrics/bpx_err_vy",
        "Episode_Metrics/bpx_err_wz",
        "Metrics/twist/error_vel_xy",
        "Metrics/twist/error_vel_yaw",
        "Episode_Reward/track_linear_velocity",
        "Episode_Reward/track_lateral_velocity",
        "Episode_Reward/track_yaw_velocity",
        "Episode_Reward/upright",
        "Episode_Termination/fell_over",
        "Episode_Termination/illegal_contact",
        "Episode_Termination/time_out",
    )

    _LABELS = {
        "Curriculum/terrain_levels/mean": "terrain mean",
        "Curriculum/terrain_levels/max": "terrain max",
        "Curriculum/terrain_levels/move_up_rate": "terrain up",
        "Curriculum/terrain_levels/move_down_rate": "terrain down",
        "Curriculum/command_vel/lin_vel_x_max": "cmd range vx+",
        "Curriculum/command_vel/lin_vel_y_max": "cmd range vy+",
        "Curriculum/command_vel/ang_vel_z_max": "cmd range wz+",
        "Episode_Metrics/bpx_cmd_vx": "cmd vx",
        "Episode_Metrics/bpx_cmd_vy": "cmd vy",
        "Episode_Metrics/bpx_cmd_wz": "cmd wz",
        "Episode_Metrics/bpx_vel_vx": "vel vx",
        "Episode_Metrics/bpx_vel_vy": "vel vy",
        "Episode_Metrics/bpx_vel_wz": "vel wz",
        "Episode_Metrics/bpx_err_vx": "err vx",
        "Episode_Metrics/bpx_err_vy": "err vy",
        "Episode_Metrics/bpx_err_wz": "err wz",
        "Metrics/twist/error_vel_xy": "err xy",
        "Metrics/twist/error_vel_yaw": "err yaw",
        "Episode_Reward/track_linear_velocity": "rew track xy",
        "Episode_Reward/track_lateral_velocity": "rew track y",
        "Episode_Reward/track_yaw_velocity": "rew track yaw",
        "Episode_Reward/upright": "rew upright",
        "Episode_Termination/fell_over": "term fell",
        "Episode_Termination/illegal_contact": "term contact",
        "Episode_Termination/time_out": "term timeout",
    }

    def log(
        self,
        it: int,
        start_it: int,
        total_it: int,
        collect_time: float,
        learn_time: float,
        loss_dict: dict,
        learning_rate: float,
        action_std: torch.Tensor,
        rnd_weight: float | None,
        print_minimal: bool = False,
        width: int = 80,
        pad: int = 24,
    ) -> None:
        if self.writer is None:
            return

        collection_size = self.cfg["num_steps_per_env"] * self.num_envs * self.gpu_world_size
        iteration_time = collect_time + learn_time
        self.tot_timesteps += collection_size
        self.tot_time += iteration_time

        extra_values: dict[str, float] = {}
        if self.ep_extras:
            extra_keys = sorted({key for ep_info in self.ep_extras for key in ep_info})
            for key in extra_keys:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in self.ep_extras:
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                if infotensor.numel() == 0:
                    continue
                value = torch.mean(infotensor)
                tag = key if "/" in key else "Episode/" + key
                self.writer.add_scalar(tag, value, it)  # type: ignore
                extra_values[tag] = float(value.detach().cpu().item())

        for key, value in loss_dict.items():
            self.writer.add_scalar(f"Loss/{key}", value, it)
        self.writer.add_scalar("Loss/learning_rate", learning_rate, it)
        self.writer.add_scalar("Policy/mean_std", action_std.mean().item(), it)

        fps = int(collection_size / iteration_time)
        self.writer.add_scalar("Perf/total_fps", fps, it)
        self.writer.add_scalar("Perf/collection_time", collect_time, it)
        self.writer.add_scalar("Perf/learning_time", learn_time, it)

        mean_reward = None
        mean_episode_length = None
        if len(self.rewbuffer) > 0:
            if self.cfg["algorithm"]["rnd_cfg"]:
                self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(self.erewbuffer), it)
                self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(self.irewbuffer), it)
                self.writer.add_scalar("Rnd/weight", rnd_weight, it)  # type: ignore
            mean_reward = statistics.mean(self.rewbuffer)
            mean_episode_length = statistics.mean(self.lenbuffer)
            self.writer.add_scalar("Train/mean_reward", mean_reward, it)
            self.writer.add_scalar("Train/mean_episode_length", mean_episode_length, it)
            if self.logger_type != "wandb":
                self.writer.add_scalar("Train/mean_reward/time", mean_reward, int(self.tot_time))
                self.writer.add_scalar("Train/mean_episode_length/time", mean_episode_length, int(self.tot_time))

        log_lines = [
            "#" * width,
            f"\033[1m{f' BPX iter {it}/{total_it} '.center(width)}\033[0m",
        ]
        run_name = self.cfg.get("run_name")
        if run_name:
            log_lines.append(f"{'run':>{pad}} {run_name}")
        log_lines.extend(
            [
                f"{'steps':>{pad}} {self.tot_timesteps}",
                f"{'fps':>{pad}} {fps}",
                f"{'lr':>{pad}} {learning_rate:.3g}",
                f"{'action std':>{pad}} {action_std.mean().item():.3f}",
            ]
        )
        if mean_reward is not None and mean_episode_length is not None:
            log_lines.extend(
                [
                    f"{'mean reward':>{pad}} {mean_reward:.2f}",
                    f"{'mean ep len':>{pad}} {mean_episode_length:.2f}",
                ]
            )

        for key in self._CONSOLE_EXTRA_KEYS:
            if key in extra_values:
                log_lines.append(f"{self._LABELS.get(key, key):>{pad}} {extra_values[key]:.4f}")

        done_it = it + 1 - start_it
        remaining_it = total_it - start_it - done_it
        eta = self.tot_time / done_it * remaining_it
        log_lines.extend(
            [
                "-" * width,
                f"{'iter time':>{pad}} {iteration_time:.2f}s",
                f"{'elapsed':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(self.tot_time))}",
                f"{'ETA':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(eta))}",
            ]
        )
        print("\n".join(log_lines))

        if self.logger_type == "wandb":
            for video in pathlib.Path(self.log_dir).rglob("*.mp4"):  # type: ignore
                self.writer.save_video(video, it)  # type: ignore

        self.ep_extras.clear()


class DreamWaqVelocityRunner(MjlabOnPolicyRunner):
    """Mjlab runner for BPX DreamWaQ rough-terrain training."""

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        super().__init__(env, train_cfg, log_dir, device)
        self.logger = BpxCompactLogger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
        )
