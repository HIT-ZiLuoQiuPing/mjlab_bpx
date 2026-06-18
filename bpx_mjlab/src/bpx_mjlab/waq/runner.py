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

    _CONSOLE_SECTIONS = (
        (
            "Terrain curriculum",
            (
                ("Mean terrain level", "Curriculum/terrain_levels/mean"),
                ("Maximum terrain level", "Curriculum/terrain_levels/max"),
                ("Terrain promotion rate", "Curriculum/terrain_levels/move_up_rate"),
                ("Terrain demotion rate", "Curriculum/terrain_levels/move_down_rate"),
            ),
        ),
        (
            "Velocity command curriculum",
            (
                ("Maximum forward command x", "Curriculum/command_vel/lin_vel_x_max"),
                ("Maximum lateral command y", "Curriculum/command_vel/lin_vel_y_max"),
                ("Maximum yaw command", "Curriculum/command_vel/ang_vel_z_max"),
            ),
        ),
        (
            "Velocity tracking",
            (
                ("Target velocity x", "Episode_Metrics/bpx_cmd_vx"),
                ("Actual velocity x", "Episode_Metrics/bpx_vel_vx"),
                ("Absolute velocity error x", "Episode_Metrics/bpx_err_vx"),
                ("Target velocity y", "Episode_Metrics/bpx_cmd_vy"),
                ("Actual velocity y", "Episode_Metrics/bpx_vel_vy"),
                ("Absolute velocity error y", "Episode_Metrics/bpx_err_vy"),
                ("Target yaw velocity", "Episode_Metrics/bpx_cmd_wz"),
                ("Actual yaw velocity", "Episode_Metrics/bpx_vel_wz"),
                ("Absolute yaw velocity error", "Episode_Metrics/bpx_err_wz"),
                ("Built-in xy velocity error", "Metrics/twist/error_vel_xy"),
                ("Built-in yaw velocity error", "Metrics/twist/error_vel_yaw"),
            ),
        ),
        (
            "Important reward terms",
            (
                ("Linear velocity tracking reward", "Episode_Reward/track_linear_velocity"),
                ("Lateral velocity tracking reward", "Episode_Reward/track_lateral_velocity"),
                ("Yaw velocity tracking reward", "Episode_Reward/track_yaw_velocity"),
                ("Upright reward", "Episode_Reward/upright"),
            ),
        ),
        (
            "Episode terminations",
            (
                ("Fell over episodes", "Episode_Termination/fell_over"),
                ("Illegal contact episodes", "Episode_Termination/illegal_contact"),
                ("Timed out episodes", "Episode_Termination/time_out"),
            ),
        ),
    )

    @staticmethod
    def _append_section(
        lines: list[str],
        title: str,
        items: tuple[tuple[str, str], ...],
        values: dict[str, float],
        pad: int,
    ) -> None:
        available = [(label, key) for label, key in items if key in values]
        if not available:
            return
        lines.append("")
        lines.append(f"{title}:")
        for label, key in available:
            lines.append(f"{label + ':':>{pad}} {values[key]:.4f}")

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
        width: int = 92,
        pad: int = 40,
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
            f"\033[1m{f' BPX training iteration {it}/{total_it} '.center(width)}\033[0m",
            "",
        ]
        run_name = self.cfg.get("run_name")
        if run_name:
            log_lines.append(f"{'Run name:':>{pad}} {run_name}")
        log_lines.extend(
            [
                f"{'Total training steps:':>{pad}} {self.tot_timesteps}",
                f"{'Steps per second:':>{pad}} {fps}",
                f"{'Collection time:':>{pad}} {collect_time:.3f}s",
                f"{'Learning time:':>{pad}} {learn_time:.3f}s",
                f"{'Learning rate:':>{pad}} {learning_rate:.3g}",
            ]
        )

        if loss_dict:
            log_lines.append("")
            log_lines.append("Policy update losses:")
            for key, value in loss_dict.items():
                label = key.replace("_", " ")
                log_lines.append(f"{f'Mean {label} loss:':>{pad}} {value:.4f}")

        log_lines.append(f"{'Mean action standard deviation:':>{pad}} {action_std.mean().item():.3f}")
        if mean_reward is not None and mean_episode_length is not None:
            log_lines.extend(
                [
                    "",
                    "Training health:",
                    f"{'Mean reward:':>{pad}} {mean_reward:.2f}",
                    f"{'Mean episode length:':>{pad}} {mean_episode_length:.2f}",
                ]
            )

        for title, items in self._CONSOLE_SECTIONS:
            self._append_section(log_lines, title, items, extra_values, pad)

        done_it = it + 1 - start_it
        remaining_it = total_it - start_it - done_it
        eta = self.tot_time / done_it * remaining_it
        log_lines.extend(
            [
                "-" * width,
                f"{'Iteration time:':>{pad}} {iteration_time:.2f}s",
                f"{'Time elapsed:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(self.tot_time))}",
                f"{'Estimated time remaining:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(eta))}",
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
