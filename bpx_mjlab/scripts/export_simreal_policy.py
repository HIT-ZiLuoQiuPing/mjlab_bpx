#!/usr/bin/env python3
"""Export a BPX DreamWaQ checkpoint for the bpx_simreal_v6 SDK UI."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import asdict
from pathlib import Path

import bpx_mjlab  # noqa: F401 - register BPX tasks with mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


def _resolve_output_path(checkpoint: Path, output: str | None, name: str) -> Path:
    if output is None:
        return checkpoint.parent / name

    output_path = Path(output).expanduser()
    if output_path.suffix:
        return output_path
    return output_path / name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a BPX DreamWaQ checkpoint to a single-input TorchScript policy."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to model_*.pt from training.")
    parser.add_argument("--output", default=None, help="Output .pt path or output directory.")
    parser.add_argument("--name", default="bpx_dwaq_v2.pt", help="Output file name when --output is a directory.")
    parser.add_argument("--task", default="Mjlab-Velocity-Rough-BPX", help="mjlab task id.")
    parser.add_argument("--device", default="cpu", help="Device used while loading the checkpoint.")
    parser.add_argument("--num-envs", type=int, default=1, help="Small env count used to infer model dimensions.")
    parser.add_argument(
        "--deploy-dir",
        default=None,
        help="Optional bpx_simreal_v6 root. If set, copy the exported policy to <deploy-dir>/policy/<name>.",
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    output_path = _resolve_output_path(checkpoint, args.output, args.name).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = args.num_envs
    agent_cfg = load_rl_cfg(args.task)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    try:
        runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
        runner = runner_cls(wrapped_env, asdict(agent_cfg), log_dir=None, device=args.device)
        runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=args.device)
        runner.export_policy_to_jit(str(output_path.parent), output_path.name)
    finally:
        env.close()

    print(f"[OK] exported sim2real TorchScript policy: {output_path}")
    print("[OK] expected inputs: history_only=225 or current_plus_history=270; output=12 actions")

    if args.deploy_dir is not None:
        deploy_policy_dir = Path(args.deploy_dir).expanduser().resolve() / "policy"
        deploy_policy_dir.mkdir(parents=True, exist_ok=True)
        deploy_path = deploy_policy_dir / args.name
        shutil.copy2(output_path, deploy_path)
        print(f"[OK] copied policy to deploy package: {deploy_path}")


if __name__ == "__main__":
    main()
