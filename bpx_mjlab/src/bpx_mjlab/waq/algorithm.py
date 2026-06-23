from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups
from tensordict import TensorDict

from .models import DreamWaqActor


class DreamWaqPPO(PPO):
    def __init__(
        self,
        actor: DreamWaqActor,
        critic: MLPModel,
        storage: RolloutStorage,
        velocity_loss_coef: float = 1.0,
        terrain_loss_coef: float = 0.2,
        kl_loss_coef: float = 1.0e-3,
        **kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, **kwargs)
        self.velocity_loss_coef = velocity_loss_coef
        self.terrain_loss_coef = terrain_loss_coef
        self.kl_loss_coef = kl_loss_coef

    actor: DreamWaqActor

    def update(self) -> dict[str, float]:
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_velocity_loss = 0.0
        mean_terrain_loss = 0.0
        mean_kl_loss = 0.0

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches,
                self.num_learning_epochs,
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches,
                self.num_learning_epochs,
            )

        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (  # type: ignore[union-attr]
                        batch.advantages.std() + 1.0e-8  # type: ignore[union-attr]
                    )

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore[arg-type]
            values = self.critic(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[1],
            )
            distribution_params = tuple(
                p[:original_batch_size] for p in self.actor.output_distribution_params
            )
            entropy = self.actor.output_entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(
                        batch.old_distribution_params,  # type: ignore[arg-type]
                        distribution_params,
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore[arg-type]
            surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore[arg-type]
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore[arg-type]
                ratio,
                1.0 - self.clip_param,
                1.0 + self.clip_param,
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(  # type: ignore[operator]
                    -self.clip_param,
                    self.clip_param,
                )
                value_losses = (values - batch.returns).pow(2)  # type: ignore[operator]
                value_losses_clipped = (value_clipped - batch.returns).pow(2)  # type: ignore[operator]
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()  # type: ignore[operator]

            estimator_losses = self.actor.compute_estimator_losses(batch.observations)
            velocity_loss = estimator_losses["estimator_velocity"]
            terrain_loss = estimator_losses["estimator_terrain"]
            estimator_kl_loss = estimator_losses["estimator_kl"]

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean()
                + self.velocity_loss_coef * velocity_loss
                + self.terrain_loss_coef * terrain_loss
                + self.kl_loss_coef * estimator_kl_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_velocity_loss += velocity_loss.item()
            mean_terrain_loss += terrain_loss.item()
            mean_kl_loss += estimator_kl_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_velocity_loss /= num_updates
        mean_terrain_loss /= num_updates
        mean_kl_loss /= num_updates

        self.storage.clear()
        return {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "estimator_velocity": mean_velocity_loss,
            "estimator_terrain": mean_terrain_loss,
            "estimator_kl": mean_kl_loss,
        }

    def save(self) -> dict:
        saved_dict = super().save()
        saved_dict["cenet_state_dict"] = self.actor.cenet.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if load_cfg is None or load_cfg.get("actor", False):
            if "cenet_state_dict" in loaded_dict and "actor_state_dict" not in loaded_dict:
                self.actor.cenet.load_state_dict(loaded_dict["cenet_state_dict"], strict=strict)
        return load_iteration

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "DreamWaqPPO":
        alg_class: type[DreamWaqPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore[assignment]
        cfg["actor"].pop("class_name", None)
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore[assignment]

        cfg["obs_groups"] = resolve_obs_groups(
            obs,
            cfg["obs_groups"],
            ["actor", "actor_history", "critic", "estimator_target"],
        )

        alg_cfg = cfg["algorithm"]
        latent_dim = alg_cfg.pop("latent_dim")
        encoder_hidden_dims = tuple(alg_cfg.pop("encoder_hidden_dims"))
        decoder_hidden_dims = tuple(alg_cfg.pop("decoder_hidden_dims"))
        velocity_loss_coef = alg_cfg.pop("velocity_loss_coef")
        terrain_loss_coef = alg_cfg.pop("terrain_loss_coef")
        kl_loss_coef = alg_cfg.pop("kl_loss_coef")
        alg_cfg.pop("share_cnn_encoders", None)
        alg_cfg["rnd_cfg"] = None
        alg_cfg["symmetry_cfg"] = None

        actor = DreamWaqActor(
            obs,
            cfg["obs_groups"],
            env.num_actions,
            latent_dim=latent_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            **cfg["actor"],
        ).to(device)
        print(f"DreamWaQ Actor Model: {actor}")

        critic = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")

        storage = RolloutStorage(
            "rl",
            env.num_envs,
            cfg["num_steps_per_env"],
            obs,
            [env.num_actions],
            device,
        )

        return alg_class(
            actor,
            critic,
            storage,
            velocity_loss_coef=velocity_loss_coef,
            terrain_loss_coef=terrain_loss_coef,
            kl_loss_coef=kl_loss_coef,
            device=device,
            multi_gpu_cfg=cfg["multi_gpu"],
            **alg_cfg,
        )
