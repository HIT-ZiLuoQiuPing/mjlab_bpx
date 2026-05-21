from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.modules import EmpiricalNormalization, HiddenState, MLP
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable
from tensordict import TensorDict

# 注释：DreamWaQ模型实现，包含CENet编码器和DreamWaqActor策略网络。
def _obs_dim(obs: TensorDict, groups: tuple[str, ...] | list[str]) -> int: # 计算指定观察组的总维度，确保每个组都是1D的。
    dim = 0
    for group in groups:
        if len(obs[group].shape) != 2:
            raise ValueError(
                f"DreamWaQ only supports 1D observation groups, got {obs[group].shape} for '{group}'."
            )
        dim += obs[group].shape[-1]
    return dim


def _concat_obs(obs: TensorDict, groups: tuple[str, ...] | list[str]) -> torch.Tensor: # 将指定观察组的张量沿最后一个维度连接起来，形成一个大的输入张量。
    return torch.cat([obs[group] for group in groups], dim=-1)


class CENet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        terrain_dim: int,
        latent_dim: int,
        hidden_dims: tuple[int, ...] = (256, 128),
        decoder_hidden_dims: tuple[int, ...] = (128, 256),
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.terrain_dim = terrain_dim
        self.latent_dim = latent_dim
        self.encoder = MLP(
            input_dim=input_dim,
            output_dim=3 + 2 * latent_dim, # 输出维度包括3维速度和VAE 隐变量均值 和VAE 隐变量 log 方差
            hidden_dims=hidden_dims,
            activation=activation,
        )
        if terrain_dim > 0:
            self.decoder = MLP(
                input_dim=latent_dim,
                output_dim=terrain_dim,
                hidden_dims=decoder_hidden_dims,
                activation=activation,
            )
        else:
            self.decoder = None

    def forward(self, history: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encoder(history)
        vel = encoded[..., :3]
        z_mu = encoded[..., 3 : 3 + self.latent_dim]
        z_logvar = encoded[..., 3 + self.latent_dim :]
        z_logvar = torch.clamp(z_logvar, min=-10.0, max=4.0)
        z = z_mu
        if self.decoder is None:
            terrain = history.new_zeros((*history.shape[:-1], 0))
        else:
            terrain = self.decoder(z) # 试图从隐变量 z 中重构某些地形/环境特权信息。
        return {
            "vel": vel,
            "z": z,
            "z_mu": z_mu,
            "z_logvar": z_logvar,
            "terrain": terrain,
        }


class DreamWaqActor(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, tuple[str, ...] | list[str]],
        output_dim: int,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = True,
        distribution_cfg: dict | None = None,
        latent_dim: int = 16,
        encoder_hidden_dims: tuple[int, ...] = (256, 128),
        decoder_hidden_dims: tuple[int, ...] = (128, 256),
    ) -> None:
        super().__init__()
        if distribution_cfg is None:
            raise ValueError("DreamWaqActor requires a stochastic distribution_cfg.")

        self.actor_groups = tuple(obs_groups["actor"]) # 当前策略观测，例如角速度、重力方向、关节角、关节速度、上一动作等
        self.history_groups = tuple(obs_groups["actor_history"]) # 历史观测，用来给 CENet 估计速度和隐变量
        self.estimator_target_groups = tuple(obs_groups["estimator_target"]) # CENet 的监督目标，前 3 维是 base velocity，后面是 terrain 信息

        self.actor_obs_dim = _obs_dim(obs, self.actor_groups)
        self.history_obs_dim = _obs_dim(obs, self.history_groups)
        self.estimator_target_dim = _obs_dim(obs, self.estimator_target_groups)
        if self.estimator_target_dim < 3:
            raise ValueError("estimator_target must contain at least the 3D base linear velocity target.")
        self.terrain_dim = self.estimator_target_dim - 3

        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(self.actor_obs_dim)
            self.history_obs_normalizer = EmpiricalNormalization(self.history_obs_dim)
        else:
            self.actor_obs_normalizer = nn.Identity()
            self.history_obs_normalizer = nn.Identity()

        self.cenet = CENet(
            input_dim=self.history_obs_dim,
            terrain_dim=self.terrain_dim,
            latent_dim=latent_dim,
            hidden_dims=encoder_hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            activation=activation,
        )

        dist_cfg = dict(distribution_cfg)
        dist_class: type[Distribution] = resolve_callable(dist_cfg.pop("class_name"))  # type: ignore[assignment]
        self.distribution: Distribution = dist_class(output_dim, **dist_cfg)
        self.mlp = MLP(
            input_dim=self.actor_obs_dim + 3 + latent_dim,
            output_dim=self.distribution.input_dim,
            hidden_dims=hidden_dims,
            activation=activation,
        )
        self.distribution.init_mlp_weights(self.mlp)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        del masks, hidden_state
        latent = self.get_latent(obs)
        mlp_output = self.mlp(latent)
        if stochastic_output:
            self.distribution.update(mlp_output)
            return self.distribution.sample()
        return self.distribution.deterministic_output(mlp_output)

    def get_latent(self, obs: TensorDict) -> torch.Tensor:
        actor_obs = self.actor_obs_normalizer(_concat_obs(obs, self.actor_groups))
        history_obs = self.history_obs_normalizer(_concat_obs(obs, self.history_groups))
        cenet_out = self.cenet(history_obs)
        return torch.cat([actor_obs, cenet_out["vel"], cenet_out["z"]], dim=-1)

    def compute_estimator_losses(self, obs: TensorDict) -> dict[str, torch.Tensor]:
        history_obs = self.history_obs_normalizer(_concat_obs(obs, self.history_groups))
        target = _concat_obs(obs, self.estimator_target_groups)
        cenet_out = self.cenet(history_obs)

        vel_target = target[..., :3]
        terrain_target = target[..., 3:]
        velocity_loss = F.mse_loss(cenet_out["vel"], vel_target)
        if self.terrain_dim > 0:
            terrain_loss = F.mse_loss(cenet_out["terrain"], terrain_target)
        else:
            terrain_loss = target.new_zeros(())
        z_mu = cenet_out["z_mu"]
        z_logvar = cenet_out["z_logvar"]
        kl_loss = -0.5 * torch.mean(1.0 + z_logvar - z_mu.pow(2) - z_logvar.exp())
        return {
            "estimator_velocity": velocity_loss,
            "estimator_terrain": terrain_loss,
            "estimator_kl": kl_loss,
        }

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        del dones, hidden_state

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        del dones

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)

    def update_normalization(self, obs: TensorDict) -> None:
        if not self.obs_normalization:
            return
        self.actor_obs_normalizer.update(_concat_obs(obs, self.actor_groups))  # type: ignore[operator]
        self.history_obs_normalizer.update(_concat_obs(obs, self.history_groups))  # type: ignore[operator]

    def as_jit(self) -> nn.Module:
        return _TorchDreamWaqActor(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _OnnxDreamWaqActor(self, verbose)


class _ExportDreamWaqActorBase(nn.Module):
    def __init__(self, model: DreamWaqActor) -> None:
        super().__init__()
        self.actor_obs_normalizer = copy.deepcopy(model.actor_obs_normalizer)
        self.history_obs_normalizer = copy.deepcopy(model.history_obs_normalizer)
        self.cenet = copy.deepcopy(model.cenet)
        self.mlp = copy.deepcopy(model.mlp)
        self.deterministic_output = model.distribution.as_deterministic_output_module()
        self.actor_input_size = model.actor_obs_dim
        self.history_input_size = model.history_obs_dim

    def forward(self, actor_obs: torch.Tensor, actor_history: torch.Tensor) -> torch.Tensor:
        actor_obs = self.actor_obs_normalizer(actor_obs)
        actor_history = self.history_obs_normalizer(actor_history)
        cenet_out = self.cenet(actor_history)
        latent = torch.cat([actor_obs, cenet_out["vel"], cenet_out["z"]], dim=-1)
        return self.deterministic_output(self.mlp(latent))


class _TorchDreamWaqActor(_ExportDreamWaqActorBase):
    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxDreamWaqActor(_ExportDreamWaqActorBase):
    is_recurrent: bool = False

    def __init__(self, model: DreamWaqActor, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose

    def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(1, self.actor_input_size),
            torch.zeros(1, self.history_input_size),
        )

    @property
    def input_names(self) -> list[str]:
        return ["actor", "actor_history"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
