from dataclasses import dataclass

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@dataclass
class DreamWaqAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "bpx_mjlab.waq.algorithm:DreamWaqPPO"
    latent_dim: int = 24
    encoder_hidden_dims: tuple[int, ...] = (512, 256)
    decoder_hidden_dims: tuple[int, ...] = (256, 512)
    velocity_loss_coef: float = 2.0
    terrain_loss_coef: float = 0.5
    kl_loss_coef: float = 1.0e-3


def bpx_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            entropy_coef=0.01,
        ),
        experiment_name="bpx_velocity",
        max_iterations=10_000,
    )


def bpx_waq_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        num_steps_per_env=16,
        max_iterations=50_000,
        obs_groups={
            "actor": ("actor",),
            "actor_history": ("actor_history",),
            "critic": ("critic",),
            "estimator_target": ("estimator_target",),
        },
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            obs_normalization=True,
        ),
        algorithm=DreamWaqAlgorithmCfg(
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            entropy_coef=0.005,
            desired_kl=0.01,
        ),
        experiment_name="bpx_waq_rough",
        run_name="dreamwaq",
        save_interval=200,
    )
