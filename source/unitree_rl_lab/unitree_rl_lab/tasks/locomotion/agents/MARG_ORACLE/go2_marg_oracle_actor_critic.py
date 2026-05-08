from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


def _build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int, activation_name: str) -> nn.Sequential:
    activation = resolve_nn_activation(activation_name)
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation)
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class Go2MargOracleActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actions,
        proprioception: int = 45,
        proprioception_history: int = 270, # proprioception * 6 past steps
        terrain_height: int = 187,
        privileged: int = 42,
        terrain_hidden_dims: list[int] = [128, 64],
        terrain_feat_dim: int = 16,
        estimator_hidden_dims: list[int] = [256, 128],
        estimator_output_dim: int = 7,
        actor_hidden_dims: list[int] = [512, 256, 128],
        critic_hidden_dims: list[int] = [512, 256, 128],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            print(
                "Go2MargOracleActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )

        self.proprioception = proprioception
        self.proprioception_history = proprioception_history
        self.terrain_height = terrain_height
        self.privileged = privileged
        self.terrain_feat_dim = terrain_feat_dim
        self.estimator_output_dim = estimator_output_dim
        self.estimator_activation = "relu"
        self.elevation_activation = "relu"


        # ====================== ElevationNet ======================
        # 128*64*16
        self.elevation_net = _build_mlp(
            terrain_height, 
            terrain_hidden_dims, 
            terrain_feat_dim, 
            activation_name=self.elevation_activation
        )
        
        # ====================== EstimatorNet ======================
        # 256*128*7
        self.estimator_net = _build_mlp(
            proprioception_history,
            estimator_hidden_dims,
            estimator_output_dim,
            activation_name=self.estimator_activation,
        )
        
        
        actor_input_dim = proprioception + terrain_feat_dim + estimator_output_dim
        critic_input_dim = proprioception + privileged + terrain_feat_dim
        
        # ====================== ActorNet ======================
        # 512*256*128*12
        self.actor = _build_mlp(
            actor_input_dim, 
            actor_hidden_dims, 
            num_actions, 
            activation_name=activation)
        
        # ====================== CriticNet ======================
        # 512*256*128*1
        self.critic = _build_mlp(
            critic_input_dim, 
            critic_hidden_dims, 
            1, 
            activation_name=activation)

        print(f"ElevationNet: {self.elevation_net}")
        print(f"EstimatorNet: {self.estimator_net}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        self._latest_estimator_output = None
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        pass

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _encode_actor_obs(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        raw_obs = observations["policy_raw_obs"]
        history_obs = observations["policy_history_obs"]
        terrain_obs = observations["policy_terrain_obs"]

        terrain_feat = self.elevation_net(terrain_obs)
        estimator_input = history_obs
        est_feat = self.estimator_net(estimator_input)
        self._latest_estimator_output = est_feat
        return torch.cat((raw_obs, terrain_feat, est_feat), dim=-1)

    def _encode_critic_obs(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        raw_obs = observations["policy_raw_obs"]
        privileged_obs = observations["privileged_obs"]
        terrain_obs = observations["policy_terrain_obs"]

        terrain_feat = self.elevation_net(terrain_obs)
        return torch.cat((raw_obs, privileged_obs, terrain_feat), dim=-1)

    def update_distribution(self, observations: dict[str, torch.Tensor]):
        mean = self.actor(self._encode_actor_obs(observations))
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, observations: dict[str, torch.Tensor], **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: dict[str, torch.Tensor]):
        return self.actor(self._encode_actor_obs(observations))

    def evaluate(self, critic_observations: dict[str, torch.Tensor], **kwargs):
        return self.critic(self._encode_critic_obs(critic_observations))

    def estimate(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        raw_obs = observations["policy_raw_obs"]
        history_obs = observations["policy_history_obs"]
        estimator_input = history_obs
        return self.estimator_net(estimator_input)
