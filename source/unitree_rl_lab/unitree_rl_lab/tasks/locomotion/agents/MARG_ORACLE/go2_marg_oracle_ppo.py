from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from isaaclab.utils.string import string_to_callable

from .go2_marg_oracle_rollout_storage import Go2MargOracleRolloutStorage


# Layout of privileged_obs in go2_marg_oracle_velocity_env_cfg.py:
# [0:3]   real_linear_velocity
# [3:7]   feet_contacts
# [7:11]  critical_masses
# [11:12] friction
# [12:14] com_shift
# [14:16] disturbance_force
# [16:42] actuator_params_26
PRIV_REAL_LINEAR_VEL = slice(0, 3)
PRIV_FEET_CONTACTS = slice(3, 7)


class Go2MargOraclePPO:
    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        estimator_loss_coef=1.0,
        velocity_loss_coef=1.0,
        contact_loss_coef=1.0,
        symmetry_cfg=None,
        **kwargs,
    ):
        self.device = device
        self.policy = policy.to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.storage: Go2MargOracleRolloutStorage | None = None
        self.transition = Go2MargOracleRolloutStorage.Transition()

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        self.estimator_loss_coef = estimator_loss_coef
        self.velocity_loss_coef = velocity_loss_coef
        self.contact_loss_coef = contact_loss_coef
        self.symmetry_cfg = symmetry_cfg
        self.use_data_augmentation = self._get_symmetry_cfg_value("use_data_augmentation", False)
        self.use_mirror_loss = self._get_symmetry_cfg_value("use_mirror_loss", False)
        self.mirror_loss_coeff = self._get_symmetry_cfg_value("mirror_loss_coeff", 0.0)
        self.data_augmentation_func = self._get_symmetry_cfg_value("data_augmentation_func", None)
        if isinstance(self.data_augmentation_func, str):
            self.data_augmentation_func = string_to_callable(self.data_augmentation_func)

    def _get_symmetry_cfg_value(self, name: str, default):
        if self.symmetry_cfg is None:
            return default
        if isinstance(self.symmetry_cfg, dict):
            return self.symmetry_cfg.get(name, default)
        return getattr(self.symmetry_cfg, name, default)

    def _augment_batch_with_symmetry(
        self,
        obs_batch,
        critic_obs_batch,
        actions_batch,
        target_values_batch,
        advantages_batch,
        returns_batch,
        old_actions_log_prob_batch,
    ):
        if not self.use_data_augmentation or not callable(self.data_augmentation_func):
            return (
                obs_batch,
                critic_obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
            )

        obs_batch, actions_batch = self.data_augmentation_func(None, obs_batch, actions_batch)
        critic_obs_batch, _ = self.data_augmentation_func(None, critic_obs_batch, None)
        target_values_batch = target_values_batch.repeat(2, 1)
        advantages_batch = advantages_batch.repeat(2, 1)
        returns_batch = returns_batch.repeat(2, 1)
        old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(2, 1)

        return (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
        )

    def _compute_mirror_loss(self, obs_batch):
        if (
            not self.use_mirror_loss
            or self.mirror_loss_coeff <= 0.0
            or not callable(self.data_augmentation_func)
        ):
            return next(iter(obs_batch.values())).new_zeros(())

        batch_size = next(iter(obs_batch.values())).shape[0]
        obs_aug, _ = self.data_augmentation_func(None, obs_batch, None)
        mirrored_obs = {key: value[batch_size:] for key, value in obs_aug.items()}

        action_mean = self.policy.act_inference(obs_batch)
        mirrored_obs_action_mean = self.policy.act_inference(mirrored_obs)
        _, action_mean_aug = self.data_augmentation_func(None, None, action_mean)
        mirrored_action_mean = action_mean_aug[batch_size:].detach()

        return torch.nn.functional.mse_loss(mirrored_obs_action_mean, mirrored_action_mean)

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shapes, critic_obs_shapes, actions_shape):
        self.storage = Go2MargOracleRolloutStorage(
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            actor_obs_shapes=actor_obs_shapes,
            critic_obs_shapes=critic_obs_shapes,
            actions_shape=actions_shape,
            device=self.device,
        )

    def act(self, obs_dict, critic_obs_dict):
        self.transition.actions = self.policy.act(obs_dict).detach()
        self.transition.values = self.policy.evaluate(critic_obs_dict).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.actor_observations = {key: value.detach() for key, value in obs_dict.items()}
        self.transition.critic_observations = {key: value.detach() for key, value in critic_obs_dict.items()}
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs_dict):
        last_values = self.policy.evaluate(last_critic_obs_dict).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def update(self):
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_velocity_loss = 0.0
        mean_contact_loss = 0.0
        mean_estimator_loss = 0.0
        mean_mirror_loss = 0.0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        num_updates = 0
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) in generator:
            if self.normalize_advantage_per_mini_batch:
                advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            mirror_loss = self._compute_mirror_loss(obs_batch)

            self.policy.act(obs_batch)
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / (old_sigma_batch + 1.0e-5) + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        dim=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            (
                obs_batch,
                critic_obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
            ) = self._augment_batch_with_symmetry(
                obs_batch,
                critic_obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
            )

            self.policy.act(obs_batch)
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(critic_obs_batch)
            entropy_batch = self.policy.entropy
            est_batch = self.policy.estimate(obs_batch)

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            privileged_obs_batch = critic_obs_batch["privileged_obs"]
            vel_target = privileged_obs_batch[:, PRIV_REAL_LINEAR_VEL]
            contact_target = privileged_obs_batch[:, PRIV_FEET_CONTACTS]
            vel_pred = est_batch[:, :3]
            contact_pred = est_batch[:, 3:7]

            velocity_loss = torch.nn.functional.mse_loss(vel_pred, vel_target)
            contact_loss = torch.nn.functional.binary_cross_entropy_with_logits(contact_pred, contact_target)
            estimator_loss = self.velocity_loss_coef * velocity_loss + self.contact_loss_coef * contact_loss

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + self.estimator_loss_coef * estimator_loss
                + self.mirror_loss_coeff * mirror_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_velocity_loss += velocity_loss.item()
            mean_contact_loss += contact_loss.item()
            mean_estimator_loss += estimator_loss.item()
            mean_mirror_loss += mirror_loss.item()
            num_updates += 1

        self.storage.clear()
        return {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "velocity": mean_velocity_loss / num_updates,
            "contact": mean_contact_loss / num_updates,
            "estimator": mean_estimator_loss / num_updates,
            "mirror": mean_mirror_loss / num_updates,
        }
