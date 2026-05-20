from __future__ import annotations

import os
import statistics
import time
from collections import deque
from importlib import import_module

import torch

from .go2_marg_oracle_ppo import Go2MargOraclePPO


def _import_class(import_path: str):
    module_name, class_name = import_path.rsplit(":", 1)
    module = import_module(module_name)
    return getattr(module, class_name)


class Go2MargOracleRunner:
    """Custom PPO runner for dict-based actor observations."""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        obs, extras = self.env.get_observations()
        obs_dict = extras["observations"]
        actor_obs_keys = ["policy_raw_obs", "policy_history_obs", "policy_terrain_obs"]
        critic_obs_keys = ["policy_raw_obs", "policy_terrain_obs", "privileged_obs"]

        actor_obs_dict = {key: obs_dict[key].to(self.device) for key in actor_obs_keys}
        critic_obs_dict = {key: obs_dict[key].to(self.device) for key in critic_obs_keys}

        policy_class = _import_class(self.policy_cfg.pop("class_name"))
        policy = policy_class(
            num_actor_obs=obs.shape[1],
            num_critic_obs=sum(value.shape[1] for value in critic_obs_dict.values()),
            num_actions=self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)

        self.alg = Go2MargOraclePPO(policy, device=self.device, **self.alg_cfg)
        actor_obs_shapes = {key: tuple(value.shape[1:]) for key, value in actor_obs_dict.items()}
        self.alg.init_storage(
            num_envs=self.env.num_envs,
            num_transitions_per_env=train_cfg["num_steps_per_env"],
            actor_obs_shapes=actor_obs_shapes,
            critic_obs_shapes={key: tuple(value.shape[1:]) for key, value in critic_obs_dict.items()},
            actions_shape=(self.env.num_actions,),
        )

        self.actor_obs_keys = actor_obs_keys
        self.critic_obs_keys = critic_obs_keys
        self.num_steps_per_env = train_cfg["num_steps_per_env"]
        self.save_interval = train_cfg["save_interval"]
        self.empirical_normalization = False
        self.log_dir = log_dir
        self.writer = None
        self.current_learning_iteration = 0
        self.tot_timesteps = 0
        self.tot_time = 0
        self.git_status_repos = []

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)

    def train_mode(self):
        self.alg.policy.train()

    def eval_mode(self):
        self.alg.policy.eval()

    def get_inference_policy(self, device=None):
        self.eval_mode()
        if device is not None:
            self.alg.policy.to(device)

        def policy(obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
            actor_obs = {key: obs_dict[key].to(device or self.device) for key in self.actor_obs_keys}
            return self.alg.policy.act_inference(actor_obs)

        return policy

    def save(self, path: str, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.policy.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, weights_only=False)
        self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def _extract_actor_obs(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: obs_dict[key].to(self.device) for key in self.actor_obs_keys}

    def _extract_critic_obs(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: obs_dict[key].to(self.device) for key in self.critic_obs_keys}

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        if self.log_dir is not None and self.writer is None:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs, extras = self.env.get_observations()
        obs_dict = extras["observations"]
        actor_obs = self._extract_actor_obs(obs_dict)
        critic_obs_dict = self._extract_critic_obs(obs_dict)
        self.train_mode()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(actor_obs, critic_obs_dict)
                    _, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    rewards, dones = rewards.to(self.device), dones.to(self.device)
                    next_obs_dict = infos["observations"]
                    actor_obs = self._extract_actor_obs(next_obs_dict)
                    critic_obs_dict = self._extract_critic_obs(next_obs_dict)

                    self.alg.process_env_step(rewards, dones, infos)

                    if "episode" in infos:
                        ep_infos.append(infos["episode"])
                    elif "log" in infos:
                        ep_infos.append(infos["log"])

                    cur_reward_sum += rewards
                    cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop
                self.alg.compute_returns(critic_obs_dict)

            loss_dict = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            if self.log_dir is not None:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()

        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        collection_size = self.num_steps_per_env * self.env.num_envs
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            all_keys = set()
            for ep_info in locs["ep_infos"]:
                all_keys.update(ep_info.keys())

            for key in sorted(all_keys):
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar(("Episode/" + key) if "/" not in key else key, value, locs["it"])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.policy.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])

        iter_header = f" Learning iteration {locs['it']}/{locs['tot_iter']} "
        log_string = (
            f"{'#' * width}\n"
            f"{iter_header:^{width}}\n\n"
            f"{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"
            f"{'Num environments:':>{pad}} {self.env.num_envs}\n"
            f"{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"
        )
        for key, value in locs["loss_dict"].items():
            log_string += f"{f'Mean {key} loss:':>{pad}} {value:.4f}\n"
        if len(locs["rewbuffer"]) > 0:
            log_string += f"{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"
            log_string += f"{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"
        log_string += ep_string
        log_string += (
            f"{'-' * width}\n"
            f"{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"
            f"{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"
            f"{'Time elapsed:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(self.tot_time))}\n"
            f"{'ETA:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(self.tot_time / (locs['it'] - locs['start_iter'] + 1) * (locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])))}\n"
        )
        print(log_string)
