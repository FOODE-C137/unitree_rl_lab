from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
    lin_vel_x_delta: tuple[float, float] = (-0.1, 0.1),
    lin_vel_y_delta: tuple[float, float] = (-0.1, 0.1),
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            ranges.lin_vel_x = _update_range_towards_limit(
                ranges.lin_vel_x, lin_vel_x_delta, limit_ranges.lin_vel_x, env.device
            )
            ranges.lin_vel_y = _update_range_towards_limit(
                ranges.lin_vel_y, lin_vel_y_delta, limit_ranges.lin_vel_y, env.device
            )

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def _update_range_towards_limit(
    current: tuple[float, float] | list[float],
    delta: tuple[float, float],
    limit: tuple[float, float],
    device: torch.device | str,
) -> list[float]:
    current_tensor = torch.tensor(current, device=device)
    delta_tensor = torch.tensor(delta, device=device)
    limit_tensor = torch.tensor(limit, device=device)
    next_tensor = current_tensor + delta_tensor

    lower = _move_endpoint_towards_limit(next_tensor[0], delta[0], limit_tensor[0])
    upper = _move_endpoint_towards_limit(next_tensor[1], delta[1], limit_tensor[1])
    return torch.stack([lower, upper]).tolist()


def _move_endpoint_towards_limit(value: torch.Tensor, delta: float, limit: torch.Tensor) -> torch.Tensor:
    if delta > 0.0:
        return torch.minimum(value, limit)
    if delta < 0.0:
        return torch.maximum(value, limit)
    return value


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # Compute the distance walked relative to the reset pose inside each terrain tile.
    # This mirrors MGDP's gap-parkour curriculum, where the configured initial x/y offset
    # is subtracted before checking progress.
    root_pos_in_terrain = asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]
    init_pos_in_terrain = asset.data.default_root_state[env_ids, :2]
    distance = torch.norm(root_pos_in_terrain - init_pos_in_terrain, dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2.0
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())
