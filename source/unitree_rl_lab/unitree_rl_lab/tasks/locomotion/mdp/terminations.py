from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terminate_stationary_for_duration(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    duration: float = 5.0,
    distance_threshold: float = 0.50,
    command_speed_threshold: float = 0.05,
) -> torch.Tensor:
    """Terminate moving-command episodes if the robot stays near the same XY position too long."""

    asset = env.scene[asset_cfg.name]
    root_pos_xy = asset.data.root_pos_w[:, :2]

    state_attr = f"_stationary_termination_{asset_cfg.name}"
    state = getattr(env, state_attr, None)
    if (
        state is None
        or state["anchor_pos_xy"].shape[0] != env.num_envs
        or state["anchor_pos_xy"].device != env.device
    ):
        state = {
            "anchor_pos_xy": root_pos_xy.clone(),
            "stationary_time": torch.zeros(env.num_envs, dtype=torch.float32, device=env.device),
        }
        setattr(env, state_attr, state)

    command = env.command_manager.get_command(command_name)
    command_active = torch.linalg.norm(command[:, :2], dim=1) > command_speed_threshold
    new_episode = env.episode_length_buf <= 1

    moved_far_enough = torch.linalg.norm(root_pos_xy - state["anchor_pos_xy"], dim=1) > distance_threshold
    reset_stationary_state = new_episode | moved_far_enough | ~command_active

    state["anchor_pos_xy"] = torch.where(reset_stationary_state.unsqueeze(1), root_pos_xy, state["anchor_pos_xy"])
    state["stationary_time"] = torch.where(
        reset_stationary_state,
        torch.zeros_like(state["stationary_time"]),
        state["stationary_time"] + env.step_dt,
    )

    return state["stationary_time"] >= duration


def terminate_feet_on_base_plane_selected_terrains(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot"),
    restricted_terrain_types: tuple[str, ...] | None = None,
    force_threshold: float = 1.0,
    plane_height_threshold: float = -0.2,
) -> torch.Tensor:
    """
    Terminate if the foot sole drops to or below the base-plane height on selected terrain types.

    Notes:
    - Terrain assignment in generator mode is column-based (`terrain_types` stores column index).
    - This function reconstructs the curriculum column->sub-terrain mapping from proportions.
    - The termination condition is based on foot sole height relative to the environment origin.
    """

    device = env.device
    num_envs = env.scene.num_envs

    terrain_generator_cfg = env.cfg.scene.terrain.terrain_generator
    sub_terrains = terrain_generator_cfg.sub_terrains
    terrain_names = list(sub_terrains.keys())
    if len(terrain_names) == 0:
        return torch.zeros(num_envs, dtype=torch.bool, device=device)

    if restricted_terrain_types is None:
        restricted_terrain_types = tuple(terrain_names)

    restricted_name_set = set(restricted_terrain_types)
    restricted_sub_indices = [i for i, name in enumerate(terrain_names) if name in restricted_name_set]
    if len(restricted_sub_indices) == 0:
        return torch.zeros(num_envs, dtype=torch.bool, device=device)

    proportions = torch.tensor(
        [sub_cfg.proportion for sub_cfg in sub_terrains.values()],
        dtype=torch.float32,
        device=device,
    )
    proportions = proportions / torch.sum(proportions)
    cumulative = torch.cumsum(proportions, dim=0)

    terrain_cols = env.scene.terrain.terrain_types.to(device)
    ratios = terrain_cols.float() / float(terrain_generator_cfg.num_cols) + 0.001
    sub_indices = torch.searchsorted(cumulative, ratios, right=False)
    sub_indices = torch.clamp(sub_indices, max=len(terrain_names) - 1)

    restricted_sub_indices_t = torch.tensor(restricted_sub_indices, dtype=torch.long, device=device)
    selected_terrain_mask = torch.any(sub_indices.unsqueeze(1) == restricted_sub_indices_t.unsqueeze(0), dim=1)
    if not torch.any(selected_terrain_mask):
        return torch.zeros(num_envs, dtype=torch.bool, device=device)

    asset = env.scene[asset_cfg.name]

    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    env_origin_z = env.scene.env_origins[:, 2].unsqueeze(1)
    foot_sole_below_threshold = (feet_z - env_origin_z) <= plane_height_threshold

    return torch.any(foot_sole_below_threshold, dim=1) & selected_terrain_mask
