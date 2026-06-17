from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
Joint penalties.
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    cmd_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < cmd_threshold)


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footvel_translated[:, i, :])
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


"""
Feet Gait rewards.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


"""
Other rewards.
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward





# ============================ MARG Reward  ===============================
# =========================================================================
def _query_terrain_height_from_scanner(
    env: ManagerBasedRLEnv,
    sample_xy_w: torch.Tensor,
    height_sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
) -> torch.Tensor:
    """Nearest-neighbor terrain height query using the height scanner ray hits.

    Args:
        sample_xy_w: [num_envs, num_feet, num_samples, 2] world-frame sample XY.

    Returns:
        terrain_z: [num_envs, num_feet, num_samples] sampled terrain heights.
    """

    if height_sensor_cfg.name not in env.scene.sensors:
        return torch.zeros(sample_xy_w.shape[:-1], device=env.device, dtype=sample_xy_w.dtype)

    height_sensor = env.scene.sensors[height_sensor_cfg.name]
    ray_hits_w = height_sensor.data.ray_hits_w
    if ray_hits_w is None:
        return torch.zeros(sample_xy_w.shape[:-1], device=env.device, dtype=sample_xy_w.dtype)

    ray_xy_w = ray_hits_w[..., :2]
    ray_z_w = ray_hits_w[..., 2]

    num_envs, num_feet, num_samples, _ = sample_xy_w.shape
    sample_points = sample_xy_w.view(num_envs, num_feet * num_samples, 2)

    # Batched nearest-neighbor lookup from sample points to ray-hit XY points.
    dists = torch.cdist(sample_points, ray_xy_w)
    nearest_idx = torch.argmin(dists, dim=-1)
    terrain_z = torch.gather(ray_z_w, dim=1, index=nearest_idx)
    return terrain_z.view(num_envs, num_feet, num_samples)


def feet_center(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    d1: float = 0.05,
    d2: float = 0.0707,
    edge_height_threshold: float = -0.20,
    height_sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
) -> torch.Tensor:
    """MARG-style foothold edge penalty around each contacted foot.

    Computes 9 sample points around each foot and penalizes feet whose surrounding
    terrain points indicate a nearby edge or gap.
    """

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    # 1) Foot world positions: [E, F, 3]
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]

    # 2) Contact state from net contact forces: [E, F]
    contact_forces_w = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    feet_contact = torch.norm(contact_forces_w, dim=-1) > 1.0

    # 3) 9-point sampling pattern around each foot in world XY.
    offsets_xy = torch.tensor(
        [
            [0.0, 0.0],
            [d1, 0.0],
            [-d1, 0.0],
            [0.0, d1],
            [0.0, -d1],
            [d1, d1],
            [d1, -d1],
            [-d1, d1],
            [-d1, -d1],
        ],
        device=env.device,
        dtype=foot_pos_w.dtype,
    )
    sample_xy_w = foot_pos_w[:, :, None, 0:2] + offsets_xy[None, None, :, :]

    # 4) Query terrain height from scanner: [E, F, 9]
    terrain_z = _query_terrain_height_from_scanner(env, sample_xy_w, height_sensor_cfg=height_sensor_cfg)

    # 5) Relative heights around each foot.
    center_z = terrain_z[:, :, 0:1]
    rel_h = terrain_z - center_z

    # 6) Edge detection on cross and diagonal neighbors.
    type2_bad = rel_h[:, :, 1:5] < edge_height_threshold
    type3_bad = rel_h[:, :, 5:9] < edge_height_threshold
    n2 = type2_bad.any(dim=-1).float()
    n3 = type3_bad.any(dim=-1).float()

    # 7) Per-foot penalty, then sum over feet.
    per_foot_penalty = feet_contact.float() * (n2 + 2.0 * n3)
    penalty = torch.sum(per_foot_penalty, dim=1)

    return penalty


def feet_air_time_low_speed_gating(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    threshold: float = 0.5,
    speed_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward feet air time with strict low-speed gating.

    Computes feet air time reward, but completely disables it when robot speed is below a threshold.
    This ensures the reward is only active during active locomotion, not during static or near-static states.

    Args:
        env: The environment.
        sensor_cfg: Configuration for the contact sensor.
        command_name: Name of the command being tracked (e.g., "base_velocity").
        threshold: Minimum air time in seconds to earn reward.
        speed_threshold: Minimum command speed (xy norm) to enable the reward. Below this, reward is 0.

    Returns:
        The air time reward tensor.
    """
    from isaaclab.sensors import ContactSensor
    
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Compute first contact and air time
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    
    # Base reward calculation
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    
    # Strict low-speed gating: only reward if speed is above threshold
    cmd = env.command_manager.get_command(command_name)
    cmd_norm = torch.norm(cmd[:, :2], dim=1)
    reward *= (cmd_norm > speed_threshold).float()
    
    return reward


def feet_swing_alignment(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    max_swing_time: float = 0.5,
) -> torch.Tensor:
    """Reward stride length and direction alignment during swing phase.
    
    Rewards each foot for moving in the direction of the velocity command during swing phase.
    The reward combines:
    - Distance traveled in XY plane (longer strides = higher reward)
    - Alignment with velocity command direction (angle between motion and command)
    
    Each foot is evaluated independently. Feet are only rewarded during active swing phases.
    
    Args:
        env: The environment
        sensor_cfg: Contact sensor configuration
        asset_cfg: Robot asset configuration  
        command_name: Name of velocity command (default "base_velocity")
        max_swing_time: Maximum swing time to consider (beyond this, foot likely caught on obstacle)
        
    Returns:
        Reward tensor of shape (num_envs,) - sum of per-foot rewards
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get velocity command direction in XY plane
    cmd = env.command_manager.get_command(command_name)  # [num_envs, 3]
    cmd_xy = cmd[:, :2]  # [num_envs, 2]
    cmd_norm = torch.norm(cmd_xy, dim=1, keepdim=True)  # [num_envs, 1]
    cmd_dir = cmd_xy / (cmd_norm + 1e-6)  # [num_envs, 2] - normalized command direction
    
    # Get current foot positions in world frame (XY only)
    current_foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]  # [num_envs, num_feet, 2]
    
    # Initialize cache for last contact positions if not exists
    if not hasattr(env, "_foot_contact_pos_cache"):
        env._foot_contact_pos_cache = current_foot_pos.clone()
    
    # Get contact and air time information
    contact_forces_w = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # [num_envs, num_feet, 3]
    feet_in_contact = torch.norm(contact_forces_w, dim=-1) > 1.0  # [num_envs, num_feet]
    
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]  # [num_envs, num_feet]
    
    # Detect feet in valid swing phase: recently lifted off, within max swing time
    in_valid_swing = (last_air_time > 0.0) & (last_air_time <= max_swing_time)  # [num_envs, num_feet]
    
    # Calculate displacement from last contact position
    foot_displacement = current_foot_pos - env._foot_contact_pos_cache  # [num_envs, num_feet, 2]
    foot_distance = torch.norm(foot_displacement, dim=-1, keepdim=True)  # [num_envs, num_feet, 1]
    
    # Calculate directional alignment (cosine similarity with command direction)
    # Avoid division by zero for stationary feet
    foot_dir = foot_displacement / (foot_distance + 1e-6)  # [num_envs, num_feet, 2]
    alignment_cosine = torch.sum(foot_dir * cmd_dir.unsqueeze(1), dim=-1)  # [num_envs, num_feet]
    alignment_cosine = torch.clamp(alignment_cosine, -1.0, 1.0)  # Numerical stability
    
    # Reward = stride_distance * max(alignment, 0) * in_swing_phase
    # Only positive alignment contributes to reward
    alignment_reward = torch.clamp(alignment_cosine, min=0.0)  # [num_envs, num_feet]
    per_foot_reward = foot_distance.squeeze(-1) * alignment_reward * in_valid_swing.float()
    
    # Update cache: when feet touch down, record their current position for next swing phase
    env._foot_contact_pos_cache[feet_in_contact] = current_foot_pos[feet_in_contact]
    
    # Sum rewards across all feet
    total_reward = torch.sum(per_foot_reward, dim=1)
    
    return total_reward