import math
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp
from .mgdp_terrain import MGDP_TERRAIN_GENERATOR_CFG


def _seed_from_start_minute_second() -> int:
    now = datetime.now()
    return now.minute * 100 + now.second


GO2_MARG_ORACLE_RISK_TERRAIN_SEED = _seed_from_start_minute_second()

GO2_MODIFIED_DESCRIPTION_DIR = Path(__file__).resolve().parents[7] / "LidarSim2Real/go2_urdf_modified"
GO2_MODIFIED_URDF_PATH = GO2_MODIFIED_DESCRIPTION_DIR / "urdf/go2_description.urdf"
GO2_MODIFIED_DAE_DIR = GO2_MODIFIED_DESCRIPTION_DIR / "dae"


def _set_mgdp_terrain_seed(terrain_generator_cfg, seed: int) -> None:
    for sub_cfg in terrain_generator_cfg.sub_terrains.values():
        sub_cfg.seed = seed


def _active_subterrain_count(terrain_generator_cfg) -> int:
    return max(1, sum(float(sub_cfg.proportion) > 0.0 for sub_cfg in terrain_generator_cfg.sub_terrains.values()))


_set_mgdp_terrain_seed(MGDP_TERRAIN_GENERATOR_CFG, GO2_MARG_ORACLE_RISK_TERRAIN_SEED)


GO2_MARG_ORACLE_SPAWN_CFG = ROBOT_CFG.spawn.replace(asset_path=str(GO2_MODIFIED_URDF_PATH))
GO2_MARG_ORACLE_SPAWN_CFG.replace_asset(
    meshes_dir=str(GO2_MODIFIED_DAE_DIR),
    urdf_path=str(GO2_MODIFIED_URDF_PATH),
    mesh_link_name="dae",
)

GO2_MARG_ORACLE_ROBOT_CFG = ROBOT_CFG.replace(
    spawn=GO2_MARG_ORACLE_SPAWN_CFG,
    actuators={
        "GO2HV": ROBOT_CFG.actuators["GO2HV"].replace(
            # DelayedPDActuator samples an integer number of physics steps.
            # With sim dt = 0.005 s, 0..2 steps corresponds to about 0..10 ms.
            min_delay=0,
            max_delay=2,
        )
    }
)


# =========================== Scene Config ===========================
# ====================================================================
@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Scene config for the Go2 Marg-Oracle Risk Terrain task."""

    num_envs: int = 4096
    env_spacing: float = 2.5

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=MGDP_TERRAIN_GENERATOR_CFG,
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = GO2_MARG_ORACLE_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


class randomize_rigid_body_material_with_cache(mdp.randomize_rigid_body_material):
    """Randomize contact materials and cache the sampled friction summary."""

    def __call__(
        self,
        env,
        env_ids,
        static_friction_range,
        dynamic_friction_range,
        restitution_range,
        num_buckets,
        asset_cfg,
        make_consistent: bool = False,
    ):
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()

        total_num_shapes = self.asset.root_physx_view.max_shapes
        bucket_ids = torch.randint(0, num_buckets, (len(env_ids), total_num_shapes), device="cpu")
        material_samples = self.material_buckets[bucket_ids]

        materials = self.asset.root_physx_view.get_material_properties()
        if self.num_shapes_per_body is not None:
            for body_id in self.asset_cfg.body_ids:
                start_idx = sum(self.num_shapes_per_body[:body_id])
                end_idx = start_idx + self.num_shapes_per_body[body_id]
                materials[env_ids, start_idx:end_idx] = material_samples[:, start_idx:end_idx]
        else:
            materials[env_ids] = material_samples[:]
        self.asset.root_physx_view.set_material_properties(materials, env_ids)

        if not hasattr(env, "_terrain_friction"):
            env._terrain_friction = torch.full(
                (env.scene.num_envs, 1),
                env.cfg.scene.terrain.physics_material.static_friction,
                device=env.device,
            )
        env._terrain_friction[env_ids.to(env.device)] = material_samples[..., 0].mean(dim=1, keepdim=True).to(env.device)


def randomize_motor_strength(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    strength_distribution_params: tuple[float, float],
):
    """Randomize motor strength factors and apply them to actuator torque limits."""

    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    if asset_cfg.joint_ids == slice(None):
        global_joint_ids = torch.arange(asset.num_joints, device=asset.device)
    else:
        global_joint_ids = torch.tensor(asset_cfg.joint_ids, device=asset.device)

    if not hasattr(env, "_motor_strength"):
        env._motor_strength = torch.ones((env.scene.num_envs, asset.num_joints), device=asset.device)
    if not hasattr(env, "_motor_offset"):
        env._motor_offset = torch.zeros((env.scene.num_envs, asset.num_joints), device=asset.device)

    sampled_strength = torch.empty((len(env_ids), len(global_joint_ids)), device=asset.device).uniform_(
        strength_distribution_params[0], strength_distribution_params[1]
    )
    env._motor_strength[env_ids[:, None], global_joint_ids] = sampled_strength

    for actuator in asset.actuators.values():
        if isinstance(actuator.joint_indices, slice):
            actuator_global_ids = torch.arange(actuator.num_joints, device=asset.device)
        else:
            actuator_global_ids = torch.tensor(actuator.joint_indices, device=asset.device)

        local_mask = torch.isin(actuator_global_ids, global_joint_ids)
        if not torch.any(local_mask):
            continue

        local_ids = torch.nonzero(local_mask).view(-1)
        selected_global_ids = actuator_global_ids[local_ids]
        selected_strength = env._motor_strength[env_ids][:, selected_global_ids]

        if hasattr(actuator, "_effort_y1"):
            if not hasattr(actuator, "_default_effort_y1"):
                actuator._default_effort_y1 = actuator._effort_y1.clone()
                actuator._default_effort_y2 = actuator._effort_y2.clone()
            actuator._effort_y1[env_ids[:, None], local_ids] = (
                actuator._default_effort_y1[env_ids[:, None], local_ids] * selected_strength
            )
            actuator._effort_y2[env_ids[:, None], local_ids] = (
                actuator._default_effort_y2[env_ids[:, None], local_ids] * selected_strength
            )

        if hasattr(actuator, "effort_limit"):
            if not hasattr(actuator, "_default_effort_limit"):
                actuator._default_effort_limit = actuator.effort_limit.clone()
            actuator.effort_limit[env_ids[:, None], local_ids] = (
                actuator._default_effort_limit[env_ids[:, None], local_ids] * selected_strength
            )


def reset_base_with_terrain_orientation(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
):
    """Reset base position and orientation for directional MGDP terrains.

    The robot's initial yaw is aligned to +x direction with ±5° deviation.
    Position offset is ±10cm from spawn center in xy plane.
    """
    asset = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    else:
        env_ids = env_ids.to(asset.device)

    num_envs = len(env_ids)
    root_states = asset.data.default_root_state[env_ids].clone()
    pos_offsets = torch.zeros((num_envs, 3), device=asset.device)
    velocities = root_states[:, 7:13].clone()

    angle_tolerance = 5.0 * math.pi / 180.0
    yaws = torch.empty((num_envs,), device=asset.device).uniform_(-angle_tolerance, angle_tolerance)
    pos_offsets[:, 0:2] = torch.empty((num_envs, 2), device=asset.device).uniform_(-0.1, 0.1)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + pos_offsets
    orientations = math_utils.quat_from_euler_xyz(
        torch.zeros_like(yaws),
        torch.zeros_like(yaws),
        yaws,
    )

    # Apply root state through Articulation APIs.
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})

    stationary = DoneTerm(
        func=mdp.terminate_stationary_for_duration,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "base_velocity",
            "duration": 1.0,
            "distance_threshold": 1.50,
            "command_speed_threshold": 0.05,
        },
    )

    feet_on_base_plane_linear = DoneTerm(
        func=mdp.terminate_feet_on_base_plane_selected_terrains,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "force_threshold": 1.0,
            "plane_height_threshold": -0.2,
        },
    )


# =========================== Domain Randomization ===================
# ====================================================================
@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=randomize_rigid_body_material_with_cache,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.25),
            "dynamic_friction_range": (0.2, 1.25),
            "restitution_range": (0.0, 0.15),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (0.0, 1.5),
            "operation": "add",
        },
    )

    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    base_com_shift = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.015, 0.015), "z": (-0.01, 0.02)},
        },
    )

    motor_strength = EventTerm(
        func=randomize_motor_strength,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "strength_distribution_params": (0.8, 1.2),
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=reset_base_with_terrain_orientation,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


# =========================== Command Space ===============================
# =========================================================================
# Exposed command interface for this training task:
# all terrain columns use the same near-forward-only velocity command.
FORWARD_ONLY_LIN_VEL_X = (0.1, 0.5)
FORWARD_ONLY_LIN_VEL_X_LIMIT = (0.4, 1.5)
FORWARD_ONLY_LIN_VEL_Y = (-0.01, 0.01)
FORWARD_ONLY_ANG_VEL_Z = (-0.01, 0.01)


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=FORWARD_ONLY_LIN_VEL_X,
            lin_vel_y=FORWARD_ONLY_LIN_VEL_Y,
            ang_vel_z=FORWARD_ONLY_ANG_VEL_Z,
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=FORWARD_ONLY_LIN_VEL_X_LIMIT,
            lin_vel_y=FORWARD_ONLY_LIN_VEL_Y,
            ang_vel_z=FORWARD_ONLY_ANG_VEL_Z,
        ),
    )


# =========================== Left-Right Mirroring ========================
# =========================================================================
PROPRIO_DIM = 45
HISTORY_LENGTH = 6
# Modified Go2 URDF revolute joint order:
# FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, FR_thigh, RL_thigh, RR_thigh, FL_calf, FR_calf, RL_calf, RR_calf.
GO2_LEFT_JOINT_IDS = [0, 2, 4, 6, 8, 10]
GO2_RIGHT_JOINT_IDS = [1, 3, 5, 7, 9, 11]
GO2_HAA_JOINT_IDS = [0, 1, 2, 3]
# GridPatternCfg(ordering="xy") flattens as y rows x x columns: 11 rows * 17 cols = 187 heights.
TERRAIN_GRID_Y_POINTS = 11
TERRAIN_GRID_X_POINTS = 17


@torch.no_grad()
def compute_symmetric_states_go2_marg_oracle(env, obs, actions):
    """Left-right data augmentation for Go2 MARG-Oracle observations/actions."""

    obs_aug = _augment_go2_marg_oracle_obs(obs) if obs is not None else None
    actions_aug = _augment_go2_marg_oracle_actions(actions) if actions is not None else None
    return obs_aug, actions_aug


def _augment_go2_marg_oracle_obs(obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    batch_size = next(iter(obs.values())).shape[0]
    obs_aug = {}

    for key, value in obs.items():
        if key in ("policy_raw_obs", "policy"):
            mirrored = _transform_proprio_obs_left_right(value)
        elif key == "policy_history_obs":
            mirrored = _transform_proprio_history_left_right(value)
        elif key == "policy_terrain_obs":
            mirrored = _transform_terrain_map_left_right(value)
        elif key == "privileged_obs":
            mirrored = _transform_privileged_obs_left_right(value)
        else:
            mirrored = value

        obs_aug[key] = torch.empty(batch_size * 2, *value.shape[1:], device=value.device, dtype=value.dtype)
        obs_aug[key][:batch_size] = value
        obs_aug[key][batch_size:] = mirrored

    return obs_aug


def _augment_go2_marg_oracle_actions(actions: torch.Tensor) -> torch.Tensor:
    batch_size = actions.shape[0]
    actions_aug = torch.empty(batch_size * 2, *actions.shape[1:], device=actions.device, dtype=actions.dtype)
    actions_aug[:batch_size] = actions
    actions_aug[batch_size:] = _switch_go2_joints_left_right(actions, flip_haa=True)
    return actions_aug


def _transform_proprio_obs_left_right(obs: torch.Tensor) -> torch.Tensor:
    obs = obs.clone()
    device = obs.device

    # Layout: base_ang_vel(3), projected_gravity(3), command(3), joint_pos(12), joint_vel(12), last_action(12).
    obs[..., 0:3] = obs[..., 0:3] * torch.tensor([-1.0, 1.0, -1.0], device=device)
    obs[..., 3:6] = obs[..., 3:6] * torch.tensor([1.0, -1.0, 1.0], device=device)
    obs[..., 6:9] = obs[..., 6:9] * torch.tensor([1.0, -1.0, -1.0], device=device)
    obs[..., 9:21] = _switch_go2_joints_left_right(obs[..., 9:21], flip_haa=True)
    obs[..., 21:33] = _switch_go2_joints_left_right(obs[..., 21:33], flip_haa=True)
    obs[..., 33:45] = _switch_go2_joints_left_right(obs[..., 33:45], flip_haa=True)
    return obs


def _transform_proprio_history_left_right(obs: torch.Tensor) -> torch.Tensor:
    original_shape = obs.shape
    obs = obs.view(*original_shape[:-1], HISTORY_LENGTH, PROPRIO_DIM)
    obs = _transform_proprio_obs_left_right(obs)
    return obs.view(original_shape)


def _transform_terrain_map_left_right(obs: torch.Tensor) -> torch.Tensor:
    original_shape = obs.shape
    obs = obs.view(*original_shape[:-1], TERRAIN_GRID_Y_POINTS, TERRAIN_GRID_X_POINTS)
    obs = torch.flip(obs, dims=[-2])
    return obs.contiguous().view(original_shape)


def _transform_privileged_obs_left_right(obs: torch.Tensor) -> torch.Tensor:
    obs = obs.clone()
    device = obs.device

    # Layout: lin_vel(3), feet_contacts(4), mass_summary(4), friction(1), com_xy(2), force_xy(2), actuator_params(26).
    obs[..., 0:3] = obs[..., 0:3] * torch.tensor([1.0, -1.0, 1.0], device=device)
    obs[..., 3:7] = _switch_go2_feet_left_right(obs[..., 3:7])
    obs[..., 12:14] = obs[..., 12:14] * torch.tensor([1.0, -1.0], device=device)
    obs[..., 14:16] = obs[..., 14:16] * torch.tensor([1.0, -1.0], device=device)
    obs[..., 18:30] = _switch_go2_joints_left_right(obs[..., 18:30], flip_haa=False)
    obs[..., 30:42] = _switch_go2_joints_left_right(obs[..., 30:42], flip_haa=True)
    return obs


def _switch_go2_feet_left_right(feet_data: torch.Tensor) -> torch.Tensor:
    feet_data_switched = torch.zeros_like(feet_data)
    feet_data_switched[..., [0, 2]] = feet_data[..., [1, 3]]
    feet_data_switched[..., [1, 3]] = feet_data[..., [0, 2]]
    return feet_data_switched


def _switch_go2_joints_left_right(joint_data: torch.Tensor, flip_haa: bool) -> torch.Tensor:
    joint_data_switched = torch.zeros_like(joint_data)
    joint_data_switched[..., GO2_LEFT_JOINT_IDS] = joint_data[..., GO2_RIGHT_JOINT_IDS]
    joint_data_switched[..., GO2_RIGHT_JOINT_IDS] = joint_data[..., GO2_LEFT_JOINT_IDS]

    if flip_haa:
        joint_data_switched[..., GO2_HAA_JOINT_IDS] *= -1.0

    return joint_data_switched


# =========================== Observation Space ===========================
# =========================================================================
@configclass
class ObservationsCfg:
    """Observation layout for the Go2 Marg-Oracle velocity task."""

    @configclass
    class ProprioObsCfg(ObsGroup):
        """45D proprioceptive observation."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5)
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class ProprioHistoryObsCfg(ProprioObsCfg):
        """5(+1)-step proprio history, flattened to 270D."""

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5 + 1  # include current step
            self.flatten_history_dim = True

    @configclass
    class TerrainMapObsCfg(ObsGroup):
        """187D oracle terrain map."""

        terrain_map = ObsTerm(
            func=mdp.oracle_terrain_map,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "asset_cfg": SceneEntityCfg("robot")},
            clip=(-1.0, 1.0),
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedObsCfg(ObsGroup):
        """Privileged state set used by the critic / auxiliary estimators."""

        real_linear_velocity = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        feet_contacts = ObsTerm(
            func=mdp.feet_contact_labels,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), "threshold": 1.0},
        )
        critical_masses = ObsTerm(func=mdp.critical_mass_summary, params={"asset_cfg": SceneEntityCfg("robot")})
        friction = ObsTerm(func=mdp.terrain_friction_label)
        com_shift = ObsTerm(func=mdp.base_com_shift_xy, params={"asset_cfg": SceneEntityCfg("robot")})
        disturbance_force = ObsTerm(
            func=mdp.disturbance_force_xoy,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
        )
        actuator_params = ObsTerm(
            func=mdp.actuator_params_26,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
        )

        def __post_init__(self):
            self.concatenate_terms = True

    privileged_obs: PrivilegedObsCfg = PrivilegedObsCfg()

    @configclass
    class PolicyRawObsCfg(ProprioObsCfg):
        """Current policy raw obs, same as proprio obs."""

    policy_raw_obs: PolicyRawObsCfg = PolicyRawObsCfg()

    @configclass
    class PolicyHistoryObsCfg(ProprioHistoryObsCfg):
        """Current policy history obs, same as proprio history obs."""

    policy_history_obs: PolicyHistoryObsCfg = PolicyHistoryObsCfg()

    @configclass
    class PolicyTerrainObsCfg(TerrainMapObsCfg):
        """Current policy terrain obs, same as terrain map obs."""

    policy_terrain_obs: PolicyTerrainObsCfg = PolicyTerrainObsCfg()

    @configclass
    class CriticObsCfg(ProprioObsCfg):
        """Critic observation: proprio + terrain + privileged."""

        # terrain map is included in the critic obs for oracle methods
        terrain_map = ObsTerm(
            func=mdp.oracle_terrain_map,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "asset_cfg": SceneEntityCfg("robot")},
            clip=(-1.0, 1.0),
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        
        # privileged terms
        real_linear_velocity = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        feet_contacts = ObsTerm(
            func=mdp.feet_contact_labels,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), "threshold": 1.0},
        )
        critical_masses = ObsTerm(func=mdp.critical_mass_summary, params={"asset_cfg": SceneEntityCfg("robot")})
        friction = ObsTerm(func=mdp.terrain_friction_label)
        com_shift = ObsTerm(func=mdp.base_com_shift_xy, params={"asset_cfg": SceneEntityCfg("robot")})
        disturbance_force = ObsTerm(
            func=mdp.disturbance_force_xoy,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
        )
        actuator_params = ObsTerm(
            func=mdp.actuator_params_26,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    critic_obs: CriticObsCfg = CriticObsCfg()

    @configclass
    class PolicyCfg(PolicyRawObsCfg):
        """Compatibility group required by current RL wrappers."""

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(CriticObsCfg):
        """Compatibility group required by current RL wrappers."""

    critic: CriticCfg = CriticCfg()


# =========================== Action Space ================================
# =========================================================================
@configclass
class ActionsCfg:
    """12D joint action space for Go2 locomotion."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
    )


# =========================== Reward Config ===============================
# =========================================================================
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-1.0,
        params={"command_name": "base_velocity", "cmd_threshold": 0.05},
    )
    a_track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    a_track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # -- smoothness
    base_linear_velocity_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)

    # -- safety
    collisions = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_.*", ".*_hip", ".*_thigh", ".*_calf"]),
        },
    )

    # -- pose
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-0.2)
    joint_motion_limit = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )

    # -- footholds
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_low_speed_gating,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
            "speed_threshold": 0.1,
        },
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    feet_center = RewTerm(
        func=mdp.feet_center,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True
            ),
            "command_name": "base_velocity",
            "height_sensor_cfg": SceneEntityCfg("height_scanner"),
            "use_foot_local_raycast": True,
            "debug_vis": False,
            "debug_env_count": 1,
        },
    )


# =========================== Curriculum Config =============================
# =========================================================================
@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(
        func=mdp.lin_vel_cmd_levels,
        params={
            "reward_term_name": "a_track_lin_vel_xy",
            "lin_vel_x_delta": (0.1, 0.1),
            "lin_vel_y_delta": (0.0, 0.0),
        },
    )


# =========================== Task & Play Config ==========================
# =========================================================================
@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Go2 Marg-Oracle velocity task config."""

    scene: RobotSceneCfg = RobotSceneCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # MGDP heightfield terrains create denser contact patches than the box-based
        # risk terrains, so the default 2**26 collision stack can overflow on GPU.
        self.sim.physx.gpu_collision_stack_size = max(self.sim.physx.gpu_collision_stack_size, 2**27)

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

        self.scene.terrain.terrain_generator.num_rows = 10  # terrain levels
        self.scene.terrain.terrain_generator.num_cols = _active_subterrain_count(self.scene.terrain.terrain_generator)


PLAY_TERRAIN_TYPE = "mgdp"


def _play_terrain_generator_cfg(terrain_type: str):
    from .test_terrain import TEST_TERRAIN_GENERATOR_CFG

    terrain_generator_cfgs = {
        "mgdp": MGDP_TERRAIN_GENERATOR_CFG,
        "test": TEST_TERRAIN_GENERATOR_CFG,
    }
    terrain_type = terrain_type.strip().lower()
    if terrain_type not in terrain_generator_cfgs:
        valid_names = ", ".join(sorted(terrain_generator_cfgs))
        raise ValueError(f"Unknown play terrain type '{terrain_type}'. Valid options: {valid_names}.")
    return terrain_type, deepcopy(terrain_generator_cfgs[terrain_type])


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    """Play config for the Go2 Marg-Oracle velocity task."""

    play_terrain_type: str = PLAY_TERRAIN_TYPE

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 256
        play_terrain_type, terrain_generator_cfg = _play_terrain_generator_cfg(self.play_terrain_type)
        self.scene.terrain.terrain_generator = terrain_generator_cfg
        if play_terrain_type == "test":
            self.scene.terrain.terrain_generator.curriculum = False
            self.scene.terrain.terrain_generator.num_rows = 3
        self.scene.terrain.terrain_generator.num_cols = _active_subterrain_count(self.scene.terrain.terrain_generator)
        self.commands.base_velocity.ranges = deepcopy(self.commands.base_velocity.limit_ranges)
        self.events.push_robot = None
        self.terminations.feet_on_base_plane_linear = None
        self.rewards.feet_center.params["debug_vis"] = True
        self.rewards.feet_center.params["debug_env_count"] = 1
