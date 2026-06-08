import math

import torch
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg import RobotEnvCfg as BaseRobotEnvCfg


GO2_MARG_ORACLE_ROBOT_CFG = ROBOT_CFG.replace(
    actuators={
        "GO2HV": ROBOT_CFG.actuators["GO2HV"].replace(
            # DelayedPDActuator samples an integer number of physics steps.
            # With sim dt = 0.005 s, 0..2 steps corresponds to about 0..10 ms.
            min_delay=0,
            max_delay=2,
        )
    }
)


# =========================== Terrain Config ===========================
# ======================================================================
COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
    )



# =========================== Scene Config ===========================
# ====================================================================
@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Scene config for the Go2 Marg-Oracle velocity task."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=COBBLESTONE_ROAD_CFG,
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
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
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
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (0.0, 3.0),
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
            "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (0.0, 0.0)},
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
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
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




# =========================== Observation Space ===========================
# =========================================================================
def oracle_terrain_map(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Robot-centered oracle terrain map as relative heights: base_z - terrain_z."""

    sensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2]


def feet_contact_labels(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """4D foot-contact label from the vertical contact force.

    A foot is marked as in contact when |f_z| > threshold.
    """

    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    return (forces_z > threshold).float()


def critical_mass_summary(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """4D critical mass summary.

    Layout:
    - m0: base / trunk mass
    - m1: mean thigh mass
    - m2: mean calf mass
    - m3: added payload mass on the base relative to the default mass
    """

    asset = env.scene[asset_cfg.name]
    device = env.device
    masses = asset.root_physx_view.get_masses().to(device)

    cache_name = "_critical_mass_body_ids"
    if not hasattr(env, cache_name):
        base_ids, _ = asset.find_bodies("base")
        thigh_ids, _ = asset.find_bodies(".*_thigh")
        calf_ids, _ = asset.find_bodies(".*_calf")
        setattr(
            env,
            cache_name,
            {
                "base": base_ids,
                "thigh": thigh_ids,
                "calf": calf_ids,
            },
        )
    body_ids = getattr(env, cache_name)

    base_mass = masses[:, body_ids["base"]].sum(dim=1, keepdim=True)
    thigh_mass = masses[:, body_ids["thigh"]].mean(dim=1, keepdim=True)
    calf_mass = masses[:, body_ids["calf"]].mean(dim=1, keepdim=True)

    if hasattr(asset.data, "default_mass") and asset.data.default_mass is not None:
        default_base_mass = asset.data.default_mass[:, body_ids["base"]].sum(dim=1, keepdim=True).to(device)
        added_base_mass = base_mass - default_base_mass
    else:
        added_base_mass = torch.zeros_like(base_mass)

    return torch.cat((base_mass, thigh_mass, calf_mass, added_base_mass), dim=1)


def terrain_friction_label(env) -> torch.Tensor:
    """1D terrain friction label.

    This uses the configured terrain friction value. It is a stable scaffold until
    a fully randomized friction ground-truth path is added.
    """

    if hasattr(env, "_terrain_friction"):
        return env._terrain_friction
    friction = env.cfg.scene.terrain.physics_material.static_friction
    return torch.full((env.num_envs, 1), friction, device=env.device)


def base_com_shift_xy(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Base COM xy shift in the base frame."""

    asset = env.scene[asset_cfg.name]
    return asset.data.body_com_pos_b[:, 0, :2]


def disturbance_force_xoy(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Applied disturbance force on the base body in x-y."""

    asset = env.scene[asset_cfg.name]
    return asset._external_force_b[:, asset_cfg.body_ids[0], :2]


def actuator_params_26(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """26D actuator/state parameter vector.

    Layout:
    - 1 kp
    - 1 kd
    - 12 motor strength values
    - 12 motor offset values
    """

    asset = env.scene[asset_cfg.name]

    # Prefer the current simulated gains. Fall back to defaults if the live fields are unavailable.
    if hasattr(asset.data, "joint_stiffness") and hasattr(asset.data, "joint_damping"):
        kp = asset.data.joint_stiffness[:, asset_cfg.joint_ids].mean(dim=1, keepdim=True)
        kd = asset.data.joint_damping[:, asset_cfg.joint_ids].mean(dim=1, keepdim=True)
    else:
        kp = asset.data.default_joint_stiffness[:, asset_cfg.joint_ids].mean(dim=1, keepdim=True)
        kd = asset.data.default_joint_damping[:, asset_cfg.joint_ids].mean(dim=1, keepdim=True)

    # Prefer explicit motor strength randomization factors if they are tracked on the environment.
    if hasattr(env, "_motor_strength"):
        motor_strength = env._motor_strength[:, asset_cfg.joint_ids]
    else:
        effort = asset.data.joint_effort_limits[:, asset_cfg.joint_ids]
        motor_strength = effort / effort.mean(dim=1, keepdim=True)

    # Motor offset is an offset/randomization term, not the default nominal pose.
    if hasattr(env, "_motor_offset"):
        motor_offset = env._motor_offset[:, asset_cfg.joint_ids]
    else:
        motor_offset = torch.zeros_like(motor_strength)

    return torch.cat((kp, kd, motor_strength, motor_offset), dim=1)



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

    proprio_obs: ProprioObsCfg = ProprioObsCfg()

    @configclass
    class ProprioHistoryObsCfg(ObsGroup):
        """5-step proprio history, flattened to 225D."""

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
            self.history_length = 5
            self.flatten_history_dim = True

    proprio_history_obs: ProprioHistoryObsCfg = ProprioHistoryObsCfg()

    @configclass
    class TerrainMapObsCfg(ObsGroup):
        """187D oracle terrain map."""

        terrain_map = ObsTerm(
            func=oracle_terrain_map,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "asset_cfg": SceneEntityCfg("robot")},
            clip=(-1.0, 5.0),
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    terrain_map_obs: TerrainMapObsCfg = TerrainMapObsCfg()

    @configclass
    class PrivilegedObsCfg(ObsGroup):
        """Privileged state set used by the critic / auxiliary estimators."""

        real_linear_velocity = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        feet_contacts = ObsTerm(
            func=feet_contact_labels,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), "threshold": 1.0},
        )
        critical_masses = ObsTerm(func=critical_mass_summary, params={"asset_cfg": SceneEntityCfg("robot")})
        friction = ObsTerm(func=terrain_friction_label)
        com_shift = ObsTerm(func=base_com_shift_xy, params={"asset_cfg": SceneEntityCfg("robot")})
        disturbance_force = ObsTerm(
            func=disturbance_force_xoy,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
        )
        actuator_params = ObsTerm(
            func=actuator_params_26,
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
            func=oracle_terrain_map,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "asset_cfg": SceneEntityCfg("robot")},
            clip=(-1.0, 5.0),
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        
        # privileged terms
        real_linear_velocity = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        feet_contacts = ObsTerm(
            func=feet_contact_labels,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), "threshold": 1.0},
        )
        critical_masses = ObsTerm(func=critical_mass_summary, params={"asset_cfg": SceneEntityCfg("robot")})
        friction = ObsTerm(func=terrain_friction_label)
        com_shift = ObsTerm(func=base_com_shift_xy, params={"asset_cfg": SceneEntityCfg("robot")})
        disturbance_force = ObsTerm(
            func=disturbance_force_xoy,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
        )
        actuator_params = ObsTerm(
            func=actuator_params_26,
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
    """12D joint action space for Go2 locomotion.
        q_t^* = q_dot + a_t
    """
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
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # -- smoothness
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
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
        func=mdp.feet_air_time,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    feet_center = RewTerm(
        func=mdp.feet_center,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
        },
    )




# =========================== Task & Play Config ==========================
# =========================================================================
@configclass
class RobotEnvCfg(BaseRobotEnvCfg):
    """Go2 Marg-Oracle velocity task config."""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=8192, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventCfg = EventCfg()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    """Play config for the Go2 Marg-Oracle velocity task."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.terrain.terrain_generator.num_rows = 3
        self.scene.terrain.terrain_generator.num_cols = 5
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
