from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass


# Terrain parameter table: each terrain type has 8 difficulty levels
# Parameters are [level_1_value, level_4_value, level_8_value]
TERRAIN_PARAMS_TABLE = {
    "single_gap": {
        "gap_range": [0.10, 0.40, 0.60],
        "height_range": [0.0, 0.0, 0.0],
    },
    "stones_everywhere": {
        "gap_range": [0.05, 0.15, 0.40],
        "stone_size_range": [0.80, 0.60, 0.30],
        "height_range": [0.08, 0.24, 0.36],
    },
    "stones_2rows": {
        "gap_range": [0.10, 0.15, 0.30],
        "stone_size_range": [0.80, 0.55, 0.30],
        "height_range": [0.08, 0.24, 0.44],
    },
    "stones_balance": {
        "gap_range": [0.10, 0.50, 0.40],
        "stone_size_range": [0.80, 0.55, 0.30],
        "height_range": [0.08, 0.24, 0.44],
    },
    "beams_balance": {
        "gap_range": [0.10, 0.20, 0.40],
        "beam_width_range": [0.30, 0.25, 0.20],
        "height_range": [0.08, 0.24, 0.44],
    },
    "air_beams_balance": {
        "gap_range": [0.10, 0.20, 0.40],
        "beam_width_range": [0.30, 0.30, 0.20],
        "height_range": [0.04, 0.24, 0.44],
    },
}

TERRAIN_TYPES = list(TERRAIN_PARAMS_TABLE.keys())


def _lerp(v0: float, v1: float, ratio: float) -> float:
    return float(v0 + ratio * (v1 - v0))


def _lerp_from_keyframes(keyframes: list[float], difficulty: float) -> float:
    """Interpolate value from 3 keyframes at difficulty levels [1, 4, 8].
    
    Args:
        keyframes: [value_at_level_1, value_at_level_4, value_at_level_8]
        difficulty: normalized difficulty in [0.0, 1.0] range
    
    Returns:
        Interpolated value
    """
    # Map difficulty to level: 0.0 -> level 1, 1.0 -> level 8
    level = 1.0 + difficulty * 7.0  # 0->1, 1->8
    
    if level <= 4.0:
        # Interpolate between level 1 and level 4
        ratio = (level - 1.0) / 3.0  # 0->0, 3->1
        return _lerp(keyframes[0], keyframes[1], ratio)
    else:
        # Interpolate between level 4 and level 8
        ratio = (level - 4.0) / 4.0  # 0->0, 4->1
        return _lerp(keyframes[1], keyframes[2], ratio)


def _make_box_xy(
    *,
    size_x: float,
    size_y: float,
    top_z: float,
    height: float,
    center_x: float,
    center_y: float,
) -> trimesh.Trimesh:
    z_center = top_z - 0.5 * height
    dims = (size_x, size_y, height)
    transform = trimesh.transformations.translation_matrix((center_x, center_y, z_center))
    return trimesh.creation.box(dims, transform)


def _terrain_type_from_seed(seed: int | None) -> str:
    """Select terrain type from seed to ensure consistency across environments.
    
    Different seeds will map to different terrain types, allowing diverse terrain
    types to coexist in the training environment.
    """
    base = 0 if seed is None else int(seed)
    terrain_idx = base % len(TERRAIN_TYPES)
    return TERRAIN_TYPES[terrain_idx]


def _rng_from_seed(seed: int | None, difficulty: float, terrain_type: str) -> np.random.Generator:
    base = 0 if seed is None else int(seed)
    diff_term = int(round(float(difficulty) * 1_000_000.0))
    terrain_idx = TERRAIN_TYPES.index(terrain_type)
    hashed = (base * 1_000_003 + terrain_idx * 7_919 + diff_term) & 0xFFFFFFFF
    return np.random.default_rng(hashed)


def _snap_to_nearest(values: np.ndarray, target: float) -> float:
    if values.size == 0:
        return float(target)
    idx = int(np.argmin(np.abs(values - target)))
    return float(values[idx])


def marg_risk_terrain(
    difficulty: float, cfg: "MargRiskTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate risk-inspired MARG terrain with 6 terrain types at multiple difficulty levels.

    Different seeds produce different terrain types. Within each type, difficulty modulates
    parameters between Level 1 (easiest), Level 4 (medium), and Level 8 (hardest).
    """

    sx, sy = cfg.size
    terrain_type = getattr(cfg, "terrain_type", None) or _terrain_type_from_seed(seed=getattr(cfg, "seed", None))
    params = TERRAIN_PARAMS_TABLE[terrain_type]
    rng = _rng_from_seed(seed=getattr(cfg, "seed", None), difficulty=difficulty, terrain_type=terrain_type)
    
    # Get interpolated parameters based on difficulty
    gap_size = _lerp_from_keyframes(params["gap_range"], difficulty)
    
    meshes: list[trimesh.Trimesh] = []

    # Base ground layer covering the entire terrain
    base_ground = _make_box_xy(
        size_x=sx,
        size_y=sy,
        top_z=-0.22,
        height=cfg.base_thickness,
        center_x=0.5 * sx,
        center_y=0.5 * sy,
    )
    meshes.append(base_ground)

    # Shared spawn region keeps resets stable while still forcing traversal to harder zones.
    # Spawn platform: use configurable size/height/center from cfg
    spawn_size_x, spawn_size_y = cfg.spawn_size
    spawn_center_y = float(cfg.spawn_center[1]) * sy
    if terrain_type == "single_gap" or terrain_type == "stones_everywhere":
        spawn_center_x = float(cfg.spawn_center[0]) * sx
    else:
        spawn_center_x = 0.0
        
        
    
    

    # For stones_everywhere, align spawn center to the nearest stone-grid center so the
    # merged 3x3/5x5/... platform center is exactly the middle tile center.
    if terrain_type == "stones_everywhere":
        stone_size_for_spawn = _lerp_from_keyframes(params["stone_size_range"], difficulty)
        pitch_for_spawn = stone_size_for_spawn + gap_size
        x_grid = np.arange(0.7, sx - 0.7, pitch_for_spawn)
        y_grid = np.arange(0.4, sy - 0.4, pitch_for_spawn)
        spawn_center_x = _snap_to_nearest(x_grid, spawn_center_x)
        spawn_center_y = _snap_to_nearest(y_grid, spawn_center_y)

    # Keep spawn top_z behavior unchanged; use fixed 1m spawn thickness.
    spawn_height = 1.0
    terrain_max_height = _lerp_from_keyframes(params.get("height_range", [0.0, 0.0, 0.0]), difficulty)
    spawn_top_z = cfg.base_thickness + 0.5 * terrain_max_height
    

    spawn = _make_box_xy(
        size_x=spawn_size_x,
        size_y=spawn_size_y,
        top_z=spawn_top_z,
        height=spawn_height,
        center_x=spawn_center_x,
        center_y=spawn_center_y,
    )
    meshes.append(spawn)


    if terrain_type == "single_gap":
        # Compute spawn boundaries

        # Four rectangular blocks with gap_size distance from spawn edges
        axial_expansion = 1.5
        radial_expansion_1 = 3*1.5 + 2*gap_size

        # y positive block 1
        meshes.append(
            _make_box_xy(
                size_x=radial_expansion_1, 
                size_y=axial_expansion,
                top_z=cfg.base_thickness,
                height=cfg.base_thickness,
                center_x=0.5 * sx,
                center_y=0.5 * sy + 0.75 + gap_size + 0.5 * axial_expansion,
            )
        )
        
        # y negative block 1
        meshes.append(
            _make_box_xy(
                size_x=radial_expansion_1, 
                size_y=axial_expansion,
                top_z=cfg.base_thickness,
                height=cfg.base_thickness,
                center_x=0.5 * sx,
                center_y=0.5 * sy -0.75 - gap_size - 0.5*axial_expansion,
            )
        )

        # x positive block 1
        meshes.append(
            _make_box_xy(
                size_x=axial_expansion,
                size_y=radial_expansion_1,
                top_z=cfg.base_thickness,
                height=cfg.base_thickness,
                center_x=0.5 * sx + 0.75 + gap_size + 0.5 * axial_expansion,
                center_y=0.5 * sy,
            )
        )
        
        # x negative block 1
        meshes.append(
            _make_box_xy(
                size_x=axial_expansion,  
                size_y=radial_expansion_1,
                top_z=cfg.base_thickness,
                height=cfg.base_thickness,
                center_x=0.5 * sx -0.75 - gap_size - 0.5 * axial_expansion,
                center_y=0.5 * sy,
            )
        )
        
        # Four rectangular blocks with gap_size distance from spawn edges
        axial_expansion_2 = (sx - radial_expansion_1 - 2*gap_size)/2
        radial_expansion_2 = sx
        

        # y positive block 2
        meshes.append(
            _make_box_xy(
                size_x=radial_expansion_2, 
                size_y=axial_expansion_2,
                top_z=cfg.base_thickness,
                height=cfg.base_thickness,
                center_x=0.5 * sx,
                center_y=0.5 * sy + 0.75 + gap_size + axial_expansion + gap_size + 0.5 * axial_expansion_2,
            )
        )

        # y negative block 2
        meshes.append(
            _make_box_xy(
                size_x=radial_expansion_2, 
                size_y=axial_expansion_2,
                top_z=cfg.base_thickness,
                height=cfg.base_thickness,
                center_x=0.5 * sx,
                center_y=0.5 * sy -0.75 - gap_size - axial_expansion - gap_size - 0.5 * axial_expansion_2,
            )
        )
        
        # x positive block 2
        meshes.append(
            _make_box_xy(
                size_x=axial_expansion_2,
                size_y=radial_expansion_2,
                top_z=cfg.base_thickness,
                height=cfg.base_thickness,
                center_x=0.5 * sx + 0.75 + gap_size + axial_expansion + gap_size + 0.5 * axial_expansion_2,
                center_y=0.5 * sy,
            )
        )
        
        # x negative block 2
        meshes.append(
            _make_box_xy(
                size_x=axial_expansion_2,  
                size_y=radial_expansion_2,
                top_z=cfg.base_thickness,
                height=cfg.base_thickness,
                center_x=0.5 * sx -0.75 - gap_size - axial_expansion - gap_size - 0.5 * axial_expansion_2,
                center_y=0.5 * sy,
            )
        )


    elif terrain_type == "stones_everywhere":
        stone_size = _lerp_from_keyframes(params["stone_size_range"], difficulty)
        max_height = _lerp_from_keyframes(params["height_range"], difficulty)
        pitch = stone_size + gap_size
        x_start = 0.7
        x_end = sx - 0.7
        y_start = 0.4
        y_end = sy - 0.4

        # Merge central stones into one flat block.
        # Start from 3x3 (9 tiles) and keep odd expansion (5x5, 7x7, ...)
        # until merged platform area is at least 1.4m x 1.4m.
        min_center_area = 1.4 * 1.4
        merged_count = 3
        merged_side = merged_count * stone_size + (merged_count - 1) * gap_size
        while merged_side * merged_side < min_center_area:
            merged_count += 2
            merged_side = merged_count * stone_size + (merged_count - 1) * gap_size
        center_x = spawn_center_x
        center_y = spawn_center_y

        x_vals = np.arange(x_start, x_end, pitch)
        y_vals = np.arange(y_start, y_end, pitch)
        for cx in x_vals:
            for cy in y_vals:
                # Skip stones that would overlap the merged center platform.
                if (
                    abs(float(cx) - center_x) < 0.5 * (merged_side + stone_size)
                    and abs(float(cy) - center_y) < 0.5 * (merged_side + stone_size)
                ):
                    continue
                top_z = cfg.base_thickness + rng.uniform(0.0, max_height)
                meshes.append(
                    _make_box_xy(
                        size_x=stone_size,
                        size_y=stone_size,
                        top_z=top_z,
                        height=1,
                        center_x=float(cx),
                        center_y=float(cy),
                    )
                )

        # spawn
        meshes.append(
            _make_box_xy(
                size_x=merged_side,
                size_y=merged_side,
                top_z=spawn_top_z,
                height=1,
                center_x=center_x,
                center_y=center_y,
            )
        )

    elif terrain_type == "stones_2rows":
        stone_size = _lerp_from_keyframes(params["stone_size_range"], difficulty)
        max_height = _lerp_from_keyframes(params["height_range"], difficulty)
        pitch = stone_size + gap_size
        x_vals = np.arange(0.8, sx - 0.8, pitch)
        y_offsets = (-0.3, 0.3)
        for cx in x_vals:
            for offset in y_offsets:
                cy = float(0.5 * sy + offset)
                top_z = cfg.base_thickness + rng.uniform(0.0, max_height)
                meshes.append(
                    _make_box_xy(
                        size_x=stone_size,
                        size_y=stone_size,
                        top_z=top_z,
                        height=max(cfg.base_thickness * 0.7, top_z + cfg.base_thickness),
                        center_x=float(cx),
                        center_y=cy,
                    )
                )

    elif terrain_type == "stones_balance":
        stone_size = _lerp_from_keyframes(params["stone_size_range"], difficulty)
        max_height = _lerp_from_keyframes(params["height_range"], difficulty)
        stone_x = stone_size
        # stone_y = max(0.20, stone_size * 0.45)
        stone_y = stone_size
        pitch = stone_x + gap_size
        x_vals = np.arange(0.7, sx - 0.7, pitch)
        for cx in x_vals:
            top_z = cfg.base_thickness + rng.uniform(0.0, max_height)
            meshes.append(
                _make_box_xy(
                    size_x=stone_x,
                    size_y=stone_y,
                    top_z=top_z,
                    height=max(cfg.base_thickness * 0.7, top_z + cfg.base_thickness),
                    center_x=float(cx),
                    center_y=0.5 * sy,
                )
            )

    elif terrain_type == "beams_balance":
        beam_width = _lerp_from_keyframes(params["beam_width_range"], difficulty)
        max_height = _lerp_from_keyframes(params["height_range"], difficulty)
        beam_len = 0.9
        pitch = beam_width + gap_size
        x_vals = np.arange(0.7, sx - 0.7, pitch)
        for cx in x_vals:
            top_z = cfg.base_thickness + rng.uniform(0.0, max_height)
            meshes.append(
                _make_box_xy(
                    size_x=beam_width,
                    size_y=beam_len,
                    top_z=top_z,
                    height=max(cfg.base_thickness * 0.7, top_z + cfg.base_thickness),
                    center_x=float(cx),
                    center_y=0.5 * sy,
                )
            )

    else:  # air_beams_balance
        beam_width = _lerp_from_keyframes(params["beam_width_range"], difficulty)
        max_height = _lerp_from_keyframes(params["height_range"], difficulty)
        beam_len = 0.8
        pitch = beam_width + gap_size
        x_vals = np.arange(0.8, sx - 0.8, pitch)
        for cx in x_vals:
            cy = float(0.5 * sy + rng.uniform(-0.15, 0.15))
            top_z = cfg.base_thickness + rng.uniform(0.02, max_height)
            meshes.append(
                _make_box_xy(
                    size_x=beam_width,
                    size_y=beam_len,
                    top_z=top_z,
                    height=max(cfg.base_thickness * 0.7, top_z + cfg.base_thickness),
                    center_x=float(cx),
                    center_y=cy,
                )
            )

    # Keep a small clearance above spawn surface to avoid initial interpenetration.
    origin = np.array([spawn_center_x, spawn_center_y, spawn_top_z + 0.02])
    return meshes, origin


@configclass
class MargRiskTerrainCfg(SubTerrainBaseCfg):
    function = marg_risk_terrain

    level_count: int = 8
    base_thickness: float = 0.08
    terrain_type: str | None = None

    gap_range: tuple[float, float] = (0.10, 0.60)
    stone_size_range: tuple[float, float] = (0.80, 0.24)
    beam_width_range: tuple[float, float] = (0.30, 0.12)
    height_range: tuple[float, float] = (0.00, 0.44)
    # Spawn platform customization:
    # - `spawn_size`: (width_x, width_y) in meters
    # - `spawn_height`: height in meters (kept at 1.0m in terrain generation)
    # - `spawn_center`: fractions (0..1) of terrain size (sx, sy), default center (0.5, 0.5)
    spawn_size: tuple[float, float] = (2, 2)
    spawn_height: float | None = 1.0
    spawn_center: tuple[float, float] = (0.5, 0.5)


MARG_RISK_TERRAIN_GENERATOR_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=20.0,
    num_rows=8,
    num_cols=16,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    curriculum=True,
    use_cache=False,
    sub_terrains={
        "single_gap": MargRiskTerrainCfg(proportion=1 / 6, terrain_type="single_gap"),
        "stones_everywhere": MargRiskTerrainCfg(proportion=1 / 6, terrain_type="stones_everywhere"),
        "stones_2rows": MargRiskTerrainCfg(proportion=0 / 6, terrain_type="stones_2rows"),
        "stones_balance": MargRiskTerrainCfg(proportion=0 / 6, terrain_type="stones_balance"),
        "beams_balance": MargRiskTerrainCfg(proportion=0 / 6, terrain_type="beams_balance"),
        "air_beams_balance": MargRiskTerrainCfg(proportion=0 / 6, terrain_type="air_beams_balance"),
    },
)
