from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass


MGDP_GAP_PARKOUR_WEIGHTS = {
    "single_gap": 0.002,
    "step_stone": 0.101,
    "stones_2rows": 0.101,
    "stones_1row": 0.101,
    "single_bridge": 0.101,
    "air_beams": 0.101,
    "air_stone": 0.101,
    "hurdle": 0.101,
    "ramp": 0.101,
    "corridor": 1.1,
}

MGDP_MIX_WEIGHTS = {
    "slope_down": 0.2,
    "pyramid": 0.2,
    "stairs_down": 0.2,
    "stairs_up": 0.2,
    "discrete_obstacles": 1.1,
    "hurdle": 0.2,
    "gap": 1.2,
    "ramp": 1.1,
    "new_stairs_up": 0.3,
    "pit": 1.0,
}


class _SubTerrain:
    def __init__(self, size: tuple[float, float], horizontal_scale: float, vertical_scale: float):
        self.horizontal_scale = float(horizontal_scale)
        self.vertical_scale = float(vertical_scale)
        self.width = int(round(size[0] / self.horizontal_scale)) + 1
        self.length = int(round(size[1] / self.horizontal_scale)) + 1
        self.height_field_raw = np.zeros((self.width, self.length), dtype=np.int16)


def _rng_from_seed(seed: int | None, difficulty: float, terrain_type: str) -> np.random.Generator:
    base = 0 if seed is None else int(seed)
    diff_term = int(round(float(difficulty) * 1_000_000.0))
    type_term = sum((i + 1) * ord(ch) for i, ch in enumerate(terrain_type))
    return np.random.default_rng((base * 1_000_003 + diff_term * 9_176 + type_term) & 0xFFFFFFFF)


def _height_to_units(terrain: _SubTerrain, height: float) -> int:
    return int(round(height / terrain.vertical_scale))


def _meter_to_index(terrain: _SubTerrain, value: float) -> int:
    return int(round(value / terrain.horizontal_scale))


def _choice(values: np.ndarray, rng: np.random.Generator, default: int = 0) -> int:
    if values.size == 0:
        return int(default)
    return int(rng.choice(values))


def _fill_rect(terrain: _SubTerrain, x0: int, x1: int, y0: int, y1: int, height_units: int) -> None:
    x0 = int(np.clip(x0, 0, terrain.width))
    x1 = int(np.clip(x1, 0, terrain.width))
    y0 = int(np.clip(y0, 0, terrain.length))
    y1 = int(np.clip(y1, 0, terrain.length))
    if x1 > x0 and y1 > y0:
        terrain.height_field_raw[x0:x1, y0:y1] = height_units


def _clear_spawn_platform(terrain: _SubTerrain, platform_size: float, height_units: int = 0) -> tuple[int, int, int, int]:
    platform = max(1, _meter_to_index(terrain, platform_size))
    x0 = (terrain.width - platform) // 2
    x1 = (terrain.width + platform) // 2
    y0 = (terrain.length - platform) // 2
    y1 = (terrain.length + platform) // 2
    _fill_rect(terrain, x0, x1, y0, y1, height_units)
    return x0, x1, y0, y1


def _random_uniform_terrain(
    terrain: _SubTerrain,
    rng: np.random.Generator,
    min_height: float,
    max_height: float,
    step: float = 0.005,
    downsampled_scale: float = 0.5,
) -> None:
    min_h = _height_to_units(terrain, min_height)
    max_h = _height_to_units(terrain, max_height)
    step_h = max(1, abs(_height_to_units(terrain, step)))
    heights = np.arange(min_h, max_h + step_h, step_h, dtype=np.int16)
    if heights.size == 0:
        return

    ds = max(terrain.horizontal_scale, float(downsampled_scale))
    ds_rows = max(2, int(round(terrain.width * terrain.horizontal_scale / ds)))
    ds_cols = max(2, int(round(terrain.length * terrain.horizontal_scale / ds)))
    coarse = rng.choice(heights, size=(ds_rows, ds_cols)).astype(np.float32)

    x_src = np.linspace(0.0, 1.0, ds_rows)
    y_src = np.linspace(0.0, 1.0, ds_cols)
    x_dst = np.linspace(0.0, 1.0, terrain.width)
    y_dst = np.linspace(0.0, 1.0, terrain.length)
    rows = np.stack([np.interp(y_dst, y_src, row) for row in coarse], axis=0)
    full = np.stack([np.interp(x_dst, x_src, rows[:, j]) for j in range(rows.shape[1])], axis=1)
    terrain.height_field_raw += np.rint(full).astype(np.int16)


def _pyramid_sloped_terrain(terrain: _SubTerrain, slope: float, platform_size: float = 3.0) -> None:
    x = np.arange(0, terrain.width)
    y = np.arange(0, terrain.length)
    center_x = terrain.width // 2
    center_y = terrain.length // 2
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = (center_x - np.abs(center_x - xx)) / max(center_x, 1)
    yy = (center_y - np.abs(center_y - yy)) / max(center_y, 1)
    max_height = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * (terrain.width / 2))
    terrain.height_field_raw += (max_height * xx.reshape(terrain.width, 1) * yy.reshape(1, terrain.length)).astype(
        np.int16
    )

    platform = max(1, _meter_to_index(terrain, platform_size) // 2)
    x0, x1 = center_x - platform, center_x + platform
    y0, y1 = center_y - platform, center_y + platform
    min_h = min(int(terrain.height_field_raw[x0, y0]), 0)
    max_h = max(int(terrain.height_field_raw[x0, y0]), 0)
    terrain.height_field_raw = np.clip(terrain.height_field_raw, min_h, max_h).astype(np.int16)
    _fill_rect(terrain, x0, x1, y0, y1, int(terrain.height_field_raw[x0, y0]))


def _pyramid_stairs_terrain(terrain: _SubTerrain, step_width: float, step_height: float, platform_size: float = 2.0):
    step_w = max(1, _meter_to_index(terrain, step_width))
    step_h = _height_to_units(terrain, step_height)
    platform = max(1, _meter_to_index(terrain, platform_size))
    height = 0
    start_x = 0
    stop_x = terrain.width
    start_y = 0
    stop_y = terrain.length
    while (stop_x - start_x) > platform and (stop_y - start_y) > platform:
        start_x += step_w
        stop_x -= step_w
        start_y += step_w
        stop_y -= step_w
        height += step_h
        _fill_rect(terrain, start_x, stop_x, start_y, stop_y, height)


def _discrete_obstacles_terrain(
    terrain: _SubTerrain,
    rng: np.random.Generator,
    max_height: float,
    min_size: float = 1.0,
    max_size: float = 2.5,
    num_rects: int = 20,
    platform_size: float = 3.0,
) -> None:
    max_h = max(1, abs(_height_to_units(terrain, max_height)))
    min_s = max(1, _meter_to_index(terrain, min_size))
    max_s = max(min_s + 1, _meter_to_index(terrain, max_size))
    heights = np.array([-max_h, -max_h // 2, max_h // 2, max_h], dtype=np.int16)
    for _ in range(num_rects):
        width = int(rng.integers(min_s, max_s))
        length = int(rng.integers(min_s, max_s))
        x0 = int(rng.integers(0, max(1, terrain.width - width)))
        y0 = int(rng.integers(0, max(1, terrain.length - length)))
        _fill_rect(terrain, x0, x0 + width, y0, y0 + length, int(rng.choice(heights)))
    _clear_spawn_platform(terrain, platform_size)


def _pit_terrain(terrain: _SubTerrain, depth: float, platform_size: float = 4.0) -> None:
    depth_units = -abs(_height_to_units(terrain, depth))
    platform = max(1, _meter_to_index(terrain, platform_size) // 2)
    cx = terrain.width // 2
    cy = terrain.length // 2
    _fill_rect(terrain, cx - platform, cx + platform, cy - platform, cy + platform, depth_units)


def _parkour_gap_terrain(terrain: _SubTerrain, difficulty: float, depth: float = 0.6, platform_size: float = 2.0):
    depth_units = -abs(_height_to_units(terrain, depth))
    gap = int(np.clip(difficulty / terrain.horizontal_scale, 2, 12))
    platform = max(1, _meter_to_index(terrain, platform_size))
    end_y = int(terrain.length - platform / 8)
    center_x = terrain.width // 2

    _fill_rect(terrain, platform, center_x, 0, end_y, depth_units)
    _fill_rect(terrain, platform + gap, center_x - gap, gap, end_y - gap, 0)

    start_x = center_x + platform // 2
    _fill_rect(terrain, start_x, terrain.width, 0, end_y, depth_units)
    _fill_rect(terrain, start_x + gap, terrain.width - gap, gap, end_y - gap, 0)
    _clear_spawn_platform(terrain, platform_size)


def _single_bridge_terrain(terrain: _SubTerrain, difficulty: float, depth: float = 0.6, platform_size: float = 2.0):
    depth_units = -abs(_height_to_units(terrain, depth))
    terrain.height_field_raw[:, :] = depth_units
    platform = max(1, _meter_to_index(terrain, platform_size))
    bridge_width = max(3, _meter_to_index(terrain, max(0.2, 0.8 - 0.6 * difficulty)))
    y0 = terrain.length // 2 - bridge_width // 2
    _fill_rect(terrain, 0, terrain.width, y0, y0 + bridge_width, 0)
    _clear_spawn_platform(terrain, platform_size)


def _stones_everywhere(
    terrain: _SubTerrain,
    rng: np.random.Generator,
    difficulty: float,
    two_rows: bool = False,
    one_row: bool = False,
    depth: float = 0.6,
    platform_size: float = 2.0,
):
    depth_units = -abs(_height_to_units(terrain, depth))
    terrain.height_field_raw[:, :] = depth_units
    stone_size_m = max(0.22, 0.8 - 0.5 * difficulty)
    stone = int(np.clip(_meter_to_index(terrain, stone_size_m), 4, 18))
    gap = int(np.clip(_meter_to_index(terrain, 0.1 + 0.35 * difficulty), 1, 9))
    max_h = int(np.clip(_height_to_units(terrain, 0.05 + 0.18 * difficulty), 1, 40))
    heights = np.arange(0, max_h + 1, step=max(1, max_h // 6), dtype=np.int16)
    platform = max(1, _meter_to_index(terrain, platform_size))
    y_center = terrain.length // 2
    rows = [y_center - stone - gap // 2, y_center + gap // 2] if two_rows else [y_center - stone // 2]

    x = 0
    while x < terrain.width:
        if one_row or two_rows:
            for y in rows:
                _fill_rect(terrain, x, x + stone, y, y + stone, _choice(heights, rng))
        else:
            y = 0
            while y < terrain.length:
                _fill_rect(terrain, x, x + stone, y, y + stone, _choice(heights, rng))
                y += stone + max(1, gap)
        x += stone + max(1, gap)

    _clear_spawn_platform(terrain, platform_size)
    _fill_rect(terrain, 0, platform, (terrain.length - platform) // 2, (terrain.length + platform) // 2, 0)


def _beam_path_terrain(
    terrain: _SubTerrain,
    rng: np.random.Generator,
    difficulty: float,
    depth: float = 0.6,
    platform_size: float = 2.0,
    cross: bool = False,
):
    depth_units = -abs(_height_to_units(terrain, depth))
    terrain.height_field_raw[:, :] = depth_units
    platform = max(1, _meter_to_index(terrain, platform_size))
    beam_len = int(np.clip(_meter_to_index(terrain, 0.35 - 0.1 * difficulty), 2, 8))
    beam_width_min = _meter_to_index(terrain, 0.75)
    beam_width_max = _meter_to_index(terrain, 1.5)
    gap = int(np.clip(_meter_to_index(terrain, 0.1 + 0.4 * difficulty), 2, 12))
    max_h = int(np.clip(_height_to_units(terrain, 0.05 + 0.18 * difficulty), 1, 35))
    heights = np.arange(0, max_h + 1, step=max(1, max_h // 5), dtype=np.int16)

    x = platform
    while x < terrain.width:
        width = int(rng.integers(max(3, beam_width_min), max(4, beam_width_max)))
        y0 = terrain.length // 2 - width // 2
        _fill_rect(terrain, x, x + beam_len, y0, y0 + width, _choice(heights, rng))
        x += beam_len + gap

    if cross:
        side_width = max(1, _meter_to_index(terrain, 0.12 + 0.1 * (1.0 - difficulty)))
        y0 = terrain.length // 2 - _meter_to_index(terrain, 0.35)
        y1 = terrain.length // 2 + _meter_to_index(terrain, 0.35)
        _fill_rect(terrain, platform, terrain.width, y0 - side_width, y0, 0)
        _fill_rect(terrain, platform, terrain.width, y1, y1 + side_width, 0)

    _fill_rect(terrain, 0, platform, (terrain.length - platform) // 2, (terrain.length + platform) // 2, 0)


def _air_beam_meshes(
    terrain: _SubTerrain,
    rng: np.random.Generator,
    difficulty: float,
    platform_size: float = 2.0,
) -> list[trimesh.Trimesh]:
    meshes: list[trimesh.Trimesh] = []
    sx = (terrain.width - 1) * terrain.horizontal_scale
    sy = (terrain.length - 1) * terrain.horizontal_scale
    beam_x = max(0.15, 0.35 - 0.1 * difficulty)
    beam_y_min = 0.35
    beam_y_max = 0.75
    gap = 0.1 + 0.4 * difficulty
    x = platform_size + 0.5 * beam_x
    while x < sx - 0.5 * beam_x:
        beam_y = float(rng.uniform(beam_y_min, beam_y_max))
        z = float(rng.uniform(0.05, 0.05 + 0.18 * difficulty))
        meshes.append(_make_box_xy(beam_x, beam_y, z + 0.05, 0.1, x, 0.5 * sy))
        x += beam_x + gap
    return meshes


def _air_stone_meshes(
    terrain: _SubTerrain,
    rng: np.random.Generator,
    difficulty: float,
    platform_size: float = 2.0,
) -> list[trimesh.Trimesh]:
    meshes: list[trimesh.Trimesh] = []
    sx = (terrain.width - 1) * terrain.horizontal_scale
    sy = (terrain.length - 1) * terrain.horizontal_scale
    stone_x = 0.45
    stone_y = float(rng.uniform(0.9, 1.25))
    z = max(0.18, 0.55 - 0.37 * difficulty)
    x = platform_size + 0.5 * stone_x
    while x < sx - 0.5 * stone_x:
        meshes.append(_make_box_xy(stone_x, stone_y, z + 0.06, 0.12, x, 0.5 * sy))
        x += stone_x + max(0.15, 0.25 + 0.35 * difficulty)
    return meshes


def _hurdle_terrain(terrain: _SubTerrain, rng: np.random.Generator, difficulty: float, depth: float = 0.6):
    terrain.height_field_raw[:, :] = -abs(_height_to_units(terrain, depth))
    corridor_w = _meter_to_index(terrain, 0.9)
    y0 = terrain.length // 2 - corridor_w // 2
    _fill_rect(terrain, 0, terrain.width, y0, y0 + corridor_w, 0)
    hurdle_h = _height_to_units(terrain, rng.uniform(0.1 + 0.35 * difficulty, 0.2 + 0.45 * difficulty))
    hurdle_x = max(3, _meter_to_index(terrain, 0.45))
    gap = max(6, _meter_to_index(terrain, 2.4 - 0.6 * difficulty))
    x = _meter_to_index(terrain, 2.2)
    while x < terrain.width - _meter_to_index(terrain, 1.0):
        _fill_rect(terrain, x, x + hurdle_x, y0, y0 + corridor_w, hurdle_h)
        x += hurdle_x + gap
    _clear_spawn_platform(terrain, 2.0)


def _half_sloped_terrain(terrain: _SubTerrain, difficulty: float, depth: float = 0.6, platform_size: float = 2.0):
    terrain.height_field_raw[:, :] = -abs(_height_to_units(terrain, depth))
    platform = max(1, _meter_to_index(terrain, platform_size))
    start = _meter_to_index(terrain, 0.25)
    end = max(start + 1, (terrain.width - platform) // 2)
    slope_units = max(1, int(2 * (difficulty * 2.5 + 1)))
    xs = np.arange(start, end)
    heights = (slope_units * (xs - start)).astype(np.int16)
    terrain.height_field_raw[start:end, :] = heights[:, None]
    top_h = int(heights[-1]) if heights.size else 0
    _fill_rect(terrain, end, end + platform, 0, terrain.length, top_h)
    for i, x in enumerate(range(end + platform, terrain.width)):
        terrain.height_field_raw[x, :] = max(0, top_h - i * slope_units)
    _clear_spawn_platform(terrain, platform_size)


def _corridor_terrain(terrain: _SubTerrain, difficulty: float, depth: float = 0.6, platform_size: float = 2.0):
    terrain.height_field_raw[:, :] = -abs(_height_to_units(terrain, depth))
    width = max(0.35, 1.0 - 0.55 * difficulty)
    corridor_w = _meter_to_index(terrain, width)
    y0 = terrain.length // 2 - corridor_w // 2
    _fill_rect(terrain, 0, terrain.width, y0, y0 + corridor_w, 0)
    _fill_rect(terrain, 0, _meter_to_index(terrain, platform_size), y0 - corridor_w, y0 + 2 * corridor_w, 0)


def _make_box_xy(size_x: float, size_y: float, top_z: float, height: float, center_x: float, center_y: float):
    z_center = top_z - 0.5 * height
    transform = trimesh.transformations.translation_matrix((center_x, center_y, z_center))
    return trimesh.creation.box((size_x, size_y, height), transform)


def _heightfield_to_trimesh(
    height_field_raw: np.ndarray,
    horizontal_scale: float,
    vertical_scale: float,
    slope_threshold: float | None,
) -> trimesh.Trimesh:
    hf = height_field_raw
    num_rows, num_cols = hf.shape
    y = np.linspace(0.0, (num_cols - 1) * horizontal_scale, num_cols)
    x = np.linspace(0.0, (num_rows - 1) * horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:
        threshold = slope_threshold * horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[: num_rows - 1, :] += hf[1:num_rows, :] - hf[: num_rows - 1, :] > threshold
        move_x[1:num_rows, :] -= hf[: num_rows - 1, :] - hf[1:num_rows, :] > threshold
        move_y[:, : num_cols - 1] += hf[:, 1:num_cols] - hf[:, : num_cols - 1] > threshold
        move_y[:, 1:num_cols] -= hf[:, : num_cols - 1] - hf[:, 1:num_cols] > threshold
        move_corners[: num_rows - 1, : num_cols - 1] += (
            hf[1:num_rows, 1:num_cols] - hf[: num_rows - 1, : num_cols - 1] > threshold
        )
        move_corners[1:num_rows, 1:num_cols] -= (
            hf[: num_rows - 1, : num_cols - 1] - hf[1:num_rows, 1:num_cols] > threshold
        )
        xx += (move_x + move_corners * (move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners * (move_y == 0)) * horizontal_scale

    vertices = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.reshape(-1)
    vertices[:, 1] = yy.reshape(-1)
    vertices[:, 2] = hf.reshape(-1) * vertical_scale

    triangles = np.empty((2 * (num_rows - 1) * (num_cols - 1), 3), dtype=np.int64)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols - 1) + i * num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2 * i * (num_cols - 1)
        stop = start + 2 * (num_cols - 1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start + 1 : stop : 2, 0] = ind0
        triangles[start + 1 : stop : 2, 1] = ind2
        triangles[start + 1 : stop : 2, 2] = ind3

    return trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)


def _spawn_origin(terrain: _SubTerrain, clearance: float = 0.08) -> np.ndarray:
    x = 0.5 * (terrain.width - 1) * terrain.horizontal_scale
    y = 0.5 * (terrain.length - 1) * terrain.horizontal_scale
    ix = int(np.clip(round(x / terrain.horizontal_scale), 0, terrain.width - 1))
    iy = int(np.clip(round(y / terrain.horizontal_scale), 0, terrain.length - 1))
    z = float(terrain.height_field_raw[ix, iy] * terrain.vertical_scale + clearance)
    return np.array([x, y, z], dtype=np.float32)


def mgdp_terrain(difficulty: float, cfg: "MGDPTerrainCfg") -> tuple[list[trimesh.Trimesh], np.ndarray]:
    terrain_type = cfg.terrain_type or _terrain_type_from_seed(getattr(cfg, "seed", None), cfg.mode)
    rng = _rng_from_seed(getattr(cfg, "seed", None), difficulty, terrain_type)
    terrain = _SubTerrain(cfg.size, cfg.horizontal_scale, cfg.vertical_scale)

    slope = 0.4 * difficulty
    step_height = 0.05 + 0.18 * difficulty
    obstacle_height = 0.05 + 0.2 * difficulty

    if terrain_type == "plane":
        pass
    elif terrain_type == "slope_down":
        _pyramid_sloped_terrain(terrain, -slope, platform_size=3.0)
    elif terrain_type == "pyramid":
        _pyramid_sloped_terrain(terrain, slope, platform_size=3.0)
        _random_uniform_terrain(terrain, rng, -0.05, 0.05, downsampled_scale=0.2)
    elif terrain_type == "stairs_down":
        _pyramid_stairs_terrain(terrain, 0.31, -step_height, platform_size=3.0)
    elif terrain_type in ("stairs_up", "new_stairs_up"):
        _pyramid_stairs_terrain(terrain, 0.31 if terrain_type == "stairs_up" else 0.5, step_height, platform_size=3.0)
    elif terrain_type == "discrete_obstacles":
        _discrete_obstacles_terrain(terrain, rng, obstacle_height)
    elif terrain_type == "pit":
        _pit_terrain(terrain, depth=max(0.1, difficulty), platform_size=4.0)
    elif terrain_type == "gap":
        gap = 0.5 * difficulty if difficulty < 0.1 else 0.1 + difficulty / terrain.horizontal_scale
        _parkour_gap_terrain(terrain, gap * terrain.horizontal_scale, depth=0.5, platform_size=2.0)
    elif terrain_type == "single_gap":
        _parkour_gap_terrain(terrain, difficulty, depth=0.6, platform_size=2.0)
    elif terrain_type == "step_stone":
        _stones_everywhere(terrain, rng, difficulty, depth=0.6, platform_size=2.0)
    elif terrain_type == "stones_2rows":
        _stones_everywhere(terrain, rng, difficulty, two_rows=True, depth=0.6, platform_size=2.0)
    elif terrain_type == "stones_1row":
        _stones_everywhere(terrain, rng, difficulty, one_row=True, depth=0.6, platform_size=2.0)
    elif terrain_type == "single_bridge":
        _single_bridge_terrain(terrain, difficulty, depth=0.6, platform_size=2.0)
    elif terrain_type == "hurdle":
        _hurdle_terrain(terrain, rng, difficulty, depth=0.6)
    elif terrain_type == "ramp":
        _half_sloped_terrain(terrain, difficulty, depth=0.6, platform_size=2.0)
    elif terrain_type == "corridor":
        _corridor_terrain(terrain, difficulty, depth=0.6, platform_size=2.0)
    elif terrain_type in ("step_beams", "rotation_beams", "narrow_beams", "cross_beams"):
        _beam_path_terrain(terrain, rng, difficulty, depth=0.6, platform_size=2.0, cross=terrain_type == "cross_beams")
    elif terrain_type == "air_beams":
        terrain.height_field_raw[:, :] = -abs(_height_to_units(terrain, 0.6))
        _clear_spawn_platform(terrain, 2.0)
    elif terrain_type == "air_stone":
        terrain.height_field_raw[:, :] = -abs(_height_to_units(terrain, 0.6))
        _clear_spawn_platform(terrain, 2.0)
    else:
        raise ValueError(f"Unknown MGDP terrain type: {terrain_type}")

    if cfg.add_roughness and terrain_type not in ("air_stone",):
        max_height = (cfg.roughness_height[1] - cfg.roughness_height[0]) * difficulty + cfg.roughness_height[0]
        height = float(rng.uniform(cfg.roughness_height[0], max_height))
        _random_uniform_terrain(terrain, rng, -height, height, step=0.005, downsampled_scale=cfg.downsampled_scale)
        _clear_spawn_platform(terrain, 2.0)

    meshes = [_heightfield_to_trimesh(terrain.height_field_raw, terrain.horizontal_scale, terrain.vertical_scale, cfg.slope_threshold)]
    if terrain_type == "air_beams" and cfg.add_air_beams:
        meshes.extend(_air_beam_meshes(terrain, rng, difficulty, platform_size=2.0))
    if terrain_type == "air_stone" and cfg.add_air_stones:
        meshes.extend(_air_stone_meshes(terrain, rng, difficulty, platform_size=2.0))

    return meshes, _spawn_origin(terrain)


def _terrain_type_from_seed(seed: int | None, mode: str) -> str:
    terrain_types = list(MGDP_GAP_PARKOUR_WEIGHTS if mode == "gap_parkour" else MGDP_MIX_WEIGHTS)
    base = 0 if seed is None else int(seed)
    return terrain_types[base % len(terrain_types)]


@configclass
class MGDPTerrainCfg(SubTerrainBaseCfg):
    function = mgdp_terrain

    terrain_type: str | None = None
    mode: str = "gap_parkour"
    horizontal_scale: float = 0.05
    vertical_scale: float = 0.005
    slope_threshold: float | None = 0.75
    add_roughness: bool = True
    roughness_height: tuple[float, float] = (0.01, 0.04)
    downsampled_scale: float = 0.5
    add_air_beams: bool = True
    add_air_stones: bool = True


MGDP_GAP_PARKOUR_TERRAIN_GENERATOR_CFG = TerrainGeneratorCfg(
    size=(10.0, 4.0),
    border_width=20.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    curriculum=True,
    use_cache=False,
    sub_terrains={
        name: MGDPTerrainCfg(proportion=weight, terrain_type=name, mode="gap_parkour")
        for name, weight in MGDP_GAP_PARKOUR_WEIGHTS.items()
    },
)


MGDP_MIX_TERRAIN_GENERATOR_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=20,
    num_cols=10,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    curriculum=True,
    use_cache=False,
    sub_terrains={
        name: MGDPTerrainCfg(proportion=weight, terrain_type=name, mode="mix", add_air_beams=False, add_air_stones=False)
        for name, weight in MGDP_MIX_WEIGHTS.items()
    },
)


MGDP_TERRAIN_GENERATOR_CFG = MGDP_GAP_PARKOUR_TERRAIN_GENERATOR_CFG
