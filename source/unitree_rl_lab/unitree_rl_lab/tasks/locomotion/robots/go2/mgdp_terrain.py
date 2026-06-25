from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass


MGDP_GAP_PARKOUR_WEIGHTS = {
    "single_gap": 0.1,
    "stone_everywhere": 0.1,
    "stones_2rows": 0.1,
    "stones_2rows_staggered": 0.0,
    "stones_1row": 0.1,
    "single_bridge": 0.1,
    "stairs": 0.0,
    "beams": 0.1,
    "air_beams": 0.1,
    "air_stairs": 0.1,
}

MGDP_PIT_DEPTH_M = 0.6
MGDP_PLATFORM_SIZE_M = 2.0

_STONE_TERRAIN_KWARGS = {
    "stone_everywhere": {},
    "stones_2rows": dict(
        two_rows=True, lateral_stone_scale=0.5, lateral_gap_scale=1.0, forward_gap_scale=0.5,
        forward_stone_scale=0.5, height_scale=0.5,
    ),
    "stones_2rows_staggered": dict(
        two_rows=True, staggered_rows=True, lateral_stone_scale=0.5, lateral_gap_scale=1.0,
        forward_gap_scale=0.5, forward_stone_scale=0.5, height_scale=0.25,
    ),
    "stones_1row": dict(one_row=True, height_scale=1.0),
}

_PIT_ONLY_TERRAINS = {"beams", "air_beams", "air_stairs"}


@configclass
class MGDPGapParkourCfg:
    single_gap_min_cells: int = 1
    single_gap_max_cells: int = 12

    stone_everywhere_gap_min_m: float = 0.05
    stone_everywhere_gap_max_m: float = 0.45
    stone_everywhere_gap_min_cells: int = 1
    stone_everywhere_gap_max_cells: int = 9
    stone_lateral_gap_min_m: float = 0.0
    stone_lateral_gap_max_m: float = 0.2

    air_beam_gap_min_m: float = 0.05
    air_beam_gap_max_m: float = 0.5
    air_beam_height_delta_max_m: float = 0.4

    beam_gap_min_m: float = 0.05
    beam_gap_max_m: float = 0.40
    beam_width_min_m: float = 0.20
    beam_width_max_m: float = 0.30
    beam_height_min_m: float = 0.05
    beam_height_max_m: float = 0.50
    beam_length_min_m: float = 0.35
    beam_length_max_m: float = 0.75

    stair_step_height_min_m: float = 0.05
    stair_step_height_max_m: float = 0.20
    stair_step_depth_min_m: float = 0.25
    stair_step_depth_max_m: float = 0.40
    stair_width_min_m: float = 0.50
    stair_width_max_m: float = 2.00
    stair_top_platform_length_min_m: float = 1.00
    stair_top_platform_length_max_m: float = 2.00

    air_stairs_step_min_m: float = 0.05
    air_stairs_step_max_m: float = 0.6
    air_stairs_height_delta_max_m: float = 0.4


MGDP_GAP_PARKOUR_CFG = MGDPGapParkourCfg()

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


def _lerp_range(min_value: float, max_value: float, difficulty: float) -> float:
    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    return float(min_value + (max_value - min_value) * difficulty)


def _fill_rect(terrain: _SubTerrain, x0: int, x1: int, y0: int, y1: int, height_units: int) -> None:
    x0 = int(np.clip(x0, 0, terrain.width))
    x1 = int(np.clip(x1, 0, terrain.width))
    y0 = int(np.clip(y0, 0, terrain.length))
    y1 = int(np.clip(y1, 0, terrain.length))
    if x1 > x0 and y1 > y0:
        terrain.height_field_raw[x0:x1, y0:y1] = height_units


def _rects_overlap(rect_a: tuple[int, int, int, int], rect_b: tuple[int, int, int, int]) -> bool:
    ax0, ax1, ay0, ay1 = rect_a
    bx0, bx1, by0, by1 = rect_b
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


def _clear_start_platform(terrain: _SubTerrain, platform_size: float, height_units: int = 0) -> int:
    platform = max(1, _meter_to_index(terrain, platform_size))
    _fill_rect(terrain, 0, platform, (terrain.length - platform) // 2, (terrain.length + platform) // 2, height_units)
    return platform


def _fill_pit_with_start_platform(
    terrain: _SubTerrain,
    depth: float = MGDP_PIT_DEPTH_M,
    platform_size: float = MGDP_PLATFORM_SIZE_M,
) -> None:
    terrain.height_field_raw[:, :] = -abs(_height_to_units(terrain, depth))
    _clear_start_platform(terrain, platform_size)


def _parkour_gap_terrain(
    terrain: _SubTerrain,
    difficulty: float,
    cfg: MGDPGapParkourCfg,
    depth: float = MGDP_PIT_DEPTH_M,
    platform_size: float = MGDP_PLATFORM_SIZE_M,
):
    depth_units = -abs(_height_to_units(terrain, depth))
    gap = int(np.clip(difficulty / terrain.horizontal_scale, cfg.single_gap_min_cells, cfg.single_gap_max_cells))
    platform = max(1, _meter_to_index(terrain, platform_size))
    end_y = int(terrain.length - platform / 8)
    center_x = terrain.width // 2

    _fill_rect(terrain, platform, center_x, 0, end_y, depth_units)
    _fill_rect(terrain, platform + gap, center_x - gap, gap, end_y - gap, 0)

    start_x = center_x + platform // 2
    _fill_rect(terrain, start_x, terrain.width, 0, end_y, depth_units)
    _fill_rect(terrain, start_x + gap, terrain.width - gap, gap, end_y - gap, 0)


def _single_bridge_terrain(
    terrain: _SubTerrain,
    difficulty: float,
    depth: float = MGDP_PIT_DEPTH_M,
    platform_size: float = MGDP_PLATFORM_SIZE_M,
):
    depth_units = -abs(_height_to_units(terrain, depth))
    terrain.height_field_raw[:, :] = depth_units
    bridge_width = max(3, _meter_to_index(terrain, max(0.2, 0.8 - 0.6 * difficulty)))
    y0 = terrain.length // 2 - bridge_width // 2
    _fill_rect(terrain, 0, terrain.width, y0, y0 + bridge_width, 0)
    _clear_start_platform(terrain, platform_size)


def _stairs_terrain(
    terrain: _SubTerrain,
    difficulty: float,
    cfg: MGDPGapParkourCfg,
    depth: float = MGDP_PIT_DEPTH_M,
    platform_size: float = MGDP_PLATFORM_SIZE_M,
) -> None:
    depth_units = -abs(_height_to_units(terrain, depth))
    terrain.height_field_raw[:, :] = depth_units

    step_height_units = max(
        1,
        _height_to_units(
            terrain,
            _lerp_range(cfg.stair_step_height_min_m, cfg.stair_step_height_max_m, difficulty),
        ),
    )
    step_depth = max(
        1,
        _meter_to_index(terrain, _lerp_range(cfg.stair_step_depth_max_m, cfg.stair_step_depth_min_m, difficulty)),
    )
    stair_width = max(
        1,
        _meter_to_index(terrain, _lerp_range(cfg.stair_width_max_m, cfg.stair_width_min_m, difficulty)),
    )
    top_platform = max(
        1,
        _meter_to_index(
            terrain,
            _lerp_range(cfg.stair_top_platform_length_max_m, cfg.stair_top_platform_length_min_m, difficulty),
        ),
    )

    platform = max(1, _meter_to_index(terrain, platform_size))
    y0 = terrain.length // 2 - stair_width // 2
    y1 = y0 + stair_width
    end_platform_start = max(platform, terrain.width - platform)
    top_start = max(platform, terrain.width // 2 - top_platform // 2)
    top_end = min(end_platform_start, top_start + top_platform)

    _clear_start_platform(terrain, platform_size)
    _fill_rect(
        terrain,
        end_platform_start,
        terrain.width,
        (terrain.length - platform) // 2,
        (terrain.length + platform) // 2,
        0,
    )

    x = platform
    stair_index = 1
    while x < top_start:
        x_next = min(top_start, x + step_depth)
        _fill_rect(terrain, x, x_next, y0, y1, stair_index * step_height_units)
        x = x_next
        stair_index += 1

    top_height_units = max(0, (stair_index - 1) * step_height_units)
    _fill_rect(terrain, top_start, top_end, y0, y1, top_height_units)

    x = top_end
    stair_index = 1
    while x < end_platform_start:
        x_next = min(end_platform_start, x + step_depth)
        height_units = max(0, top_height_units - stair_index * step_height_units)
        _fill_rect(terrain, x, x_next, y0, y1, height_units)
        x = x_next
        stair_index += 1


def _stones_everywhere(
    terrain: _SubTerrain,
    rng: np.random.Generator,
    difficulty: float,
    cfg: MGDPGapParkourCfg,
    two_rows: bool = False,
    staggered_rows: bool = False,
    one_row: bool = False,
    depth: float = MGDP_PIT_DEPTH_M,
    platform_size: float = MGDP_PLATFORM_SIZE_M,
    lateral_stone_scale: float = 1.0,
    lateral_gap_scale: float = 1.0,
    forward_gap_scale: float = 1.0,
    forward_stone_scale: float = 1.0,
    height_scale: float = 1.0,
):
    depth_units = -abs(_height_to_units(terrain, depth))
    terrain.height_field_raw[:, :] = depth_units
    stone_size_m = max(0.22, 0.8 - 0.5 * difficulty)
    stone = int(np.clip(_meter_to_index(terrain, stone_size_m), 4, 18))
    stone_forward = max(1, int(round(stone * forward_stone_scale)))
    stone_lateral = max(1, int(round(stone * lateral_stone_scale)))
    forward_gap = int(
        np.clip(
            _meter_to_index(
                terrain,
                (
                    cfg.stone_everywhere_gap_min_m
                    + (cfg.stone_everywhere_gap_max_m - cfg.stone_everywhere_gap_min_m) * difficulty
                )
                * forward_gap_scale,
            ),
            cfg.stone_everywhere_gap_min_cells,
            cfg.stone_everywhere_gap_max_cells,
        )
    )
    lateral_gap_m = float(
        np.clip(
            cfg.stone_lateral_gap_min_m + (cfg.stone_lateral_gap_max_m - cfg.stone_lateral_gap_min_m) * difficulty,
            cfg.stone_lateral_gap_min_m,
            cfg.stone_lateral_gap_max_m,
        )
    )
    lateral_gap = int(np.floor(lateral_gap_m * lateral_gap_scale / terrain.horizontal_scale + 1e-9))
    max_h = int(
        np.clip(
            _height_to_units(terrain, (0.05 + 0.35 * difficulty) * height_scale),
            1,
            _height_to_units(terrain, 0.4 * height_scale),
        )
    )
    heights = np.arange(0, max_h + 1, step=max(1, max_h // 6), dtype=np.int16)
    platform = max(1, _meter_to_index(terrain, platform_size))
    spawn_rect = (0, platform, (terrain.length - platform) // 2, (terrain.length + platform) // 2)
    stone_rects: list[tuple[int, int, int, int]] = []
    y_center = terrain.length // 2
    if two_rows:
        first_row_end = y_center - lateral_gap // 2
        rows = [first_row_end - stone_lateral, first_row_end + lateral_gap]
    else:
        rows = [y_center - stone_lateral // 2]

    x = 0
    row_offsets = [0, (stone_forward + forward_gap) // 2] if two_rows and staggered_rows else [0] * len(rows)
    while x < terrain.width:
        if one_row or two_rows:
            for y, x_offset in zip(rows, row_offsets):
                x0 = x + x_offset
                rect = (x0, x0 + stone_forward, y, y + stone_lateral)
                stone_rects.append(rect)
                _fill_rect(terrain, *rect, _choice(heights, rng))
        else:
            y = 0
            while y < terrain.length:
                rect = (x, x + stone_forward, y, y + stone_lateral)
                stone_rects.append(rect)
                _fill_rect(terrain, *rect, _choice(heights, rng))
                y += stone_lateral + lateral_gap
        x += stone_forward + forward_gap

    for rect in stone_rects:
        if _rects_overlap(rect, spawn_rect):
            _fill_rect(terrain, *rect, 0)

    _clear_start_platform(terrain, platform_size)


def _air_beam_meshes(
    terrain: _SubTerrain,
    rng: np.random.Generator,
    difficulty: float,
    cfg: MGDPGapParkourCfg,
    platform_size: float = MGDP_PLATFORM_SIZE_M,
    first_top_z: float = 0.0,
) -> list[trimesh.Trimesh]:
    beam_specs: list[tuple[float, float, float, float, float, float]] = []
    sx = (terrain.width - 1) * terrain.horizontal_scale
    sy = (terrain.length - 1) * terrain.horizontal_scale
    beam_x = max(0.15, 0.35 - 0.1 * difficulty)
    beam_y_min = 0.35
    beam_y_max = 0.75
    gap = cfg.air_beam_gap_min_m + (cfg.air_beam_gap_max_m - cfg.air_beam_gap_min_m) * difficulty
    x = platform_size + 0.5 * beam_x
    while x < sx - 0.5 * beam_x:
        beam_y = float(rng.uniform(beam_y_min, beam_y_max))
        z = float(rng.uniform(0.05, 0.05 + cfg.air_beam_height_delta_max_m * difficulty))
        beam_specs.append((beam_x, beam_y, z + 0.05, 0.1, x, 0.5 * sy))
        x += beam_x + gap

    if not beam_specs:
        return []
    z_offset = beam_specs[0][2] - first_top_z
    return [
        _make_box_xy(size_x, size_y, top_z - z_offset, height, center_x, center_y)
        for size_x, size_y, top_z, height, center_x, center_y in beam_specs
    ]


def _beam_meshes(
    terrain: _SubTerrain,
    rng: np.random.Generator,
    difficulty: float,
    cfg: MGDPGapParkourCfg,
    depth: float = MGDP_PIT_DEPTH_M,
    platform_size: float = MGDP_PLATFORM_SIZE_M,
) -> list[trimesh.Trimesh]:
    meshes: list[trimesh.Trimesh] = []
    sx = (terrain.width - 1) * terrain.horizontal_scale
    sy = (terrain.length - 1) * terrain.horizontal_scale
    beam_x = _lerp_range(cfg.beam_width_max_m, cfg.beam_width_min_m, difficulty)
    gap = _lerp_range(cfg.beam_gap_min_m, cfg.beam_gap_max_m, difficulty)
    max_height = _lerp_range(cfg.beam_height_min_m, cfg.beam_height_max_m, difficulty)
    base_z = -abs(depth)
    platform = max(1, _meter_to_index(terrain, platform_size))
    spawn_rect = (0, platform, (terrain.length - platform) // 2, (terrain.length + platform) // 2)
    x = platform_size + 0.5 * beam_x

    while x < sx - 0.5 * beam_x:
        beam_y = float(rng.uniform(cfg.beam_length_min_m, cfg.beam_length_max_m))
        center_y = 0.5 * sy
        top_z = float(rng.uniform(0.0, max_height))
        rect = (
            int(np.floor((x - 0.5 * beam_x) / terrain.horizontal_scale + 1e-9)),
            int(np.ceil((x + 0.5 * beam_x) / terrain.horizontal_scale - 1e-9)),
            int(np.floor((center_y - 0.5 * beam_y) / terrain.horizontal_scale + 1e-9)),
            int(np.ceil((center_y + 0.5 * beam_y) / terrain.horizontal_scale - 1e-9)),
        )
        if _rects_overlap(rect, spawn_rect):
            top_z = 0.0
        meshes.append(
            _make_box_xy(
                size_x=beam_x,
                size_y=beam_y,
                top_z=top_z,
                height=max(terrain.vertical_scale, top_z - base_z),
                center_x=x,
                center_y=center_y,
            )
        )
        x += beam_x + gap

    return meshes


def _air_stairs_meshes(
    terrain: _SubTerrain,
    rng: np.random.Generator,
    difficulty: float,
    cfg: MGDPGapParkourCfg,
    platform_size: float = MGDP_PLATFORM_SIZE_M,
    first_top_z: float = 0.0,
) -> list[trimesh.Trimesh]:
    stair_specs: list[tuple[float, float, float, float, float, float]] = []
    sx = (terrain.width - 1) * terrain.horizontal_scale
    sy = (terrain.length - 1) * terrain.horizontal_scale
    stair_x = 0.45
    stair_y = float(rng.uniform(0.9, 1.25))
    x = platform_size + 0.5 * stair_x
    while x < sx - 0.5 * stair_x:
        top_z = float(rng.uniform(0.0, cfg.air_stairs_height_delta_max_m * difficulty))
        stair_specs.append((stair_x, stair_y, top_z, 0.12, x, 0.5 * sy))
        x += stair_x + (cfg.air_stairs_step_min_m + (cfg.air_stairs_step_max_m - cfg.air_stairs_step_min_m) * difficulty)

    if not stair_specs:
        return []
    z_offset = stair_specs[0][2] - first_top_z
    return [
        _make_box_xy(size_x, size_y, top_z - z_offset, height, center_x, center_y)
        for size_x, size_y, top_z, height, center_x, center_y in stair_specs
    ]


def _make_box_xy(size_x: float, size_y: float, top_z: float, height: float, center_x: float, center_y: float):
    z_center = top_z - 0.5 * height
    transform = trimesh.transformations.translation_matrix((center_x, center_y, z_center))
    return trimesh.creation.box((size_x, size_y, height), transform)


def _heightfield_to_terraced_trimesh(
    height_field_raw: np.ndarray,
    horizontal_scale: float,
    vertical_scale: float,
    min_thickness: float = 0.02,
    outer_wall_edges: tuple[bool, bool, bool, bool] = (False, False, False, False),
    outer_wall_top_z: float = 0.0,
) -> trimesh.Trimesh:
    hf = height_field_raw
    num_rows, num_cols = hf.shape
    cell_heights = hf[:-1, :-1].astype(np.float32) * vertical_scale
    bottom_z = float(np.min(cell_heights)) - max(float(min_thickness), abs(float(vertical_scale)))
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    vertex_indices: dict[tuple[float, float, float], int] = {}

    def add_vertex(vertex) -> int:
        key = tuple(round(float(component), 10) for component in vertex)
        index = vertex_indices.get(key)
        if index is None:
            index = len(vertices)
            vertex_indices[key] = index
            vertices.append(key)
        return index

    def add_quad(v0, v1, v2, v3) -> None:
        indices = (add_vertex(v0), add_vertex(v1), add_vertex(v2), add_vertex(v3))
        if len(set(indices)) == 4:
            faces.extend(((indices[0], indices[1], indices[2]), (indices[0], indices[2], indices[3])))

    def add_polygon(points) -> None:
        indices = tuple(add_vertex(point) for point in points)
        for index in range(1, len(indices) - 1):
            triangle = (indices[0], indices[index], indices[index + 1])
            if len(set(triangle)) == 3:
                faces.append(triangle)

    corner_levels = [[set() for _ in range(num_cols)] for _ in range(num_rows)]
    for i in range(num_rows - 1):
        for j in range(num_cols - 1):
            z = float(cell_heights[i, j])
            corner_levels[i][j].add(z)
            corner_levels[i + 1][j].add(z)
            corner_levels[i][j + 1].add(z)
            corner_levels[i + 1][j + 1].add(z)

    def levels_between(corner_i: int, corner_j: int, z_low: float, z_high: float) -> list[float]:
        return sorted(level for level in corner_levels[corner_i][corner_j] if z_low < level < z_high)

    def add_vertical_strip(
        start_xy: tuple[float, float],
        end_xy: tuple[float, float],
        start_corner: tuple[int, int],
        end_corner: tuple[int, int],
        z_low: float,
        z_high: float,
    ) -> None:
        start_levels = levels_between(*start_corner, z_low, z_high)
        end_levels = levels_between(*end_corner, z_low, z_high)
        points = [(start_xy[0], start_xy[1], z_low), (end_xy[0], end_xy[1], z_low)]
        points.extend((end_xy[0], end_xy[1], level) for level in end_levels)
        points.extend(((end_xy[0], end_xy[1], z_high), (start_xy[0], start_xy[1], z_high)))
        points.extend((start_xy[0], start_xy[1], level) for level in reversed(start_levels))
        add_polygon(points)

    left, right, bottom, top = outer_wall_edges
    wall_z = float(outer_wall_top_z)

    for i in range(num_rows - 1):
        x0 = i * horizontal_scale
        x1 = (i + 1) * horizontal_scale
        for j in range(num_cols - 1):
            y0 = j * horizontal_scale
            y1 = (j + 1) * horizontal_scale
            z = float(cell_heights[i, j])
            add_quad((x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z))
            add_quad((x0, y1, bottom_z), (x1, y1, bottom_z), (x1, y0, bottom_z), (x0, y0, bottom_z))

    for i in range(1, num_rows - 1):
        x = i * horizontal_scale
        for j in range(num_cols - 1):
            z_left = float(cell_heights[i - 1, j])
            z_right = float(cell_heights[i, j])
            if z_left == z_right:
                continue
            y0 = j * horizontal_scale
            y1 = (j + 1) * horizontal_scale
            z_low, z_high = sorted((z_left, z_right))
            if z_left > z_right:
                add_vertical_strip((x, y0), (x, y1), (i, j), (i, j + 1), z_low, z_high)
            else:
                add_vertical_strip((x, y1), (x, y0), (i, j + 1), (i, j), z_low, z_high)

    for j in range(1, num_cols - 1):
        y = j * horizontal_scale
        for i in range(num_rows - 1):
            z_front = float(cell_heights[i, j - 1])
            z_back = float(cell_heights[i, j])
            if z_front == z_back:
                continue
            x0 = i * horizontal_scale
            x1 = (i + 1) * horizontal_scale
            z_low, z_high = sorted((z_front, z_back))
            if z_front > z_back:
                add_vertical_strip((x1, y), (x0, y), (i + 1, j), (i, j), z_low, z_high)
            else:
                add_vertical_strip((x0, y), (x1, y), (i, j), (i + 1, j), z_low, z_high)

    for i in range(num_rows - 1):
        x0 = i * horizontal_scale
        x1 = (i + 1) * horizontal_scale
        z = float(cell_heights[i, 0])
        add_vertical_strip((x0, 0.0), (x1, 0.0), (i, 0), (i + 1, 0), bottom_z, max(z, wall_z) if bottom else z)

        y = (num_cols - 1) * horizontal_scale
        z = float(cell_heights[i, num_cols - 2])
        add_vertical_strip(
            (x1, y),
            (x0, y),
            (i + 1, num_cols - 1),
            (i, num_cols - 1),
            bottom_z,
            max(z, wall_z) if top else z,
        )

    for j in range(num_cols - 1):
        y0 = j * horizontal_scale
        y1 = (j + 1) * horizontal_scale
        z = float(cell_heights[0, j])
        add_vertical_strip((0.0, y1), (0.0, y0), (0, j + 1), (0, j), bottom_z, max(z, wall_z) if left else z)

        x = (num_rows - 1) * horizontal_scale
        z = float(cell_heights[num_rows - 2, j])
        add_vertical_strip(
            (x, y0),
            (x, y1),
            (num_rows - 1, j),
            (num_rows - 1, j + 1),
            bottom_z,
            max(z, wall_z) if right else z,
        )

    return trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )


def _spawn_origin(terrain: _SubTerrain, platform_size: float = MGDP_PLATFORM_SIZE_M, clearance: float = 0.08) -> np.ndarray:
    x = 0.5 * platform_size
    y = 0.5 * (terrain.length - 1) * terrain.horizontal_scale
    ix = int(np.clip(round(x / terrain.horizontal_scale), 0, terrain.width - 1))
    iy = int(np.clip(round(y / terrain.horizontal_scale), 0, terrain.length - 1))
    z = float(terrain.height_field_raw[ix, iy] * terrain.vertical_scale + clearance)
    return np.array([x, y, z], dtype=np.float32)


def mgdp_terrain(difficulty: float, cfg: "MGDPTerrainCfg") -> tuple[list[trimesh.Trimesh], np.ndarray]:
    terrain_type = cfg.terrain_type or _terrain_type_from_seed(getattr(cfg, "seed", None))
    rng = _rng_from_seed(getattr(cfg, "seed", None), difficulty, terrain_type)
    terrain = _SubTerrain(cfg.size, cfg.horizontal_scale, cfg.vertical_scale)
    gap_cfg = getattr(cfg, "gap_cfg", MGDP_GAP_PARKOUR_CFG)

    if terrain_type == "single_gap":
        _parkour_gap_terrain(terrain, difficulty, gap_cfg)
    elif terrain_type in _STONE_TERRAIN_KWARGS:
        _stones_everywhere(
            terrain,
            rng,
            difficulty,
            gap_cfg,
            **_STONE_TERRAIN_KWARGS[terrain_type],
        )
    elif terrain_type == "single_bridge":
        _single_bridge_terrain(terrain, difficulty)
    elif terrain_type == "stairs":
        _stairs_terrain(terrain, difficulty, gap_cfg)
    elif terrain_type in _PIT_ONLY_TERRAINS:
        _fill_pit_with_start_platform(terrain)
    else:
        raise ValueError(f"Unknown MGDP terrain type: {terrain_type}")

    meshes = [
        _heightfield_to_terraced_trimesh(
            terrain.height_field_raw,
            terrain.horizontal_scale,
            terrain.vertical_scale,
            outer_wall_edges=getattr(cfg, "outer_wall_edges", (False, False, False, False)),
            outer_wall_top_z=getattr(cfg, "outer_wall_top_z", 0.0),
        )
    ]
    if terrain_type == "beams":
        meshes.extend(_beam_meshes(terrain, rng, difficulty, gap_cfg))
    elif terrain_type == "air_beams" and cfg.add_air_beams:
        meshes.extend(
            _air_beam_meshes(terrain, rng, difficulty, gap_cfg, first_top_z=getattr(cfg, "air_first_top_z", 0.0))
        )
    elif terrain_type == "air_stairs" and cfg.add_air_stairs:
        meshes.extend(
            _air_stairs_meshes(terrain, rng, difficulty, gap_cfg, first_top_z=getattr(cfg, "air_first_top_z", 0.0))
        )

    return meshes, _spawn_origin(terrain)


def _terrain_type_from_seed(seed: int | None) -> str:
    terrain_types = list(MGDP_GAP_PARKOUR_WEIGHTS)
    base = 0 if seed is None else int(seed)
    return terrain_types[base % len(terrain_types)]


@configclass
class MGDPTerrainCfg(SubTerrainBaseCfg):
    function = mgdp_terrain

    terrain_type: str | None = None
    seed: int | None = None
    gap_cfg: MGDPGapParkourCfg = MGDP_GAP_PARKOUR_CFG
    horizontal_scale: float = 0.05
    vertical_scale: float = 0.01
    outer_wall_edges: tuple[bool, bool, bool, bool] = (False, False, False, False)
    outer_wall_top_z: float = 0.0
    air_first_top_z: float = 0.0
    add_air_beams: bool = True
    add_air_stairs: bool = True


MGDP_GAP_PARKOUR_TERRAIN_GENERATOR_CFG = TerrainGeneratorCfg(
    size=(10.0, 5.0),
    border_width=20.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.05,
    vertical_scale=0.01,
    difficulty_range=(0.0, 1.0),
    curriculum=True,
    use_cache=False,
    sub_terrains={
        name: MGDPTerrainCfg(proportion=weight, terrain_type=name)
        for name, weight in MGDP_GAP_PARKOUR_WEIGHTS.items()
    },
)


MGDP_TERRAIN_GENERATOR_CFG = MGDP_GAP_PARKOUR_TERRAIN_GENERATOR_CFG
