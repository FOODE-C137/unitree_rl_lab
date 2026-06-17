"""Trimesh-only test terrain presets for Go2 raycast experiments."""

import isaaclab.terrains as terrain_gen


TEST_TERRAIN_GENERATOR_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=8,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    curriculum=True,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.10),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.03, 0.22),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.03, 0.22),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "random_grid": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15,
            grid_width=0.45,
            grid_height_range=(0.03, 0.20),
            platform_width=2.0,
            holes=False,
        ),
        "rails": terrain_gen.MeshRailsTerrainCfg(
            proportion=0.10,
            rail_thickness_range=(0.05, 0.15),
            rail_height_range=(0.05, 0.30),
            platform_width=2.0,
        ),
        "pit": terrain_gen.MeshPitTerrainCfg(
            proportion=0.10,
            pit_depth_range=(0.05, 0.60),
            platform_width=2.0,
            double_pit=False,
        ),
        "box": terrain_gen.MeshBoxTerrainCfg(
            proportion=0.10,
            box_height_range=(0.05, 0.30),
            platform_width=2.0,
            double_box=False,
        ),
        "gap": terrain_gen.MeshGapTerrainCfg(
            proportion=0.15,
            gap_width_range=(0.05, 0.80),
            platform_width=2.0,
        ),
    },
)


TEST_TERRAIN_CFG = TEST_TERRAIN_GENERATOR_CFG
