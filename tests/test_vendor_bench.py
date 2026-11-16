"""Maintained bench dimensions and material assignments."""
from helpers import bench_scene
from pathlib import Path
import numpy as np
from dexstation.model.appearance import load_appearance, match_material
ROOT = Path(__file__).parents[1]


def test_maintained_bench_dimensions_and_materials():
    scene = bench_scene()
    np.testing.assert_allclose(2 * scene.spec.table.half_extents[2], .015, atol=1e-6)
    column = next(f for f in scene.spec.fixtures if f.name == 'station_column')
    np.testing.assert_allclose(2 * column.half_extents[0], .060, atol=1e-6)
    appearance = load_appearance(ROOT / 'src/dexstation/examples/ar5_g20/appearance/station_v3_3_fixture.json')
    for fixture in scene.spec.fixtures:
        if fixture.name.startswith('station_column'):
            assert match_material(appearance, fixture.name, '') == match_material(appearance, 'station_column', '')
        elif fixture.name.startswith('station_table_frame_'):
            assert match_material(appearance, fixture.name, '') == match_material(appearance, 'station_table_frame_x', '')
