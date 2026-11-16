from helpers import bench_spec
from pathlib import Path

from dexstation.model.appearance import load_appearance


ROOT = Path(__file__).resolve().parents[1]


def test_workspace_override_resolves_scene_and_visual_parts(tmp_path, monkeypatch):
    monkeypatch.setenv("GAMEPAD_VLA_WORKSPACE", str(tmp_path))
    spec = bench_spec()
    assert spec.left_arm.urdf.startswith(str(tmp_path))
    assert all(f.mesh.startswith(str(tmp_path)) for f in spec.fixtures if f.mesh)
    appearance_path = tmp_path / "appearance.json"
    appearance_path.write_text('''{
      "name": "example",
      "color_space": "linear_rgb",
      "materials": [{
        "name": "example", "match": {"link_glob": ["example"], "mesh_glob": []},
        "base_color": [0.5, 0.5, 0.5, 1.0], "metallic": 0.0, "roughness": 0.5,
        "status": "design", "source": "Synthetic workspace resolution fixture"
      }],
      "visual_splits": [{
        "link": "example", "source_mesh": "workspace://assets/example/source.stl",
        "parts": [
          {"mesh": "workspace://assets/example/part_a.stl", "material": "example"},
          {"mesh": "workspace://assets/example/part_b.stl", "material": "example"}
        ]
      }]
    }''')
    appearance = load_appearance(appearance_path)
    for split in appearance["visual_splits"]:
        assert split["source_mesh"].startswith(str(tmp_path))
        assert all(p["mesh"].startswith(str(tmp_path)) for p in split["parts"])
