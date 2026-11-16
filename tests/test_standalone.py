"""The new runtime must not silently load its former parent Python package."""
from pathlib import Path
import subprocess,sys

ROOT=Path(__file__).resolve().parents[1]


def test_runtime_imports_without_gamepad_vla():
    script="""
import importlib.abc,sys
class NoOldPackage(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='gamepad_vla' or fullname.startswith('gamepad_vla.'):
   raise RuntimeError('old project imported: '+fullname)
sys.meta_path.insert(0,NoOldPackage())
from dexstation.runtime import controller as collection_controller
from dexstation.server import app as collection_server
from dexstation.backends import sapien_physics as collection_sapien
from dexstation.backends import mujoco_export as collection_mujoco
from dexstation.render import scene_snapshot
from dexstation.render import operator_view
from dexstation.inputs import supervisor as input_chain
from dexstation.examples.ar5_g20 import mapping as g20_mapping
assert not any(name=='gamepad_vla' or name.startswith('gamepad_vla.') for name in sys.modules)
"""
    result=subprocess.run([sys.executable,'-c',script],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
