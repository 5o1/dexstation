"""The generic camera panel can represent a service that is unavailable."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.skipif(shutil.which('node') is None, reason='requires existing Node executable')
def test_unavailable_camera_service_disables_controls():
    module = Path(__file__).parents[1] / 'src/dexstation/server/web/camera-panel.mjs'
    script = '''import {installCameraPanel,cameraActions} from MODULE_URL;
import assert from 'node:assert/strict';
const elements = new Map(), created = [];
function element() {return {textContent:'',value:'',open:true,classList:{toggle(){}},append(){},replaceChildren(){}}}
const get = id => {if(!elements.has(id))elements.set(id,element());return elements.get(id)};
globalThis.setTimeout = () => 0;
const fetcher = async path => ({ok:true,json:async()=>path.endsWith('/status')
  ? {available:false,mode:'unavailable',error:'Camera service not available',cameras:[]}:[]});
installCameraPanel({getElementById:get,createElement:()=>{const value=element();created.push(value);return value}},fetcher);
await new Promise(setImmediate);
assert.equal(get('cameraMode').textContent,'Unavailable');
assert.equal(get('cameraMessage').textContent,'Camera service not available');
for(const name of Object.keys(cameraActions))assert.equal(get(name).disabled,true,name);
assert.equal(created.find(value=>value.id==='cameraPreviewTarget').disabled,true);
'''.replace('MODULE_URL', json.dumps(module.as_uri()))
    result = subprocess.run(['node', '--input-type=module', '-e', script], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
