import assert from 'node:assert/strict';
import fs from 'node:fs';
const source=fs.readFileSync(new URL('../src/dexstation/server/web/auxiliary-panel.mjs',import.meta.url),'utf8').replace(/^import .*;\n/,'');
const {paneLayout,savedViews,orderedPanes,paneVisible}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
const task=JSON.parse(fs.readFileSync(new URL('../src/dexstation/task_plugins/xbox_gamepad_v3_3/data/task.json',import.meta.url)));
const core=[{id:'table',title:'Table top',weight:60,camera:'table'},
 {id:'arm_takeover',title:'Arm takeover',weight:90,visible_when:{'source_capabilities.teleop_calibration':true}},
 {id:'status',title:'Status',weight:100}];
const all=[...task.panes,...core],names=all.map(p=>p.id);
assert.deepEqual(orderedPanes([{id:'z'},{id:'b',weight:1},{id:'a',weight:1},{id:'c'}]).map(p=>p.id),['a','b','c','z']);
assert.deepEqual(savedViews(['overview','face','old','right','face'],all),['face','right']);
assert.deepEqual(savedViews(null,all),names);
assert.ok(savedViews(['face','overview'],all,true).includes('gamepad_signals'),'New module panes default on for legacy selections');
assert.ok(!savedViews(['face'],all).includes('gamepad_signals'),'A current saved unchecked module stays unchecked');
let checked=0;
for(const source of ['udp','ik_autopilot','replay']){
 const state={source,source_capabilities:{teleop_calibration:source==='udp'}};
 const available=all.filter(p=>paneVisible(p,state));
 const optional=available.filter(p=>p.camera||p.module),fixed=available.filter(p=>!p.camera&&!p.module);
 for(const [width,height] of [[1440,700],[1920,917],[1536,701],[864,1341],[1048,500]])for(let mask=0;mask<(1<<optional.length);mask++){
  const selected=[...optional.filter((_,i)=>mask&(1<<i)),...fixed];
  const p=paneLayout(width,height,selected),[cw,ch]=p.cell,[ew,eh]=p.extent;
  assert.equal(p.panes.length,selected.length);
  assert.deepEqual(p.main,[0,0,2*cw,2*ch]);
  assert.ok(ew<=width+1e-8&&eh<=height+1e-8);
  const rects=[p.main,...p.panes.map(p=>p.rect)];
  for(const [i,[x,y,w,h]] of rects.entries()){
   assert.ok(x>=0&&y>=0&&x+w<=ew+1e-8&&y+h<=eh+1e-8);
   if(i===0||i<rects.length-1)assert.ok(Math.abs(w/h-16/9)<1e-8);
   for(const [xx,yy,ww,hh] of rects.slice(i+1))assert.ok(Math.min(x+w,xx+ww)-Math.max(x,xx)<1e-8||Math.min(y+h,yy+hh)-Math.max(y,yy)<1e-8,'No overlap');
  }
  assert.ok(Math.abs(rects.reduce((sum,[x,y,w,h])=>sum+w*h,0)-ew*eh)<1e-6,'No empty workspace area');
  const last=p.panes.at(-1).rect;assert.ok(Math.abs(last[0]+last[2]-ew)<1e-8&&Math.abs(last[1]+last[3]-eh)<1e-8);
  checked++;
 }
 const p=paneLayout(1440,700,available);
 assert.deepEqual([p.columns,p.rows],[4,3]);assert.deepEqual(p.main,[0,0,720,405]);
 assert.equal(p.panes.at(-1).id,'status');
}
console.log(`${checked} layouts: all checkboxes/sources, ordering, legacy selection, 16:9 cells, no holes/overlaps, last row absorbed`);
