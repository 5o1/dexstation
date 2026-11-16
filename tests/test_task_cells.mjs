import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
const {taskCells, configureTaskSelection, takeoverView} = await import('../src/dexstation/server/web/task-cells.mjs');
const example = name => JSON.parse(readFileSync(new URL(`../src/dexstation/contracts/examples/http_${name}.json`, import.meta.url))).response.body;

// A small DOM double, without a browser dependency. Browser acceptance uses Chrome.
class Element {
 constructor(tag, ownerDocument) {this.tagName=tag.toUpperCase();this.ownerDocument=ownerDocument;this.children=[];this.dataset={};this.style={};this.hidden=false;this._text='';this.value='';}
 append(...children) {this.children.push(...children);}
 prepend(child) {this.children.unshift(child);}
 replaceChildren(...children) {this.children=children;this._text='';}
 set textContent(text) {this._text=String(text);this.children=[];}
 get textContent() {return this._text+this.children.map(c=>c.textContent).join('');}
 attachShadow() {return this.shadowRoot=new Element('shadow-root',this.ownerDocument);}
 click() {return this.onclick?.();}
}
const listeners = new Set();
const document = {createElement: tag=>new Element(tag,document),createTextNode:text=>({textContent:text}),addEventListener:(_,fn)=>listeners.add(fn),removeEventListener:(_,fn)=>listeners.delete(fn)};
const root=()=>document.createElement('div');
const find=(element,predicate)=>[element,...element.children.flatMap(child=>child.children?find(child,()=>true):[])].filter(predicate);
const content=(container,id)=>container.children.find(host=>host.dataset.cellId===id)?.shadowRoot;
const status=example('status'), config=example('config'), spec=example('task_cells');

const selection=root();configureTaskSelection(selection,config);
assert.equal(selection.value,'empty');assert.equal(selection.children[0].textContent,'Empty table');
configureTaskSelection(selection,{...config,defaults:{task:'tabletop_cube_grasp'}});assert.equal(selection.value,'tabletop_cube_grasp');
configureTaskSelection(selection,{tasks:config.tasks});assert.equal(selection.value,'empty');

const calls=[],banners=[],spoken=[],container=root();
const cells=taskCells(container,{request:async(path,body)=>{calls.push([path,body]);return body?{accepted:false,error:'Recording active'}:spec;},banner:(...args)=>banners.push(args),speak:text=>spoken.push(text)});
await cells.update(status);
assert.equal(calls[0][0],'/api/v1/tasks/tabletop_cube_grasp/cells');
assert.equal(container.children.length,3);assert.match(content(container,'cube').textContent,/Height: 0.792 m/);
const reset=find(content(container,'controls'),e=>e.tagName==='BUTTON')[0];await reset.click();
assert.deepEqual(calls.at(-1),['/api/v1/task/action',{action:'reset',params:{}}]);
assert.match(content(container,'controls').textContent,/Recording active/);assert.equal(banners.at(-1)[0],'Recording active');
assert.doesNotMatch(content(container,'cube').textContent,/channels|cell_state/); // Builtin sees only its own payload.
const revised=structuredClone(status);revised.task.channels={cube_pose:[0,0,999]};revised.task.cell_state.result={text:'SUCCESS',level:'ok'};
await cells.update(revised);assert.match(content(container,'result').textContent,/SUCCESS/);assert.equal(spoken.at(-1),'SUCCESS');
await cells.update(revised);assert.equal(spoken.filter(text=>text==='SUCCESS').length,1);
for(const flag of ['accepted','success']) {revised.task.cell_state.result={[flag]:false,error:'Action failed'};await cells.update(revised);assert.match(content(container,'result').textContent,/Action failed/);assert.ok(find(content(container,'result'),e=>e.className==='banner error').length);}
cells.unmount();assert.equal(listeners.size,0);assert.equal(container.children.length,0);

// Mount, import and sync/async subscription failures remain inside their cells.
const errorRoot=root(), received=[], lifecycle=[], external=[];
const modules={
 good:{mount(root,api){api.subscribe(value=>{received.push(value);root.textContent=String(value?.channels?.value);});external.push(api);},unmount(){lifecycle.push('good');}},
 mount:{mount(){throw Error('mount exploded');}},
 callback:{mount(root,api){api.subscribe(()=>{throw Error('callback exploded');});}},
 async:{mount(root,api){api.subscribe(async()=>{throw Error('async exploded');});}},
};
const moduleSpec={cells:['good','mount','callback','async','import'].map(id=>({id,title:id,module:id}))};
const modular=taskCells(errorRoot,{request:async()=>moduleSpec,importModule:async id=>{if(id==='import')throw Error('import exploded');return modules[id];}});
const moduleTask={id:'mixed',channels:{value:7},actions:{},cell_state:{other:{secret:99}}};
await modular.update({task:moduleTask});
await Promise.resolve();
assert.deepEqual(received,[moduleTask]);
assert.equal(received[0].channels.value,7);
for(const id of ['mount','callback','async','import']) {assert.match(content(errorRoot,id).textContent,new RegExp(`${id} exploded`));assert.equal(find(content(errorRoot,id),e=>e.className==='cell-error').length,1);}
const nextModuleTask={...moduleTask,channels:{value:8},cell_state:{good:{plugin_payload:true}}};
await modular.update({task:nextModuleTask});assert.deepEqual(received,[moduleTask,nextModuleTask]);
assert.equal(received[1].cell_state.good.plugin_payload,true);
assert.equal(external[0].status().task.id,'mixed');assert.match(content(errorRoot,'good').textContent,/8/);
modular.unmount();assert.deepEqual(lifecycle,['good']);

const lampRoot=root();
const lampCells=taskCells(lampRoot,{request:async()=>({cells:[{id:'lights',title:'Signals',builtin:'lamps',source:'nested',items:[{key:'left',label:'Left'},{key:'right',label:'Right'}]},{id:'note',title:'Note',builtin:'banner',source:'nested',states:{warn:'warn'}}]})});
await lampCells.update({task:{id:'lamps',cell_state:{lights:{nested:{left:'ok'}},note:{nested:{text:'Hold',level:'warn'}}}}});
assert.deepEqual(find(content(lampRoot,'lights'),e=>e.className==='signal-lamp').map(e=>e.dataset.state),['ok','off']);assert.match(content(lampRoot,'note').textContent,/Hold/);

const loadRoot=root();await taskCells(loadRoot,{request:async()=>{throw Error('HTTP unavailable');}}).update({task:{id:'offline'}});
assert.match(content(loadRoot,'load').textContent,/HTTP unavailable/);

// A slow old task must never mount into the newly selected empty task.
let resolveOld;const raceRoot=root();
const race=taskCells(raceRoot,{request:async path=>path.includes('/old/')?await new Promise(resolve=>{resolveOld=resolve;}):{cells:[]}});
const old=race.update({task:{id:'old'}});await race.update({task:{id:'empty',cell_state:{}}});resolveOld(moduleSpec);await old;
assert.equal(raceRoot.children.length,0);assert.equal(raceRoot.hidden,true);

const takeover=root(),cues=[];const render=takeoverView(takeover,{speak:text=>cues.push(text)});
render(status);assert.match(takeover.textContent,/Left arm waiting for input/);assert.equal(cues.length,1);
render(status);assert.equal(cues.length,1);
render({takeover:{left:{state:'matching',deviation_deg:[12,-8]},right:{state:'following'}}});
assert.match(takeover.textContent,/J1 \+12.0°, J2 -8.0°/);assert.match(takeover.textContent,/Right arm following/);
render({takeover:{left:{state:'matching',deviation_deg:[11,-7]},right:{state:'following'}}});assert.equal(cues.length,2,'Changing numbers must not repeat speech at status frequency');
render({takeover:{left:{state:'hold',reason:'Angular limit'},right:{state:'following'}}});assert.match(takeover.textContent,/HOLD: Angular limit/);assert.match(cues.at(-1),/HOLD/);
render({});assert.equal(takeover.hidden,true);

// Execute the actual session submit handler, including authenticated request path.
const html=readFileSync(new URL('../src/dexstation/server/web/index.html',import.meta.url),'utf8');
const handler=html.slice(html.indexOf("$('sessionForm').onsubmit="),html.indexOf("$('endSession').onclick="));
const elements=Object.fromEntries(['sessionForm','task','backend','source','motionProfile','retarget','replay','autopilotTasks','duration'].map(id=>[id,{value:{task:'tabletop_cube_grasp',backend:'sapien',source:'udp',motionProfile:'station'}[id]||''}]));
let start;
vm.runInNewContext(handler,{$:id=>elements[id],action:fn=>fn(),gloveAudio:{unlock(){}},api:async(path,body)=>{start={path,body};},videoSession:null,cells:{sessionOptions:async()=>({})}});
elements.sessionForm.onsubmit({preventDefault(){}});await new Promise(resolve=>setImmediate(resolve));
assert.equal(start.path,'/api/v1/session/start');assert.equal(start.body.task,'tabletop_cube_grasp');assert.equal(start.body.scene_variant,'full');
console.log('Task cells: fixture rendering, isolation, actions, teardown, race, selection, takeover and start request passed');

// Mount both real task modules through the production mounter and its fake API.
Element.prototype.setAttribute = function(name,value) {this[name]=String(value);};
Element.prototype.remove = function(){if(this.parentNode)this.parentNode.children=this.parentNode.children.filter(child=>child!==this);};
const append = Element.prototype.append;
Element.prototype.append = function(...children){children.forEach(child=>{child.parentNode=this;});append.apply(this,children);};
const pluginRoot=root(), gridRoot=root(), sessionControls=root();
const grid={
 setPanes(panes){gridRoot.replaceChildren(...panes.map(p=>{const host=root();host.definition=p;return host;}));},
 host(id){return gridRoot.children.find(h=>h.definition.id===id);},
 update(status){for(const host of gridRoot.children)host.hidden=!Object.entries(host.definition.visible_when||{}).every(([key,value])=>status[key]===value);}
};
const taskConfig={autopilot_tasks:['/lab/first.json','/lab/second.json'],defaults:{autopilot_tasks:'/lab/first.json'}};
const pluginPanes=[
 {id:'gamepad_signals',title:'Gamepad signals',module:'gamepad-overlay.mjs',weight:50},
 {id:'autopilot',title:'Autopilot',module:'autopilot-status.mjs',visible_when:{source:'ik_autopilot'}},
];
const actualModules={};
for(const name of ['gamepad-overlay.mjs','autopilot-status.mjs'])actualModules[name]=await import(`../src/dexstation/task_plugins/xbox_gamepad_v3_3/web/${name}`);
const actual=taskCells(pluginRoot,{config:taskConfig,grid,sessionControls,request:async()=>({cells:[],panes:pluginPanes}),importModule:async name=>actualModules[name]});
const pluginState={source:'ik_autopilot',autopilot:{set:'Fixture',success:3},task:{id:'xbox_gamepad_v3_3',channels:{gamepad_signal:Array(32).fill(0),gamepad_mask:Array(32).fill(true)}}};
await actual.update(pluginState);
assert.equal(sessionControls.children.length,1);
assert.equal(pluginRoot.children.length,0,'Gamepad task has no below-grid cells');
const taskSelect=find(sessionControls.children[0].shadowRoot,e=>e.tagName==='SELECT')[0];
assert.deepEqual(taskSelect.children.map(option=>option.textContent),['first.json','second.json']);
assert.equal(taskSelect.value,'/lab/first.json');
assert.match(content(gridRoot,'autopilot').textContent,/"success": 3/);
const frame=find(content(gridRoot,'gamepad_signals'),e=>e.tagName==='IFRAME')[0];assert.ok(frame.src.includes('gamepad-overlay.html?source=canonical'));
const frames=[];frame.contentWindow={postMessage:(value,origin)=>frames.push({value,origin})};frame.onload();
assert.deepEqual(frames[0].value.values,pluginState.task.channels.gamepad_signal);
taskSelect.value='/lab/second.json';
assert.deepEqual(await actual.sessionOptions(),{autopilot_tasks:'/lab/second.json'});
await actual.update({...pluginState,source:'udp'});
assert.equal(gridRoot.children.find(host=>host.dataset.cellId==='autopilot').hidden,true);
assert.equal(sessionControls.children[0].hidden,true);
await actual.update(pluginState);assert.equal(sessionControls.children[0].hidden,false);
// Exercise the real submit handler with the chosen plugin option.
vm.runInNewContext(handler,{$:id=>elements[id],action:fn=>fn(),gloveAudio:{unlock(){}},api:async(path,body)=>{start={path,body};},videoSession:null,cells:actual});
elements.sessionForm.onsubmit({preventDefault(){}});await new Promise(resolve=>setImmediate(resolve));
assert.equal(start.body.autopilot_tasks,'/lab/second.json');
await actual.update({task:{id:'empty'}}); // The fixture still returns cells; unmount explicitly verifies resource ownership.
actual.unmount();assert.equal(sessionControls.children.length,0);assert.equal(gridRoot.children.length,0,'Unmount clears plugin grid panes');
assert.doesNotMatch(html,/gamepad|autopilot/i);
console.log('Gamepad plugin cells: actual mounts, values, visibility, frame data, chosen start option and cleanup passed');
