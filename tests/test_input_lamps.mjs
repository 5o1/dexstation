import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFileSync} from 'node:fs';

// Execute shipped lamp logic with inert DOM objects; source extraction is deferred.
const html=readFileSync(new URL('../src/dexstation/server/web/index.html',import.meta.url),'utf8');
const block=html.slice(html.indexOf('const inputLamps='),html.indexOf('function show(state)'));
const handler=html.slice(html.indexOf('events.onerror='),html.lastIndexOf('</script>'));
{

const lamps=[],animations=[];
const healthy=()=>({age_s:.01,warning:false});
const state={source:'ros2',glove_health:{left:{state:'ready'},right:{state:'starting'}},
 input_chain:{state:'ready',ownership:'manual',components:{retarget:{state:'ready'},linkerta:{state:'starting'}}},
 input_health:{left_hand:healthy(),right_hand:healthy(),left_arm:healthy(),right_arm:healthy()}};
const ctx={state,document:{createElement:()=>({dataset:{},setAttribute(){},prepend(dot){this.dot=dot},
 animate(frames,options){animations.push({target:this,frames,options});return {cancel(){}}}})},
 $:()=>({replaceChildren:()=>{lamps.length=0},append:el=>lamps.push(el),classList:{add(){}}}),events:{},lastState:state,message:()=>{}};
vm.createContext(ctx);vm.runInContext(block,ctx);
const render=()=>vm.runInContext('showInputLamps(state)',ctx);
const states=()=>lamps.map(l=>l.dataset.state);
render();
assert.deepEqual(lamps.map(l=>l.textContent),['GLOVE L','GLOVE R','RETARGET','LTA']);
assert.deepEqual(states(),['ready','starting','ready','starting']);
assert(lamps.every(l=>l.title.includes('Manual')));
assert.equal(animations.length,0);
const original=lamps[2];for(let i=0;i<10;i++)render();
assert.equal(lamps[2],original);assert.equal(lamps.length,4);assert.equal(animations.length,0);
state.input_health.right_hand={age_s:.4,warning:true};render();
assert.equal(lamps[2].dataset.state,'starting');assert(lamps[2].title.includes('Stale right_hand'));
assert.equal(animations.length,1);render();assert.equal(animations.length,1);
state.input_chain.components.linkerta.state='ready';state.input_health.right_arm={age_s:null,warning:true};render();
assert.equal(lamps[3].dataset.state,'starting');assert(lamps[3].title.includes('Waiting right_arm'));
state.input_health.right_hand=healthy();state.input_health.right_arm=healthy();render();
assert.deepEqual(states(),['ready','starting','ready','ready']);assert.equal(animations.length,3);
state.input_chain.state='failed';state.input_chain.error='PID 4242 busy';render();
assert.deepEqual(states(),['failed','failed','failed','failed']);
assert(lamps.every(l=>l.title.includes('PID 4242')));assert.equal(animations.length,7);
assert(animations.every(a=>a.target.className==='signal-dot'&&a.options.duration===140&&a.options.iterations===undefined));
state.source='replay';render();assert.equal(lamps.length,0);
state.source='ros2';state.input_chain.state='ready';delete state.input_chain.error;render();
vm.runInContext(handler,ctx);const beforeDisconnect=animations.length;ctx.events.onerror();
assert.deepEqual(states(),['starting','starting','starting','starting']);
assert(lamps.every(l=>l.title==='Status disconnected'));assert.equal(animations.length,beforeDisconnect+3);
ctx.events.onerror();assert.equal(animations.length,beforeDisconnect+3);

}

{

const els={};const el=id=>els[id]??=(id==='inputChain'?{dataset:{},kids:[],append(x){this.kids.push(x)},replaceChildren(){this.kids=[]}}:{value:'',dataset:{}});
const state={source:'ros2',session_state:'running',signal_source:'real_robot',recording_status:{state:'idle'},input_chain:{state:'ready',components:{}},
 real_robot:{left_arm:{state:'ready'},right_arm:{state:'failed',error:'connect: timeout'},left_hand:{state:'busy',error:'Another program is using can1'},right_hand:{state:'ready'}},
 input_health:{left_arm:{age_s:.01},left_hand:{age_s:.02},right_hand:{age_s:.02}}};
const ctx={state,$:el,localStorage:{setItem(){}},document:{activeElement:null,createElement:()=>({dataset:{},setAttribute(){},prepend(){},animate(){return {cancel(){}}},lastChild:{}})}};
vm.createContext(ctx);vm.runInContext(block,ctx);vm.runInContext('showInputLamps(state)',ctx);
const lamps=els.inputChain.kids;assert.deepEqual(lamps.map(l=>l.textContent),['ARM L','ARM R','HAND L','HAND R']);
assert.deepEqual(lamps.map(l=>l.dataset.state),['ready','failed','busy','ready']);
assert.equal(lamps[1].title,'connect: timeout');assert.equal(els.startRecord.disabled,true);assert.equal(els.signalSource.value,'real_robot');
state.signal_source='teleop';vm.runInContext('showInputLamps(state)',ctx);
assert.deepEqual(els.inputChain.kids.map(l=>l.textContent),['GLOVE L','GLOVE R','RETARGET','LTA']);assert.equal(els.startRecord.disabled,false);

}

{

const select={value:'manus_dex'},root={dataset:{},append(){},replaceChildren(){}};
const state={source:'ros2',session_state:'running',recording_status:{state:'recording'},input_chain:{state:'starting',retarget:'external_retarget'}};
const saved={};const ctx={state,retargetStorageKey:'dexstation.input.retarget',$:id=>id==='retarget'?select:root,localStorage:{setItem:(k,v)=>saved[k]=v},document:{activeElement:null,createElement:()=>({dataset:{},setAttribute(){},prepend(){},animate(){return {cancel(){}}}})}};
vm.createContext(ctx);vm.runInContext(block,ctx);vm.runInContext('showInputLamps(state)',ctx);
assert.equal(select.value,'external_retarget');assert.equal(select.disabled,true);assert.equal(saved['dexstation.input.retarget'],select.value);
state.recording_status.state='idle';state.input_chain.retarget='manus_anatomical';vm.runInContext('showInputLamps(state)',ctx);
assert.equal(select.disabled,false);assert.equal(select.value,'manus_anatomical');

}
console.log('Input lamps: freshness, transitions, real-robot mode and retarget locking passed');
