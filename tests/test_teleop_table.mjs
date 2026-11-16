import assert from 'node:assert/strict';
import test from 'node:test';
import {pathToFileURL} from 'node:url';
import {teleopRows,teleopSummary} from '../src/dexstation/server/web/teleop-table.mjs';

const t={arm_error_deg:10,hand_error_deg:15,error_duration_s:.5,input_stale_s:.1};
function status(){
 const s={session_state:'running',source:'udp',commands:{},arm_q:[[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]],input_health:{},glove_health:{},teleop:{thresholds:t,arms:{},hands:{}}};
 for(const side of ['left','right']){
  s.commands[side+'_arm']={positions:[0,0,0,0,0,0,0],raw_master_deg:[0,0,0,0,0,0,0],clamp_flags:{absolute:Array(7).fill(false),startup:Array(7).fill(false)}};
  s.input_health[side+'_arm']={age_s:.005};
  s.teleop.arms[side]={effective_target_deg:Array(7).fill(0),error_deg:Array(7).fill(0),error_duration_s:Array(7).fill(0)};
 }
 return s;
}
function onlyRed(rows,expected){
 const actual=[];
 rows.arm.forEach((r,i)=>r.cells.forEach((c,j)=>{if(c.state==='bad')actual.push([i,j]);else assert.equal(c.state,'ok');}));
 assert.deepEqual(actual,expected);
}
const clamp=status();clamp.commands.right_arm.raw_master_deg[2]=171;clamp.commands.right_arm.positions[2]=175*Math.PI/180;clamp.commands.right_arm.clamp_flags.absolute[2]=true;
onlyRed(teleopRows(clamp),[[2,4]]);assert.equal(teleopRows(clamp).arm[2].cells[4].text,'175.0');
const stuck=status();stuck.arm_q[1][2]=-20*Math.PI/180;stuck.teleop.arms.right.error_deg[2]=-20;stuck.teleop.arms.right.error_duration_s[2]=.51;
onlyRed(teleopRows(stuck),[[2,5]]);
stuck.teleop.arms.right.error_duration_s[2]=.5;onlyRed(teleopRows(stuck),[]);
stuck.teleop.arms.right.error_duration_s[2]=1;stuck.teleop.arms.right.error_deg[2]=10;onlyRed(teleopRows(stuck),[]);
const startup=status();startup.commands.left_arm.clamp_flags.startup[6]=true;onlyRed(teleopRows(startup),[[6,1]]);
const stale=status();stale.input_health.right_arm.age_s=.101;
onlyRed(teleopRows(stale),Array.from({length:7},(_,i)=>[i,3]).flatMap(([i,j])=>[[i,j],[i,j+1],[i,j+2]]));
stale.input_health.right_arm.age_s=.1;onlyRed(teleopRows(stale),[]);
stale.realtime_metrics={input_age_s:{right_arm:.2}}; // A decode failure must not refresh the effective input.
onlyRed(teleopRows(stale),Array.from({length:7},(_,i)=>[i,3]).flatMap(([i,j])=>[[i,j],[i,j+1],[i,j+2]]));
for(const state of [status(),{...status(),source:'ros2'},{}, {...status(),session_state:'stopped'}]){
 for(const row of teleopRows(state).hand)for(const c of row.cells){assert.equal(c.text,'—');assert.equal(c.state,'none');}
}
const hands=status();
for(const side of ['left','right']){
 hands.commands[side+'_hand']={joint_positions:{}};hands.input_health[side+'_hand']={age_s:.005};hands.glove_health[side]={age_s:.005};
 hands.teleop.hands[side]={fingers:Object.fromEntries(['thumb','index','middle','ring','pinky'].map(f=>[f,{max_error_deg:1,error_duration_s:0,limited:false}]))};
}
hands.teleop.hands.right.fingers.index={max_error_deg:16,error_duration_s:.51,limited:false};
hands.teleop.hands.left.fingers.thumb.limited=true;
const rows=teleopRows(hands).hand;
assert.deepEqual(rows[0].cells.map(c=>c.state),['ok','ok','ok','ok']);
assert.equal(rows[0].cells[1].text,'YES');
assert.deepEqual(rows[1].cells.map(c=>c.state),['ok','ok','bad','bad']);
assert.ok(rows.slice(2).every(r=>r.cells.every(c=>c.state==='ok')));

// Every summary item owns its exact set of logical red cells. Grouped stale
// sides and fingers neither omit a red cell nor manufacture additional ones.
const combined=structuredClone(clamp);
combined.teleop.arms.right.error_deg[3]=-20;combined.teleop.arms.right.error_duration_s[3]=.51;
assert.deepEqual(teleopSummary(teleopRows(combined)).map(i=>i.label),['R J3 Cmd','R J4 Sim']);
assert.deepEqual(teleopSummary(teleopRows(stale)).map(i=>i.label),['R arm stale']);
assert.deepEqual(teleopSummary(teleopRows(hands)).map(i=>i.label),['R index']);
const expiredHands=structuredClone(hands);expiredHands.glove_health.right.age_s=.2;expiredHands.input_health.left_hand.age_s=.2;
for(const s of [status(),clamp,stuck,startup,stale,hands,combined,expiredHands,{}, {...combined,session_state:'stopped'}]){
 const data=teleopRows(s),red=Object.values(data).flatMap(rs=>rs.flatMap(r=>r.cells.filter(c=>c.state==='bad').map(c=>c.id))).sort();
 const summary=teleopSummary(data);
 assert.deepEqual(summary.flatMap(i=>i.cells).sort(),red);
 assert(summary.every(i=>i.label&&i.cells.length));
}
assert.deepEqual(teleopSummary(teleopRows(status())),[]);
console.log('Teleop table and exact summary/red-cell correspondence: passed');

// Set these only for the isolated UDP fixture on the configured compute node.
// This runs in Node, using a real layout engine rather than a mocked scrollTop.
test('fixed summary remains visible after scrolling the smallest six-view pane to the bottom',{
 skip:!process.env.TELEOP_REVIEW_URL,timeout:60000,
},async()=>{
 const {firefox}=await import(pathToFileURL(process.env.TELEOP_PLAYWRIGHT_MODULE).href);
 const browser=await firefox.launch({headless:true,env:{...process.env,DISPLAY:'',CUDA_VISIBLE_DEVICES:'',MOZ_WEBRENDER_SOFTWARE:'1'},firefoxUserPrefs:{'gfx.webrender.software':true,'layers.acceleration.disabled':true}});
 try{
  const context=await browser.newContext({baseURL:process.env.TELEOP_REVIEW_URL,viewport:{width:1048,height:663},deviceScaleFactor:1});
  await context.addInitScript(()=>localStorage.setItem('collection.auxiliaryViews',JSON.stringify(['face','back','left','right','table'])));
  const page=await context.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto('/');
  const mode=async value=>{const r=await page.request.post('/review/mode',{headers:{'X-Gamepad-VLA-Client':'web'},data:{mode:value}});assert(r.ok(),await r.text());};
  await mode('normal');
  await page.waitForFunction(()=>document.querySelector('#teleopSummary')?.textContent==='OK');
  assert.equal(await page.locator('#teleopSummary').evaluate(e=>getComputedStyle(e).color),'rgb(61, 220, 132)');
  await mode('abnormal');
  await page.waitForFunction(()=>[...document.querySelectorAll('#teleopSummary span')].map(e=>e.textContent).join('|')==='R J3 Cmd|R J4 Sim');
  await page.evaluate(()=>document.fonts.ready);
  const before=await page.locator('#teleopSummary').boundingBox();
  await page.locator('.teleop-content').evaluate(e=>{e.scrollTop=e.scrollHeight;});
  const result=await page.evaluate(()=>{
   const summary=document.querySelector('#teleopSummary'),scroll=document.querySelector('.teleop-content'),pane=document.querySelector('#statusTile');
   const box=summary.getBoundingClientRect(),p=pane.getBoundingClientRect();
   return {scrollTop:scroll.scrollTop,bottom:scroll.scrollHeight-scroll.clientHeight,scrollWidth:scroll.scrollWidth,clientWidth:scroll.clientWidth,paneWidth:p.width,
    visible:box.top>=p.top&&box.bottom<=p.bottom&&box.left>=p.left&&box.right<=p.right&&getComputedStyle(summary).visibility==='visible',
    color:getComputedStyle(summary).color,red:[...new Set([...document.querySelectorAll('.teleop-table td[data-state=bad]')].filter(td=>td.getClientRects().length).map(td=>td.dataset.cellId))].sort(),
    summarized:[...summary.children].flatMap(e=>e.dataset.cells.split(' ')).sort()};
  });
  assert(result.paneWidth<260);assert(result.scrollTop>0);assert(Math.abs(result.scrollTop-result.bottom)<=1);
  assert.equal(result.scrollWidth,result.clientWidth);assert(result.visible);assert.equal(result.color,'rgb(255, 59, 48)');
  assert.deepEqual(await page.locator('#teleopSummary').boundingBox(),before);
  assert.deepEqual(result.summarized,result.red);assert.deepEqual(errors,[]);
  if(process.env.TELEOP_REVIEW_SCREENSHOT){
   await page.screenshot({path:process.env.TELEOP_REVIEW_SCREENSHOT});
   await page.locator('#statusTile').screenshot({path:process.env.TELEOP_REVIEW_SCREENSHOT.replace(/\.png$/,'-status.png')});
  }
  console.log('Scrolled summary browser checks:',JSON.stringify(result));
 }finally{await browser.close();}
});
