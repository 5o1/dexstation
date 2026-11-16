const sides=['left','right'];
const fingers=[['thumb','Thumb'],['index','Index'],['middle','Middle'],['ring','Ring'],['pinky','Little']];
const finite=value=>typeof value==='number'&&Number.isFinite(value);
const deg=value=>finite(value)?value*180/Math.PI:null;
const cell=(value,bad=false,title='',issue='')=>({text:finite(value)?value.toFixed(1):value??'—',state:value==null?'none':bad?'bad':'ok',title,issue});

// Pure status -> cells, shared by the shipped DOM renderer and Node tests.
export function teleopRows(state){
 const live=state.session_state==='running',t=state.teleop?.thresholds;
 const stale=age=>!!t&&finite(age)&&age>t.input_stale_s;
 const inputAge=channel=>state.realtime_metrics?.input_age_s?.[channel]??state.input_health?.[channel]?.age_s;
 const ages=side=>inputAge(side+'_arm');
 const arm=Array.from({length:7},(_,i)=>({label:`J${i+1}`,cells:sides.flatMap(side=>{
  const c=live?state.commands?.[side+'_arm']:null,sim=live?state.arm_q?.[sides.indexOf(side)]?.[i]:null;
  const follow=state.teleop?.arms?.[side],error=follow?.error_deg?.[i],duration=follow?.error_duration_s?.[i];
  const old=stale(ages(side)),clamped=!!(c?.clamp_flags?.absolute?.[i]||c?.clamp_flags?.startup?.[i]);
  const stuck=!!t&&finite(error)&&Math.abs(error)>t.arm_error_deg&&duration>t.error_duration_s;
  const ageTitle=finite(ages(side))?`Input ${(ages(side)*1000).toFixed(1)} ms`:'';
  const label=side==='left'?'L':'R',staleLabel=`${label} arm stale`;
  return [cell(c?.raw_master_deg?.[i],old,ageTitle,staleLabel),cell(deg(c?.positions?.[i]),old||clamped,
   [ageTitle,c?.clamp_flags?.absolute?.[i]?'Absolute clamp':'',c?.clamp_flags?.startup?.[i]?'Startup clamp':''].filter(Boolean).join(' · '),old?staleLabel:`${label} J${i+1} Cmd`),
   cell(c?deg(sim):null,old||stuck,finite(error)?`Δ ${error.toFixed(1)}° · ${duration.toFixed(2)} s · Target ${follow.effective_target_deg[i].toFixed(1)}°`:ageTitle,old?staleLabel:`${label} J${i+1} Sim`)];
 })}));
 const handAvailable=side=>live&&!!state.commands?.[side+'_hand']&&(state.source!=='ros2'||finite(state.glove_health?.[side]?.age_s));
 const hand=fingers.map(([finger,label])=>({label,cells:sides.flatMap(side=>{
  const f=handAvailable(side)?state.teleop?.hands?.[side]?.fingers?.[finger]:null;
  const old=stale(inputAge(side+'_hand'))||stale(state.glove_health?.[side]?.age_s);
  const stuck=!!t&&f?.max_error_deg>t.hand_error_deg&&f?.error_duration_s>t.error_duration_s;
  // Open or closed fingers sit on joint endpoints in normal use; Limit is
  // shown for inspection only and never raises an alert.
  const bad=old||stuck;
  const issue=`${side==='left'?'L':'R'} ${finger}`;
  return [cell(f?.max_error_deg,bad,f?`Δ max · ${f.error_duration_s.toFixed(2)} s`:'',issue),cell(f?.limited==null?null:f.limited?'YES':'NO',bad,'',issue)];
 })}));
 for(const [key,label] of [['glove','Glove'],['retarget','Retarget']])hand.push({label,cells:sides.map(side=>{
  const age=key==='glove'?state.glove_health?.[side]?.age_s:inputAge(side+'_hand');
  const available=key==='glove'?live&&finite(age):handAvailable(side)&&finite(age);
  return {...cell(available?`${Math.round(age*1000)} ms`:null,stale(age),'',`${side==='left'?'L':'R'} ${key} stale`),span:2};
 })});
 for(const [kind,rows] of Object.entries({arm,hand}))rows.forEach((r,i)=>r.cells.forEach((c,j)=>{c.id=`${kind}:${i}:${j}`;}));
 return {arm,hand};
}

// Summarize the exact red cells, including grouped stale sides/finger cells.
// No threshold or status decision is repeated here.
export function teleopSummary(rows){
 const issues=new Map();
 for(const row of Object.values(rows).flat())for(const c of row.cells)if(c.state==='bad'){
  if(!issues.has(c.issue))issues.set(c.issue,[]);
  issues.get(c.issue).push(c.id);
 }
 return [...issues].map(([label,cells])=>({label,cells}));
}

export function teleopTable(root,summary){
 const tables=[];
 let summarySignature;
 for(const [kind,label,columns,sideIndex] of [['arm','Arm °',['Master','Cmd','Sim'],null],['arm','Arm °',['Master','Cmd','Sim'],0],['arm','Arm °',['Master','Cmd','Sim'],1],['hand','Hand °',['Δ max','Limit'],null]]){
  const table=document.createElement('table');table.className='teleop-table';table.dataset.kind=kind;table.setAttribute('aria-label',label);
  if(kind==='arm')table.classList.add(sideIndex===null?'teleop-arm-wide':'teleop-arm-side');
  const head=table.createTHead(),groups=head.insertRow(),names=head.insertRow();
  const corner=document.createElement('th');corner.rowSpan=2;corner.textContent=label;groups.append(corner);
  const visibleSides=sideIndex===null?['L','R']:[['L','R'][sideIndex]];
  for(const side of visibleSides){const th=document.createElement('th');th.colSpan=columns.length;th.textContent=side;groups.append(th);}
  for(const side of visibleSides)for(const name of columns){const th=document.createElement('th');th.textContent=name;names.append(th);}
  const body=table.createTBody();root.append(table);tables.push({kind,sideIndex,body,rows:[]});
 }
 return state=>{
  const data=teleopRows(state);
  const issues=teleopSummary(data),signature=JSON.stringify(issues);
  if(signature!==summarySignature){
   summarySignature=signature;
   summary.dataset.state=issues.length?'bad':'ok';
   summary.title=issues.map(i=>i.label).join(' · ')||'OK';
   summary.replaceChildren(...(issues.length?issues:[{label:'OK',cells:[]}]).map(issue=>{
    const span=document.createElement('span');span.textContent=issue.label;span.dataset.cells=issue.cells.join(' ');return span;
   }));
  }
  for(const entry of tables){
   data[entry.kind].forEach((original,i)=>{
    const row=entry.sideIndex===null?original:{...original,cells:original.cells.slice(entry.sideIndex*3,entry.sideIndex*3+3)};
    let cells=entry.rows[i];
    if(!cells){const tr=entry.body.insertRow(),th=document.createElement('th');th.scope='row';th.textContent=row.label;tr.append(th);
     cells=row.cells.map(c=>{const td=tr.insertCell();td.colSpan=c.span||1;return td;});entry.rows[i]=cells;}
    row.cells.forEach((c,j)=>{cells[j].textContent=c.text;cells[j].dataset.state=c.state;cells[j].dataset.cellId=c.id;cells[j].title=c.title;});
   });
  }
 };
}
