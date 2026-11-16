import {operatorVideo} from '/web/video.mjs';

const storageKey='collection.auxiliaryViews';
const versionKey='collection.paneSelectionVersion';
const at=(value,path)=>path.split('.').reduce((v,key)=>v?.[key],value);
export function orderedPanes(panes){
  return [...panes].sort((a,b)=>(Number.isInteger(a.weight)?0:1)-(Number.isInteger(b.weight)?0:1)||
    ((a.weight??0)-(b.weight??0))||(a.id<b.id?-1:a.id>b.id?1:0));
}
export function paneVisible(pane,state){
  return Object.entries(pane.visible_when||{}).every(([key,value])=>at(state,key)===value);
}

// Main camera occupies a fixed 2x2 block. The final ordered pane absorbs the
// trailing row cells. Maximize occupied width, then common 16:9 cell size.
export function paneLayout(width,height,panes){
  const ordered=orderedPanes(panes),count=ordered.length;
  let best;
  for(let columns=2;columns<=6;columns++){
    const rows=Math.max(2,Math.ceil((4+count)/columns));
    const free=[];
    for(let y=0;y<rows;y++)for(let x=0;x<columns;x++)if(x>=2||y>=2)free.push([x,y]);
    const used=free.slice(0,count),last=used.at(-1),tail=free.slice(count);
    if(used.length!==count||(!last&&tail.length)||tail.some(([x,y])=>y!==last[1]||x<last[0]))continue;
    const w=Math.min(width/columns,height/rows*16/9),h=w*9/16;
    const extentWidth=columns*w;
    if(best){
      if(extentWidth<best.extent[0]-1e-6)continue;
      if(Math.abs(extentWidth-best.extent[0])<1e-6&&
         (w<best.cell[0]-1e-6||(Math.abs(w-best.cell[0])<1e-6&&columns*rows>=best.columns*best.rows)))continue;
    }
    const rects=used.map(([x,y],i)=>[x*w,y*h,(i===used.length-1?columns-x:1)*w,h]);
    best={columns,rows,cell:[w,h],extent:[columns*w,rows*h],main:[0,0,2*w,2*h],
      panes:ordered.map((p,i)=>({...p,rect:rects[i]}))};
  }
  return best;
}

export function savedViews(value,panes,legacy=false){
  return panes.filter(p=>!Array.isArray(value)||value.includes(p.id)||(legacy&&p.module)).map(p=>p.id);
}

export function auxiliaryPanel(root,status,toolbar) {
  const panes=new Map();let session=null,state={},saved=null,legacy=true;
  try{saved=JSON.parse(localStorage.getItem(storageKey));legacy=localStorage.getItem(versionKey)!=='1';}catch{}
  const choices=new Map();
  const workspace=root.parentElement,main=workspace.querySelector('.view');
  function place(element,[x,y,w,h]){Object.assign(element.style,{left:x+'px',top:y+'px',width:w+'px',height:h+'px'});}
  function layout(){
    const visible=[...panes.values()].filter(p=>!p.element.hidden).map(p=>p.definition);
    const container=workspace.parentElement;
    const occupied=[...container.children].filter(e=>e!==workspace).reduce((sum,e)=>sum+e.getBoundingClientRect().height,0);
    const plan=paneLayout(container.clientWidth,Math.max(1,container.clientHeight-occupied),visible);
    Object.assign(workspace.style,{flex:'none',width:plan.extent[0]+'px',height:plan.extent[1]+'px'});
    place(main,plan.main);
    plan.panes.forEach(p=>place(panes.get(p.id).element,p.rect));
    workspace.dataset.layout=`${plan.columns}x${plan.rows}`;
  }
  const video=operatorVideo(document.createElement('canvas'),{
    path:'/aux-video',onStatus:value=>status.textContent=value,
    drawFrame(frame,meta){
      for(const pane of panes.values()){
        if(!pane.canvas||pane.element.hidden)continue;
        const name=pane.definition.camera,t=meta.tiles[name];
        if(!t||!meta.views.includes(name))continue;
        if(pane.canvas.width!==t.width||pane.canvas.height!==t.height){pane.canvas.width=t.width;pane.canvas.height=t.height;}
        pane.canvas.getContext('2d',{alpha:false}).drawImage(frame,t.x,t.y,t.width,t.height,0,0,t.width,t.height);
      }
    }
  });
  function update(){
    const names=[];
    for(const pane of panes.values()){
      pane.element.hidden=!(pane.check?.checked??true)||!paneVisible(pane.definition,state);
      if(pane.label)pane.label.hidden=!paneVisible(pane.definition,state);
      if(!pane.element.hidden&&pane.canvas)names.push(pane.definition.camera);
      if(pane.element.hidden){
        pane.element.classList.remove('enlarged');pane.element.setAttribute('aria-expanded','false');
        if(pane.canvas){const ctx=pane.canvas.getContext('2d',{alpha:false});ctx.fillStyle='#000';ctx.fillRect(0,0,pane.canvas.width,pane.canvas.height);}
      }
    }
    root.hidden=false;layout();video.send({views:names});
    if(session&&names.length&&!document.hidden)video.connect(session);else video.stop();
  }
  function setPanes(definitions){
    for(const pane of panes.values()){pane.label?.remove();if(!pane.core)pane.element.remove();}
    panes.clear();
    const selected=savedViews(saved,definitions,legacy);
    for(const definition of orderedPanes(definitions)){
      const {id,title}=definition;
      let element=workspace.querySelector(`[data-core-pane="${id}"]`);
      const core=!!element;
      if(!element){element=document.createElement('section');element.className='aux-pane '+(definition.camera?'video-pane':'web-pane');root.append(element);}
      element.dataset.paneId=id;
      const pane={definition,element,core};
      if(!core){
        const heading=document.createElement('span');heading.className='pane-label';heading.textContent=title;element.append(heading);
      }
      if(definition.camera&&!core){
        pane.canvas=document.createElement('canvas');pane.canvas.width=960;pane.canvas.height=540;pane.canvas.setAttribute('aria-label',title+' simulation view');element.append(pane.canvas);
      }
      if(definition.camera||definition.module){
        const label=document.createElement('label'),check=document.createElement('input');
        check.type='checkbox';check.checked=choices.has(id)?choices.get(id):selected.includes(id);check.setAttribute('aria-label',title+' view');
        label.append(check,document.createTextNode(title));toolbar.append(label);pane.check=check;pane.label=label;
        check.onchange=()=>{
          for(const [name,p] of panes)if(p.check)choices.set(name,p.check.checked);
          const stored=new Set(Array.isArray(saved)?saved:[]);
          for(const [name,checked] of choices){if(checked)stored.add(name);else stored.delete(name);}
          saved=[...stored];legacy=false;
          try{localStorage.setItem(storageKey,JSON.stringify(saved));localStorage.setItem(versionKey,'1');}catch{}
          update();
        };
      }
      panes.set(id,pane);
    }
    update();
  }
  const observer=new ResizeObserver(layout);observer.observe(workspace.parentElement);
  for(const sibling of workspace.parentElement.children)if(sibling!==workspace)observer.observe(sibling);
  document.addEventListener('visibilitychange',()=>{if(document.hidden)video.stop();else update();});
  return {stats:video.stats,setPanes,host(id){return panes.get(id).element;},
    update(value){state=value;update();},setSession(id){session=id;update();},stop(){session=null;video.stop();}};
}

export function installPaneZoom(root){
  const panes=()=>root.querySelectorAll('.video-pane');
  function toggle(pane){const expand=!pane.classList.contains('enlarged');for(const p of panes()){p.classList.remove('enlarged');p.setAttribute('aria-expanded','false');}if(expand){pane.classList.add('enlarged');pane.setAttribute('aria-expanded','true');}}
  function prepare(pane){pane.tabIndex=0;pane.setAttribute('role','button');pane.setAttribute('aria-expanded','false');pane.title='Click to enlarge; click again to restore';}
  for(const pane of panes())prepare(pane);
  new MutationObserver(()=>{for(const pane of panes())if(!pane.hasAttribute('tabindex'))prepare(pane);}).observe(root,{childList:true,subtree:true});
  root.addEventListener('click',event=>{const pane=event.target.closest('.video-pane');if(pane)toggle(pane);});
  root.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){const pane=event.target.closest('.video-pane');if(pane){event.preventDefault();toggle(pane);}}});
  document.addEventListener('keydown',event=>{if(event.key==='Escape')for(const p of panes()){p.classList.remove('enlarged');p.setAttribute('aria-expanded','false');}});
}
