export const CLIENT_HEADER = 'X-Dexstation-Client';
// Task UI protocol: modules receive status.task; builtins receive cell_state[their own id].
const at = (value, path) => path ? String(path).replace(/\[(\d+)\]/g, '.$1').split('.').reduce((v, key) => v?.[key], value) : value;
const display = value => value == null ? '—' : typeof value === 'object' ? JSON.stringify(value) : String(value);
const styles = `
:host{display:block;color:var(--text);font:14px var(--font-ui)}
*{box-sizing:border-box}h3{font-size:14px;margin:0 0 8px;color:var(--muted)}
.values{display:flex;flex-wrap:wrap;gap:8px 20px}.value{font-family:var(--font-number)}
button{font:inherit;background:var(--bg);color:var(--text);border:1px solid var(--line);padding:4px 8px;cursor:pointer;margin:0 6px 6px 0}
.banner{font-size:24px;overflow-wrap:anywhere}.info{color:var(--text)}.ok{color:var(--ok)}.warn{color:var(--wait)}.error{color:var(--err)}
.cell-error{color:var(--err);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin:0}
.signal-lamp{display:inline-flex;align-items:center;gap:4px;font:12px var(--font-number);color:var(--muted);margin-right:12px}
.signal-dot{display:block;flex:none;width:6px;height:6px;border-radius:50%;background:var(--muted)}
.signal-lamp[data-state=ok] .signal-dot{background:var(--ok)}.signal-lamp[data-state=warn] .signal-dot{background:var(--wait)}.signal-lamp[data-state=error] .signal-dot{background:var(--err)}
`;

function builtin(cell, root, api, document) {
 const value = payload => at(payload, typeof cell.source === 'string' ? cell.source : '');
 if (cell.builtin === 'controls') {
  const listeners = [];
  for (const control of cell.controls || []) {
   const button = document.createElement('button');button.type = 'button';button.textContent = control.label;
   button.onclick = () => api.action(control.action, control.params || {});
   root.append(button);
   if (control.hotkey) {
    const keydown = event => {
     if (event.repeat || event.ctrlKey || event.metaKey || event.altKey || /INPUT|TEXTAREA|SELECT/.test(event.target.tagName)) return;
     if (event.key.toLowerCase() === control.hotkey.toLowerCase()) {event.preventDefault();button.click();}
    };
    document.addEventListener('keydown', keydown);listeners.push(keydown);
   }
  }
  return () => listeners.forEach(fn => document.removeEventListener('keydown', fn));
 }
 if (cell.builtin === 'values') {
  const values = document.createElement('div');values.className = 'values';root.append(values);
  api.subscribe(payload => {
   const data = value(payload);
   const fields = cell.fields || Object.keys(data || {}).map(key => ({key, label:key}));
   values.replaceChildren(...fields.map(field => {
    const node = document.createElement('span'), item = at(data, field.key);
    const text = typeof item === 'number' && field.digits != null ? item.toFixed(field.digits) : display(item);
    node.className = 'value';node.textContent = `${field.label}: ${text}${field.unit ? ' '+field.unit : ''}`;return node;
   }));
  });return;
 }
 if (cell.builtin === 'lamps') {
  const lamps = (cell.items || []).map(item => {
   const lamp = document.createElement('span'), dot = document.createElement('span');
   lamp.className = 'signal-lamp';dot.className = 'signal-dot';lamp.append(dot, document.createTextNode(item.label));root.append(lamp);return {item, lamp};
  });
  api.subscribe(payload => lamps.forEach(({item, lamp}) => {lamp.dataset.state = at(value(payload), item.key) ?? 'off';}));return;
 }
 if (cell.builtin === 'banner') {
  const banner = document.createElement('div');root.append(banner);let previous;
  api.subscribe(payload => {
   const data = value(payload) || {};
   const failed = data.accepted === false || data.success === false;
   const text = String((failed ? data.error : data.text) ?? ''), level = failed ? 'error' : data.level ?? 'info';
   banner.className = `banner ${cell.states?.[level] ?? level}`;banner.textContent = text;
   if (cell.speak_on_change && text && text !== previous) api.speak(text);
   previous = text;
  });return;
 }
 throw Error(`Unknown builtin: ${cell.builtin}`);
}

export function taskCells(root, {request, config = {}, sessionControls = null, grid = null, banner = () => {}, speak = () => {}, importModule = url => import(url), document = root.ownerDocument} = {}) {
 let latest = {}, taskId = null, generation = 0, records = [], pending = Promise.resolve();
 const current = record => !record.disposed && !record.failed;
 function fail(record, error) {
  if (!current(record)) return;
  record.failed = true;record.callbacks.clear();banner(null, record.cell.id);
  const line = document.createElement('p');line.className = 'cell-error';line.textContent = `Cell error: ${String(error?.message || error).replace(/\s+/g, ' ')}`;line.title = line.textContent;
  record.content.replaceChildren(line);
 }
 function call(record, fn, ...args) {
  if (!current(record)) return;
  try {const result = fn(...args);if (result?.then) result.catch(error => fail(record, error));} catch (error) {fail(record, error);}
 }
 function cleanup(record) {
  record.disposed = true;record.callbacks.clear();banner(null, record.cell.id);
  record.controls.forEach(node => node.remove());
  try {const result = record.unmount?.();result?.catch?.(() => {});} catch {} // Detached cells cannot affect the active page.
 }
 function notify(record) {
  const visible = Object.entries(record.cell.visible_when || {}).every(([key, value]) => at(latest, key) === value);
  if (!record.grid) record.host.hidden = !visible;
  record.controls.forEach(node => {node.hidden = !visible;});
  for (const fn of record.callbacks) call(record, fn, (record.cell.module ? latest.task : latest.task?.cell_state?.[record.cell.id]));
 }
 async function mount(cell, epoch, loadError, inGrid = false) {
  const host = inGrid ? grid.host(cell.id) : document.createElement('section');
  if (!inGrid) host.className = 'task-cell';host.dataset.cellId = cell.id;host.dataset.size = cell.size || '';
  const shadow = host.attachShadow({mode:'open'}), style = document.createElement('style'), title = document.createElement('h3'), content = document.createElement('div');
  style.textContent = styles;title.textContent = cell.title;shadow.append(style, title, content);if (!inGrid) root.append(host);
  if (inGrid) style.textContent += ':host{height:100%;overflow:hidden}h3{position:absolute;left:0;top:0;z-index:1;margin:0;padding:3px 5px;font:12px/1.2 var(--font-number);text-transform:uppercase;color:var(--text);background:#000b;pointer-events:none}.content{height:100%;min-height:0;overflow:auto;padding:26px 8px 6px}';
  content.className = 'content';
  const record = {grid:inGrid, cell, host, content, callbacks:new Set(), controls:[], options:null, disposed:false, failed:false};records.push(record);
  if (loadError) {fail(record, loadError);return;}
  const api = {
   config,
   addSessionControl(node) {
    if (sessionControls && current(record)) {sessionControls.append(node);record.controls.push(node);node.hidden=record.host.hidden;}
   },
   sessionOptions(read) {record.options = read;},
   subscribe(fn) {if (current(record)) {record.callbacks.add(fn);call(record, fn, (cell.module ? latest.task : latest.task?.cell_state?.[cell.id]));}},
   async action(name, params = {}) {
    let reply;
    try {reply = await request('/api/v1/task/action', {action:name, params});}
    catch (error) {reply = {accepted:false, error:String(error.message || error)};}
    if (current(record) && (reply.accepted === false || reply.success === false)) {
     const text = String(reply.error ?? '');banner(text, cell.id);
     const line = document.createElement('div');line.className = 'banner error';line.textContent = text;content.append(line);
    }
    return reply;
   },
   banner(text) {if (current(record)) banner(text, cell.id);},
   speak(text) {if (current(record)) speak(text);},
   status() {return latest;}
  };
  notify(record);
  try {
   if (cell.css) {const link = document.createElement('link');link.rel = 'stylesheet';link.href = cell.css;link.onerror = () => fail(record, Error('Stylesheet failed'));shadow.prepend(link);}
   if (cell.module) {
    const module = await importModule(cell.module);
    if (epoch !== generation) return;
    await module.mount(content, api);record.unmount = module.unmount ? () => module.unmount() : undefined;
    if (record.disposed || record.failed) {try {await record.unmount?.();} catch {} record.unmount = undefined;}
   } else record.unmount = builtin(cell, content, api, document);
  } catch (error) {fail(record, error);}
 }
 async function select(id) {
  taskId = id;const epoch = ++generation;
  records.forEach(cleanup);records = [];root.replaceChildren();root.hidden = true;
  grid?.setPanes(config.panes || []);grid?.update(latest);
  if (!id) return;
  let definition;
  try {definition = await request(`/api/v1/tasks/${encodeURIComponent(id)}/cells`);}
  catch (error) {
   if (epoch !== generation) return;
   await mount({id:'load', title:'Task'}, epoch, error);root.hidden = false;return;
  }
  if (epoch !== generation) return;
  root.hidden = !definition.cells.length;
  grid?.setPanes(definition.panes || []);grid?.update(latest);
  await Promise.all([...definition.cells.map(cell => mount(cell, epoch)),
   ...(grid ? (definition.panes || []).filter(p=>p.module).map(p=>mount(p,epoch,undefined,true)) : [])]);
 }
 return {
  async sessionOptions() {
   await pending;
   return Object.assign({}, ...records.filter(current).map(record => record.options?.() || {}));
  },
  update(status) {
   latest = {...status,source_capabilities:config.pane_source_capabilities?.[status.source] || {}};grid?.update(latest);const id = status.task?.id ?? null;
   if (id !== taskId) pending = select(id);
   else records.forEach(notify);
   return pending;
  },
  unmount() {generation++;records.forEach(cleanup);records = [];taskId = null;root.replaceChildren();root.hidden = true;grid?.setPanes(config.panes || []);}
 };
}

export function configureTaskSelection(select, config) {
 select.replaceChildren();
 for (const task of config.tasks || [{id:'empty', title:'Empty table'}]) {
  const option = select.ownerDocument.createElement('option');option.value = task.id;option.textContent = task.title;select.append(option);
 }
 select.value = config.defaults?.task ?? 'empty';
}

export function takeoverPrompt(side, decision) {
 if (!decision) return '';
 const label = side === 'left' ? 'Left' : 'Right';
 if (decision.state === 'following') return `${label} arm following`;
 if (decision.state === 'waiting') return `${label} arm waiting for input`;
 if (decision.state === 'hold') return `${label} arm HOLD${decision.reason ? ': '+decision.reason : ''}`;
 if (decision.state === 'matching') {
  const deviations = (decision.deviation_deg || []).map((value, index) => `J${index+1} ${value >= 0 ? '+' : ''}${Number(value).toFixed(1)}°`);
  return `Match ${label.toUpperCase()} ARM${deviations.length ? ': '+deviations.join(', ') : ''}`;
 }
 return `${label} arm ${decision.state}`;
}

export function takeoverView(root, {speak = () => {}} = {}) {
 let previous = {};
 return status => {
  const cues = [], nodes = [];
  for (const side of ['left', 'right']) {
   const decision = status.takeover?.[side], text = takeoverPrompt(side, decision);
   if (!text) continue;
   const node = root.ownerDocument.createElement('div');node.textContent = text;node.dataset.state = decision.state;node.dataset.side = side;nodes.push(node);
   const key = `${decision.state}:${decision.reason || ''}`;
   if (previous[side] !== key) cues.push(decision.state === 'matching' ? `Match ${side} arm` : text);
  }
  previous = Object.fromEntries(['left','right'].filter(side => status.takeover?.[side]).map(side => [side, `${status.takeover[side].state}:${status.takeover[side].reason || ''}`]));
  root.replaceChildren(...nodes);root.hidden = !nodes.length;
  if (cues.length) speak(cues.join('. '));
 };
}
