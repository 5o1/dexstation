// Task-owned session options and read-only progress.
export function mount(root, api) {
 const document = root.ownerDocument;
 const control = document.createElement('span'), shadow = control.attachShadow({mode:'open'});
 const style = document.createElement('style');
 style.textContent = ':host{font:inherit}label{display:inline-flex;align-items:center;gap:3px;white-space:nowrap;margin-right:8px}select{font:inherit;padding:3px 6px;margin:0;border:1px solid var(--line);border-radius:2px;min-width:0;background:var(--bg);color:var(--text)}select:focus-visible{outline:1px solid var(--focus);outline-offset:1px}';
 const label = document.createElement('label');label.textContent = 'Task set';
 const select = document.createElement('select');select.id = 'autopilotTasks';
 select.setAttribute('aria-label', 'Task set');
 for (const path of api.config.autopilot_tasks || []) {
  const option = document.createElement('option');option.value = path;option.textContent = path.split('/').pop();select.append(option);
 }
 select.value = api.config.defaults?.autopilot_tasks || api.config.autopilot_tasks?.[0] || '';
 label.append(select);shadow.append(style,label);api.addSessionControl(control);
 api.sessionOptions(() => ({autopilot_tasks:select.value || null}));
 const status = document.createElement('pre');status.setAttribute('role','status');root.append(status);
 api.subscribe(() => {
  const progress = api.status().autopilot;
  status.textContent = progress ? JSON.stringify(progress, null, 2) : 'Waiting for autopilot';
 });
}
