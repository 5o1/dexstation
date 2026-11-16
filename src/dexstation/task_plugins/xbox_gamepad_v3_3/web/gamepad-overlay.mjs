// The vendored renderer has window globals; isolate it inside this cell's frame.
export function mount(root, api) {
 root.style.cssText = 'height:100%;padding:0;display:flex;align-items:center;justify-content:center';
 const frame = root.ownerDocument.createElement('iframe');
 frame.title = 'Gamepad signals';frame.src = new URL('gamepad-overlay.html?source=canonical&theme=research', import.meta.url).href;
 frame.style.cssText = 'display:block;border:0;width:100%;height:100%';root.append(frame);
 const update = task => {
  const channels = task?.channels || {}, values = channels.gamepad_signal;
  if (values) frame.contentWindow?.postMessage({type:'task-signals',values:Array.from(values),mask:channels.gamepad_mask},new URL(frame.src).origin);
 };
 frame.onload = () => update(api.status().task);
 api.subscribe(update);
}
