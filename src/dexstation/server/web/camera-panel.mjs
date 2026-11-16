// Independent of simulation setup, overlays and WebCodecs availability.
export const cameraActions = Object.freeze({
  cameraStart: '/api/v1/camera/start',
  cameraStop: '/api/v1/camera/stop',
  cameraSnapshot: '/api/v1/camera/snapshot',
  cameraRecordStart: '/api/v1/camera/record/start',
  cameraRecordStop: '/api/v1/camera/record/stop',
});

// Optional services supply presentation; the built-in panel needs no device metadata.
let cameraDetails = s => `${s.width || 'Unknown'} × ${s.height || 'Unknown'}\nK: ${s.K ? JSON.stringify(s.K) : 'Waiting for camera_info'}`
  + (s.cameras || []).map(camera => `\n\n${camera.name} · ${camera.mode}`).join('');
let cameraError = s => s.error;
let cameraFrameError = () => 'Waiting for a fresh camera frame';
/* camera-service-presentation */

export function installCameraPanel(doc = document, fetcher = fetch) {
  const $ = id => doc.getElementById(id);
  let mode = 'idle', busy = false, imageURL = null, count = 0, begin = performance.now(), signature = '';
  let displayedFPS = 0;
  let deviceModes = ['idle'];
  let selectedCamera = null, cameraStatus = {}, available = true;
  const metrics = {frames: 0, fps: 0, age_s: null};
  const selector = doc.createElement('select'), selectorLabel = doc.createElement('label');
  selector.id = 'cameraPreviewTarget';
  selectorLabel.textContent = 'Preview camera ';
  selectorLabel.append(selector);
  $('cameraPanel').append(selectorLabel);
  const target = path => selectedCamera == null ? path
    : path + (path.includes('?') ? '&' : '?') + 'camera_name=' + encodeURIComponent(selectedCamera);
  metrics.selectCamera = async name => {
    selectedCamera = name || null;
    signature = '';
    await refresh(); await photos();
  };
  selector.onchange = () => metrics.selectCamera(selector.value).catch(e => { $('cameraMessage').textContent = e.message; });
  async function api(path, body) {
    const r = await fetcher(path, body === undefined ? {} : {
      method: 'POST', headers: {'Content-Type': 'application/json', 'X-Dexstation-Client': 'web'},
      body: JSON.stringify(body),
    });
    const value = await r.json();
    if (!r.ok) throw Error(value.message || value.error || r.statusText);
    return value;
  }
  function buttons() {
    $('cameraStart').disabled = !available || busy || deviceModes.some(value => value !== 'idle');
    $('cameraStop').disabled = !available || busy || deviceModes.every(value => value === 'idle');
    $('cameraSnapshot').disabled = !available || busy || mode !== 'preview';
    $('cameraRecordStart').disabled = !available || busy || !deviceModes.includes('preview') || deviceModes.some(value => !['idle', 'preview'].includes(value));
    $('cameraRecordStop').disabled = !available || busy || !deviceModes.includes('recording');
  }
  function report(value) {
    if (value) $('cameraReport').textContent = JSON.stringify(value, null, 2);
  }
  async function photos() {
    const list = await api(target('/api/v1/camera/photos'));
    const next = JSON.stringify(list);
    if (next === signature) return;
    signature = next;
    $('cameraPhotos').replaceChildren();
    for (const photo of list) {
      const figure = doc.createElement('figure'), img = doc.createElement('img'), caption = doc.createElement('figcaption');
      img.src = target('/api/v1/camera/photos/' + encodeURIComponent(photo.name + '.png')); img.alt = photo.label || 'Calibration photo'; img.loading = 'lazy';
      caption.textContent = `${photo.label} · ${photo.captured_at_utc}`;
      figure.append(img, caption);
      for (const ext of ['png', 'json']) {
        const a = doc.createElement('a'); a.href = target('/api/v1/camera/photos/' + encodeURIComponent(photo.name + '.' + ext));
        a.download = photo.name + '.' + ext; a.textContent = ` Download ${ext.toUpperCase()} `; figure.append(a);
      }
      $('cameraPhotos').append(figure);
    }
  }
  async function refresh() {
    const s = await api(target('/api/v1/camera/status'));
    selector.replaceChildren();
    const automatic = doc.createElement('option');
    automatic.value = ''; automatic.textContent = 'First started camera'; selector.append(automatic);
    for (const camera of s.cameras || []) {
      const option = doc.createElement('option');
      option.value = camera.name; option.textContent = `${camera.name} · ${camera.mode}`;
      selector.append(option);
    }
    selector.value = selectedCamera || '';
    mode = s.mode;
    cameraStatus = s;
    available = s.available !== false && mode !== 'unavailable';
    selector.disabled = !available;
    deviceModes = s.cameras ? s.cameras.map(camera => camera.mode) : [mode];
    $('cameraMode').textContent = ({idle:'Idle', preview:'Preview', recording:'Recording', starting:'Starting preview', starting_recording:'Starting recording', stopping_recording:'Saving recording', failed:'Camera failed', unavailable:'Unavailable'})[mode] || mode;
    if (s.error) $('cameraMessage').textContent = cameraError(s);
    metrics.age_s = s.age_s;
    $('cameraInfo').textContent = cameraDetails(s);
    $('cameraOutput').textContent = s.output_dir ? `Photo folder: ${s.output_dir}` : 'Start the camera to create a calibration session.';
    $('cameraMetrics').textContent = mode === 'preview'
      ? `Browser preview ${displayedFPS.toFixed(1)} fps · Camera ${Number(s.fps || 0).toFixed(1)} fps · Frame age ${s.age_s == null ? 'Unknown' : s.age_s.toFixed(3) + ' s'}`
      : 'Preview paused';
    const reports = Object.fromEntries((s.cameras || []).filter(camera => camera.recording_report != null)
      .map(camera => [camera.name, camera.recording_report]));
    report(Object.keys(reports).length ? reports : s.recording_report);
    buttons();
  }
  for (const [id, path] of Object.entries(cameraActions)) {
    $(id).onclick = async () => {
      busy = true; buttons(); $('cameraMessage').textContent = 'Working…';
      try {
        const value = await api(id === 'cameraSnapshot' ? target(path) : path, id === 'cameraSnapshot' ? {label: $('cameraLabel').value} : {});
        if (id === 'cameraSnapshot') {
          $('cameraLastPhoto').src = target('/api/v1/camera/photos/' + encodeURIComponent(value.name + '.png'));
          $('cameraLastPhoto').hidden = false;
        }
        report(value.cameras || value.recording_report);
        $('cameraMessage').textContent = id === 'cameraSnapshot' ? `Saved: ${value.name}` : 'Done';
        await refresh(); await photos();
      } catch (e) { $('cameraMessage').textContent = e.message; }
      finally { busy = false; buttons(); }
    };
  }
  $('cameraGrid').onchange = () => $('cameraGuide').classList.toggle('grid', $('cameraGrid').checked);
  $('cameraCross').onchange = () => $('cameraGuide').classList.toggle('cross', $('cameraCross').checked);
  async function preview() {
    const start = performance.now();
    if (mode === 'preview' && $('cameraPanel').open) {
      try {
        const r = await fetcher(target('/api/v1/camera/preview.jpg?t=' + Date.now()));
        if (!r.ok) throw Error(cameraFrameError(cameraStatus));
        const next = URL.createObjectURL(await r.blob());
        $('cameraPreview').src = next;
        if (imageURL) URL.revokeObjectURL(imageURL);
        imageURL = next; count++; metrics.frames++;
        const elapsed = (performance.now() - begin) / 1000;
        if (elapsed >= 1) {
          displayedFPS = count / elapsed; metrics.fps = displayedFPS;
          count = 0; begin = performance.now();
        }
      } catch (e) { $('cameraMessage').textContent = e.message; }
    } else { count = 0; begin = performance.now(); }
    setTimeout(preview, Math.max(0, 66 - (performance.now() - start)));
  }
  async function poll() {
    try { await refresh(); await photos(); }
    catch (e) { $('cameraMessage').textContent = e.message; }
    setTimeout(poll, 500);
  }
  buttons(); preview(); poll();
  return metrics;
}
if (typeof document !== 'undefined') window.cameraPanelMetrics = installCameraPanel();
