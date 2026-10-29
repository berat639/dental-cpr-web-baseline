const { app, BrowserWindow, dialog, ipcMain } = require('electron');
const { spawn } = require('node:child_process');
const fs = require('node:fs');
const crypto = require('node:crypto');
const path = require('node:path');

app.name = 'PineClinic Lokal Servisler';

const PORT = 8000;
const BASE = `http://127.0.0.1:${PORT}`;
const API_VERSION = 3;

let windowRef;
let service;
let controlToken = crypto.randomBytes(24).toString('hex');
let eventAbort;
let stopping = false;
let status = { state: 'stopped', apiBase: '', logs: [], detail: '', viewers: 0, datasets: [], jobs: [], engine: '', provider: '', adopted: false };

// Fingerprint of the manager code (electron/* + package version), shown in the log to tie a test run to the code that produced it.
function managerCodeVersion() {
  const digest = crypto.createHash('sha1');
  try {
    for (const name of fs.readdirSync(__dirname).sort()) digest.update(name).update(fs.readFileSync(path.join(__dirname, name)));
  } catch (_) { digest.update('unreadable'); }
  return `${app.getVersion()}+${digest.digest('hex').slice(0, 8)}`;
}
let MANAGER_VERSION = 'unknown';

function pushLog(line) {
  if (!line) return;
  status.logs = [...status.logs, line].slice(-500);
  publish();
}
// The window can be closed or its renderer gone at any moment; sending to it must never throw.
function publish() {
  if (quitting || !windowRef || windowRef.isDestroyed()) return;
  const contents = windowRef.webContents;
  if (!contents || contents.isDestroyed()) return;
  try { contents.send('service:update', status); } catch (_) { /* window is closing */ }
}
process.on('uncaughtException', (error) => { try { pushLog(`Uncaught exception: ${error.stack || error}`); } catch (_) { /* nothing else to do */ } });
process.on('unhandledRejection', (reason) => { try { pushLog(`Unhandled rejection: ${reason && reason.stack || reason}`); } catch (_) { /* nothing else to do */ } });
function setStatus(next) { status = { ...status, ...next }; publish(); }
const clearLive = () => ({ viewers: 0, datasets: [], jobs: [], engine: '', provider: '' });

function backendCommand(port) {
  if (app.isPackaged) {
    const binary = process.platform === 'win32' ? 'pineclinic-local-services.exe' : 'pineclinic-local-services';
    // PyInstaller's onedir build keeps the executable beside its private
    // runtime folder. Spawning the containing directory produces EACCES.
    const executable = path.join(process.resourcesPath, 'backend', 'pineclinic-local-services', binary);
    return { command: executable, args: ['--host', '127.0.0.1', '--port', String(port)] };
  }
  const python = process.platform === 'win32' ? path.join(__dirname, '..', 'backend', '.venv', 'Scripts', 'python.exe') : path.join(__dirname, '..', 'backend', '.venv', 'bin', 'python');
  return { command: python, args: [path.join(__dirname, '..', 'backend', 'main.py'), '--host', '127.0.0.1', '--port', String(port)] };
}

async function probe() {
  try {
    const reply = await fetch(`${BASE}/api/health`, { signal: AbortSignal.timeout(1500) });
    if (!reply.ok) return { kind: 'foreign' };
    const body = await reply.json();
    return body && body.instanceId ? { kind: 'pineclinic', body } : { kind: 'foreign' };
  } catch (_) { return { kind: 'free' }; }
}

// --- live state from the service (single source of truth) -------------------------------------
function applyEvent(event, data) {
  switch (event) {
    case 'hello':
      setStatus({ viewers: data.clients?.viewers ?? 0, datasets: data.datasets, jobs: data.jobs, engine: data.engine, provider: data.provider });
      break;
    case 'clients': setStatus({ viewers: data.viewers }); break;
    case 'dataset.opened': setStatus({ datasets: [...status.datasets.filter((d) => d.datasetId !== data.datasetId), data] }); break;
    case 'dataset.released': setStatus({ datasets: status.datasets.filter((d) => d.datasetId !== data.datasetId), jobs: status.jobs.filter((j) => j.datasetId !== data.datasetId) }); break;
    case 'job.queued': case 'job.running': case 'job.progress': case 'job.done': case 'job.failed': case 'job.cancelled':
      setStatus({ jobs: [...status.jobs.filter((j) => j.jobId !== data.jobId), data] });
      if (event === 'job.done' || event === 'job.failed' || event === 'job.cancelled') pushLog(`AI job ${data.jobId.slice(0, 8)} ${event.slice(4)}${data.detail ? `: ${data.detail}` : ''}`);
      break;
    case 'pick-folder.request': pickFolder(data.requestId); break;
    case 'service.stopping': stopping = true; setStatus({ state: 'stopping' }); break;
    default: break;
  }
}

async function pickFolder(requestId) {
  let chosen = null;
  try {
    // The dialog must appear in front of the browser the clinician is using.
    app.focus({ steal: true });
    if (!windowRef) createWindow();
    const result = await dialog.showOpenDialog(windowRef, { title: 'DICOM seri klasörünü seçin', properties: ['openDirectory'] });
    chosen = result.canceled ? null : result.filePaths[0];
  } catch (error) { pushLog(`Folder dialog failed: ${error.message}`); }
  try {
    await fetch(`${BASE}/api/control/pick-folder/${requestId}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Control-Token': controlToken }, body: JSON.stringify({ path: chosen }),
    });
  } catch (error) { pushLog(`Could not deliver folder choice: ${error.message}`); }
}

async function watchEvents() {
  eventAbort?.abort();
  const abort = eventAbort = new AbortController();
  try {
    const response = await fetch(`${BASE}/api/events?role=manager`, { headers: { 'X-Control-Token': controlToken }, signal: abort.signal });
    if (!response.ok || !response.body) throw new Error(`events stream refused (${response.status})`);
    const decoder = new TextDecoder();
    let buffer = '';
    for await (const chunk of response.body) {
      buffer += decoder.decode(chunk, { stream: true });
      let split;
      while ((split = buffer.indexOf('\n\n')) >= 0) {
        const frame = buffer.slice(0, split); buffer = buffer.slice(split + 2);
        const event = /^event: (.+)$/m.exec(frame)?.[1], data = /^data: (.+)$/m.exec(frame)?.[1];
        if (event && data && event !== 'heartbeat') applyEvent(event, JSON.parse(data));
      }
    }
  } catch (error) {
    if (abort.signal.aborted) return;
    pushLog(`Event stream ended: ${error.message}`);
  }
  // Stream closed without our asking: a stop is expected, anything else is a crash.
  if (!abort.signal.aborted && eventAbort === abort && status.state === 'running') {
    setStatus({ state: 'error', detail: 'Lost connection to the service', ...clearLive() });
  }
}

// --- lifecycle --------------------------------------------------------------------------------
async function startService() {
  if (service || status.state === 'starting' || status.state === 'running') return;
  const existing = await probe();
  if (existing.kind === 'foreign') {
    setStatus({ state: 'error', detail: `Port ${PORT} is used by another application. Close it and start the service again.` });
    return;
  }
  if (existing.kind === 'pineclinic') {
    // A previous PineClinic service survived (crashed manager, CLI start). Its control
    // token is unknown to us, so it cannot be managed; ask the user to stop it first.
    setStatus({ state: 'error', detail: 'A PineClinic service is already running on this port (started elsewhere). Stop that process first, then press Start.' });
    return;
  }
  stopping = false;
  controlToken = crypto.randomBytes(24).toString('hex');
  const spec = backendCommand(PORT);
  setStatus({ state: 'starting', apiBase: BASE, detail: '', logs: [], ...clearLive() });
  pushLog(`Manager code ${MANAGER_VERSION}; starting service.`);
  try {
    service = spawn(spec.command, spec.args, {
      windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env, PINECLINIC_CONTROL_TOKEN: controlToken, PINECLINIC_MANAGER_PID: String(process.pid) },
    });
  } catch (error) {
    service = undefined; setStatus({ state: 'error', detail: String(error) }); return;
  }
  const child = service;
  for (const stream of [child.stdout, child.stderr]) stream?.on('data', (data) => String(data).split(/\r?\n/).forEach(pushLog));
  child.once('error', (error) => { if (service === child) service = undefined; setStatus({ state: 'error', detail: error.message, ...clearLive() }); pushLog(error.stack || error.message); });
  child.once('exit', (code, signal) => {
    if (service === child) service = undefined;
    eventAbort?.abort();
    if (status.state === 'stopped' || stopping) { setStatus({ state: 'stopped', apiBase: '', detail: '', ...clearLive() }); return; }
    setStatus({ state: 'error', detail: `Service exited (${signal || code})`, ...clearLive() });
  });
  // Loading the embedded PyTorch/MPS libraries can take tens of seconds on a
  // cold launch. Keep the panel in "starting" instead of reporting a false
  // error while the local process is still becoming ready.
  for (let attempt = 0; attempt < 240; attempt += 1) {
    await new Promise((resolve) => setTimeout(resolve, 250));
    if (service !== child) return;
    const result = await probe();
    if (result.kind === 'pineclinic') {
      if (result.body.apiVersion !== API_VERSION) { setStatus({ state: 'error', detail: `Service API v${result.body.apiVersion} does not match manager v${API_VERSION}.` }); return; }
      setStatus({ state: 'running', detail: '', engine: result.body.engine, provider: result.body.provider });
      pushLog(`Service health check passed. Backend code ${result.body.codeVersion || 'unknown'} (api v${result.body.apiVersion}); manager code ${MANAGER_VERSION}.`);
      void watchEvents();
      return;
    }
  }
  setStatus({ state: 'error', detail: 'Service did not pass its health check within 60 seconds. Review the log below.' });
}

function activeWork() {
  const running = status.jobs.filter((j) => j.state === 'queued' || j.state === 'running').length;
  return { running, viewers: status.viewers, datasets: status.datasets.length };
}

/** Returns false if the user declined. `force` skips the confirmation (app quit, crash cleanup). */
async function stopService({ force = false } = {}) {
  const child = service;
  if (!child) { setStatus({ state: 'stopped', apiBase: '', detail: '', ...clearLive() }); return true; }
  const work = activeWork();
  if (!force && (work.running || work.viewers)) {
    const parts = [];
    if (work.running) parts.push(`${work.running} AI job running`);
    if (work.viewers) parts.push(`${work.viewers} open viewer${work.viewers > 1 ? 's' : ''}`);
    const { response } = await dialog.showMessageBox(windowRef, {
      type: 'warning', buttons: ['Stop service', 'Cancel'], defaultId: 1, cancelId: 1,
      message: 'Stop the local service?', detail: `${parts.join(' and ')}. The viewer will be told the service stopped and the open case will have to be reloaded.`,
    });
    if (response !== 0) return false;
  }
  stopping = true;
  setStatus({ state: 'stopping' });
  pushLog('Stopping service…');
  const exited = new Promise((resolve) => child.once('exit', resolve));
  // 1) polite: the service tells every viewer first, cancels the AI job, then exits.
  fetch(`${BASE}/api/control/shutdown`, { method: 'POST', headers: { 'X-Control-Token': controlToken }, signal: AbortSignal.timeout(2000) }).catch(() => undefined);
  const outcome = await Promise.race([exited.then(() => 'exited'), new Promise((r) => setTimeout(() => r('timeout'), 3000))]);
  if (outcome === 'timeout') {
    // 2) SIGTERM, 3) SIGKILL: the AI worker may be inside a long native call.
    try { child.kill(); } catch (_) { /* already gone */ }
    const second = await Promise.race([exited.then(() => 'exited'), new Promise((r) => setTimeout(() => r('timeout'), 3000))]);
    if (second === 'timeout') { try { child.kill('SIGKILL'); } catch (_) { /* already gone */ } await Promise.race([exited, new Promise((r) => setTimeout(r, 2000))]); }
  }
  service = undefined;
  eventAbort?.abort();
  stopping = false;
  setStatus({ state: 'stopped', apiBase: '', detail: '', ...clearLive() });
  pushLog('Service stopped by user.');
  return true;
}

function createWindow() {
  windowRef = new BrowserWindow({ width: 760, height: 760, minWidth: 620, minHeight: 560, webPreferences: { preload: path.join(__dirname, 'preload.cjs'), contextIsolation: true, nodeIntegration: false } });
  windowRef.on('closed', () => { windowRef = undefined; });
  windowRef.webContents.on('render-process-gone', (_event, details) => pushLog(`Manager window renderer gone: ${details.reason}`));
  windowRef.loadFile(path.join(__dirname, 'service-manager.html'));
}

// pineclinic://start (from the web viewer) opens this app and starts the local service.
const PROTOCOL = 'pineclinic';
if (process.defaultApp && process.argv.length >= 2) app.setAsDefaultProtocolClient(PROTOCOL, process.execPath, [path.resolve(process.argv[1])]);
else app.setAsDefaultProtocolClient(PROTOCOL);

let pendingLaunch = process.argv.some((arg) => arg.startsWith(`${PROTOCOL}://`));
function handleLaunchUrl(url) {
  if (!String(url).startsWith(`${PROTOCOL}://`)) return;
  pushLog(`Launch request from viewer: ${url}`);
  if (!status || status.state === 'stopped' || status.state === 'error') startService().catch((error) => pushLog(`Auto-start failed: ${error.message}`));
}

if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on('second-instance', (_event, argv) => {
    const url = argv.find((arg) => arg.startsWith(`${PROTOCOL}://`));
    if (url) handleLaunchUrl(url);
  });
  app.on('open-url', (event, url) => { event.preventDefault(); if (app.isReady()) handleLaunchUrl(url); else pendingLaunch = true; });
}

app.whenReady().then(() => {
  ipcMain.handle('service:status', () => status);
  ipcMain.handle('service:start', startService);
  ipcMain.handle('service:stop', () => stopService());
  ipcMain.handle('service:cancel-job', async (_event, jobId) => {
    try { await fetch(`${BASE}/api/jobs/${encodeURIComponent(jobId)}/cancel`, { method: 'POST', signal: AbortSignal.timeout(2000) }); } catch (_) { /* viewer gets the real state from the stream */ }
  });
  ipcMain.handle('service:release-dataset', async (_event, datasetId) => {
    try { await fetch(`${BASE}/api/datasets/${encodeURIComponent(datasetId)}`, { method: 'DELETE', signal: AbortSignal.timeout(2000) }); } catch (_) { /* ignored */ }
  });
  MANAGER_VERSION = managerCodeVersion();
  pushLog(`PineClinic Lokal Servisler yönetici kodu ${MANAGER_VERSION}.`);
  createWindow();
  if (pendingLaunch) handleLaunchUrl(`${PROTOCOL}://start`);
});
app.on('window-all-closed', () => { app.quit(); });
let quitting = false;
app.on('before-quit', (event) => {
  if (quitting || !service) return;
  // Wait for the service to exit instead of orphaning it on port 8000.
  event.preventDefault();
  quitting = true;
  stopService({ force: true }).finally(() => app.quit());
});
app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });
