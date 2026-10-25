const { app, BrowserWindow, dialog, ipcMain } = require('electron');
const { spawn } = require('node:child_process');
const path = require('node:path');

let windowRef;
let service;
let status = { state: 'stopped', apiBase: '', logs: [], detail: '' };

function pushLog(line) {
  if (!line) return;
  status.logs = [...status.logs, line].slice(-500);
  publish();
}
function publish() { windowRef?.webContents.send('service:update', status); }
function setStatus(next) { status = { ...status, ...next }; publish(); }
function backendCommand(port) {
  if (app.isPackaged) {
    const binary = process.platform === 'win32' ? 'pineclinic-backend.exe' : 'pineclinic-backend';
    // PyInstaller's onedir build keeps the executable beside its private
    // runtime folder. Spawning the containing directory produces EACCES.
    const executable = path.join(process.resourcesPath, 'backend', 'pineclinic-backend', binary);
    return { command: executable, args: ['--host', '127.0.0.1', '--port', String(port)] };
  }
  const python = process.platform === 'win32' ? path.join(__dirname, '..', 'backend', '.venv', 'Scripts', 'python.exe') : path.join(__dirname, '..', 'backend', '.venv', 'bin', 'python');
  return { command: python, args: [path.join(__dirname, '..', 'backend', 'main.py'), '--host', '127.0.0.1', '--port', String(port)] };
}
async function startService() {
  if (service) return;
  // The clinical viewer and this manager share one documented local endpoint.
  // A random port makes a running service indistinguishable from an offline one.
  const port = 8000;
  const spec = backendCommand(port);
  setStatus({ state: 'starting', apiBase: `http://127.0.0.1:${port}`, detail: '', logs: [] });
  try {
    service = spawn(spec.command, spec.args, { windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] });
  } catch (error) {
    service = undefined; setStatus({ state: 'error', detail: String(error) }); return;
  }
  for (const stream of [service.stdout, service.stderr]) stream?.on('data', (data) => String(data).split(/\r?\n/).forEach(pushLog));
  service.once('error', (error) => { service = undefined; setStatus({ state: 'error', detail: error.message }); pushLog(error.stack || error.message); });
  service.once('exit', (code, signal) => { service = undefined; if (status.state !== 'stopped') setStatus({ state: 'error', detail: `Service exited (${signal || code})` }); });
  const base = status.apiBase;
  // Loading the embedded PyTorch/MPS libraries can take tens of seconds on a
  // cold launch. Keep the panel in "starting" instead of reporting a false
  // error while the local process is still becoming ready.
  for (let attempt = 0; attempt < 240; attempt += 1) {
    await new Promise((resolve) => setTimeout(resolve, 250));
    try {
      const reply = await fetch(`${base}/api/health`);
      if (reply.ok) { setStatus({ state: 'running', detail: '' }); pushLog('Service health check passed.'); return; }
    } catch (_) { /* wait while Uvicorn starts */ }
  }
  setStatus({ state: 'error', detail: 'Service did not pass its health check within 60 seconds. Review the log below.' });
}
function stopService() {
  if (!service) { setStatus({ state: 'stopped', apiBase: '', detail: '' }); return; }
  const child = service; service = undefined;
  child.kill(); setStatus({ state: 'stopped', apiBase: '', detail: '' }); pushLog('Service stopped by user.');
}
function createWindow() {
  windowRef = new BrowserWindow({ width: 720, height: 620, minWidth: 620, minHeight: 500, webPreferences: { preload: path.join(__dirname, 'preload.cjs'), contextIsolation: true, nodeIntegration: false } });
  windowRef.loadFile(path.join(__dirname, 'service-manager.html'));
}
app.whenReady().then(() => {
  ipcMain.handle('service:status', () => status);
  ipcMain.handle('service:start', startService);
  ipcMain.handle('service:stop', stopService);
  ipcMain.handle('dialog:dicom-folder', async () => {
    const result = await dialog.showOpenDialog(windowRef, { title: 'Choose DICOM series folder', properties: ['openDirectory'] });
    return result.canceled ? null : result.filePaths[0];
  });
  createWindow();
});
app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });
app.on('before-quit', stopService);
app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });
