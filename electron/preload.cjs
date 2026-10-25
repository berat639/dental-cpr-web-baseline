const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('pineclinic', {
  serviceStatus: () => ipcRenderer.invoke('service:status'),
  startService: () => ipcRenderer.invoke('service:start'),
  stopService: () => ipcRenderer.invoke('service:stop'),
  cancelJob: (jobId) => ipcRenderer.invoke('service:cancel-job', jobId),
  releaseDataset: (datasetId) => ipcRenderer.invoke('service:release-dataset', datasetId),
  onServiceUpdate: (callback) => {
    const listener = (_event, status) => callback(status);
    ipcRenderer.on('service:update', listener);
    return () => ipcRenderer.removeListener('service:update', listener);
  },
});
