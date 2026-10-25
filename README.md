# PineClinic local CBCT prototype

The web interface talks only to a local service on `127.0.0.1:8000`. DICOM
data and the optional 3D model do not leave the current computer.

Run the local service in one terminal:

```bash
cd backend
REQUESTS_CA_BUNDLE=.certs.pem SSL_CERT_FILE=.certs.pem .venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000
```

Run the interface in another terminal:

```bash
npm run dev
```

Open the displayed Vite address, then either paste a local DICOM folder path or
select the DICOM files. The initial spline and panorama do not wait for AI.
`Refine with Local AI` starts a cancellable background segmentation job; it is
only applied if the active dataset and spline have not changed while it ran.

## Acceleration

The panorama uses WebGL2 on the local browser GPU. The 3D model is kept in the
backend and never sent to the browser. On Apple Silicon the service selects the
local original checkpoint through PyTorch MPS; set `TS_ENGINE=onnx` to compare
it with the exported ONNX CPU path. On Windows, ONNX Runtime selects CUDA or
DirectML when an installed provider exposes one, otherwise it uses CPU. Core ML
is deliberately not used for this 3D graph because its ONNX operator support is
incomplete.
