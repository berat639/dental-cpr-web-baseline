"""Build a self-contained local backend executable for the current OS.

Run this on macOS for the Apple Silicon binary and on Windows for the .exe.
The executable carries its Python runtime, DICOM reader, ONNX Runtime and model;
end users do not install Python, Node, or packages.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
RELEASE = ROOT / "release" / "backend"

def main() -> None:
    if not (BACKEND / "models" / "mandible.onnx").exists():
        raise SystemExit("Missing backend/models/mandible.onnx")
    if not (BACKEND / "models" / "nnunet").exists():
        raise SystemExit("Missing bundled MPS model assets in backend/models/nnunet")
    RELEASE.mkdir(parents=True, exist_ok=True)
    command = [
        # A one-file build extracts hundreds of MB of ML runtime on every
        # launch. `onedir` remains fully self-contained inside the desktop
        # app but starts the local service without that multi-minute delay.
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir",
        "--name", "pineclinic-local-services", "--distpath", str(RELEASE), "--workpath", str(ROOT / "build" / "pyinstaller"),
        "--specpath", str(ROOT / "build"),
        "--add-data", f"{BACKEND / 'models'}{';' if sys.platform == 'win32' else ':'}models",
        "--collect-all", "SimpleITK", "--collect-all", "onnxruntime", "--collect-all", "nnunetv2",
        "--paths", str(BACKEND), "--hidden-import", "sync",
        "--hidden-import", "uvicorn.logging", "--hidden-import", "uvicorn.loops.auto", str(BACKEND / "main.py"),
    ]
    subprocess.run(command, check=True, cwd=ROOT)

if __name__ == "__main__": main()
