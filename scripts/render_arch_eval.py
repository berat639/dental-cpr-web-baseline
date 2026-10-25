"""Overlay report.json arches on the axial slice used, one panel per scan -> /tmp/arch_eval/overlay.png"""
import json, os
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = Path(os.environ.get("ARCH_EVAL_DIR", "/tmp/arch_eval"))
rep = json.load(open(d / "report.json"))
cols = 3; rows = (len(rep) + cols - 1) // cols
fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
for ax, r in zip(np.ravel(axs), rep):
    nx, ny, nz = r["n"]; sx, sy, sz = r["s"]
    arr = np.frombuffer((d / f"{r['name']}.raw").read_bytes(), dtype=np.int16).reshape(nz, ny, nx)
    z = int(r.get("zc") or r["occlusalSlice"] or nz // 2)
    sl = arr[max(0, min(nz - 1, z))].astype(float)
    ax.imshow(sl, cmap="gray", vmin=-300, vmax=2200, origin="lower", extent=[0, nx * sx, 0, ny * sy])
    p = np.array(r["points"]) if r["points"] else np.zeros((0, 2))
    if len(p): ax.plot(p[:, 0], p[:, 1], "-", color="lime", lw=1.8)
    ax.set_title(f"{r['name'][-28:]}  {'OK' if r['accepted'] else 'REJECTED'} {r.get('fit')} sup={r.get('support', 0):.2f} gap={r.get('gapMm', 0):.1f} z={z}", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.savefig(d / "overlay.png", dpi=70)
print(d / "overlay.png")
