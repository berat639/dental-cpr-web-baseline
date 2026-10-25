"""Export every DICOM series folder found under the given roots as raw int16 + json, for src/archEval.test.ts."""
import json, os, sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk

out = Path(os.environ.get("ARCH_EVAL_DIR", "/tmp/arch_eval")); out.mkdir(parents=True, exist_ok=True)
roots = [Path(p) for p in sys.argv[1:]]
folders = set()
for root in roots:
    for f in root.rglob("*.dcm"):
        folders.add(f.parent)
done = 0
for folder in sorted(folders):
    try:
        reader = sitk.ImageSeriesReader()
        ids = reader.GetGDCMSeriesIDs(str(folder))
        if not ids: continue
        best = max(ids, key=lambda s: len(reader.GetGDCMSeriesFileNames(str(folder), s)))
        names = reader.GetGDCMSeriesFileNames(str(folder), best)
        if len(names) < 60: continue
        reader.SetFileNames(names)
        img = reader.Execute()
        nx, ny, nz = img.GetSize(); sx, sy, sz = img.GetSpacing()
        if nz < 60 or max(sx, sy, sz) > 1.5: continue
        arr = sitk.GetArrayFromImage(sitk.Cast(img, sitk.sitkInt16))
        name = "__".join(folder.parts[-3:]).replace(" ", "_")
        (out / f"{name}.raw").write_bytes(np.ascontiguousarray(arr).tobytes())
        (out / f"{name}.json").write_text(json.dumps({"name": name, "path": str(folder), "n": [nx, ny, nz], "s": [sx, sy, sz],
            "min": int(arr.min()), "max": int(arr.max())}))
        print("exported", name, (nx, ny, nz), (round(sx,3), round(sy,3), round(sz,3)), int(arr.min()), int(arr.max()), flush=True)
        done += 1
    except Exception as e:
        print("skip", folder, e, flush=True)
print("total", done)
