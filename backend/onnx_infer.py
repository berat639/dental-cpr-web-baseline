"""Fast mandible segmentation with the exported nnU-Net (local ONNX Runtime).

Replicates the nnU-Net inference path for TotalSegmentator `craniofacial_structures` (Dataset115):
  RAS orientation, 0.5 mm isotropic, CT normalisation, sliding window (112x160x128), softmax.
Only the mandible probability is accumulated (1 channel) to keep memory small.
"""
import json
import os
import time
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import onnxruntime as ort
import SimpleITK as sitk
import torch
from nnunetv2.inference.sliding_window_prediction import compute_gaussian

HERE = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
MODEL = HERE / "models" / "mandible.onnx"
_BUNDLED_PLANS = HERE / "models" / "nnunet" / "nnUNetTrainer_DASegOrd0_NoMirroring__nnUNetPlans__3d_fullres" / "plans.json"
_LEGACY_PLANS = Path.home() / ".totalsegmentator/nnunet/results/Dataset115_mandible/nnUNetTrainer_DASegOrd0_NoMirroring__nnUNetPlans__3d_fullres/plans.json"
PLANS = _BUNDLED_PLANS if _BUNDLED_PLANS.exists() else _LEGACY_PLANS
MANDIBLE_CH = 1
PATCH = (112, 160, 128)  # (z, y, x) in RAS-array order, from plans.json
SPACING = 0.5
TILE_STEP = float(__import__("os").environ.get("TS_TILE_STEP", "0.6"))

_session: ort.InferenceSession | None = None
_provider: str | None = None


class SegmentationCancelled(Exception):
    """Raised between tiles when the local UI cancels an AI job."""


def available() -> bool:
    return MODEL.exists() and PLANS.exists()


def mps_available() -> bool:
    """Expose MPS capability without falsely claiming ONNX Core ML runs this 3D net."""
    return bool(torch.backends.mps.is_built() and torch.backends.mps.is_available())


def _providers() -> list[str]:
    available_providers = ort.get_available_providers()
    preferred = os.environ.get("PINECLINIC_ORT_PROVIDER", "").strip()
    candidates = [preferred] if preferred else ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
    return [provider for provider in candidates if provider in available_providers] or ["CPUExecutionProvider"]


def provider_name() -> str:
    if _provider:
        return _provider
    return _providers()[0]


def _sess() -> ort.InferenceSession:
    global _session, _provider
    if _session is None:
        providers = _providers()
        _session = ort.InferenceSession(str(MODEL), providers=providers)
        _provider = _session.get_providers()[0]
    return _session


def _norm_props():
    p = json.load(open(PLANS))["foreground_intensity_properties_per_channel"]["0"]
    return p["mean"], p["std"], p["percentile_00_5"], p["percentile_99_5"]


def _starts(size: int, patch: int, step: float) -> list[int]:
    if size <= patch:
        return [0]
    n = int(np.ceil((size - patch) / (patch * step))) + 1
    return sorted({int(round(i * (size - patch) / (n - 1))) for i in range(n)}) if n > 1 else [0]


def segment_mandible(
    orig: sitk.Image,
    log=print,
    on_progress: Callable[[int, int, float], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Returns uint8 mask [z,y,x] on the ORIGINAL grid of `orig`."""
    t0 = time.time()
    lps = sitk.DICOMOrient(orig, "LPS")
    size = lps.GetSize()
    sp = lps.GetSpacing()
    new_size = [max(1, int(round(size[i] * sp[i] / SPACING))) for i in range(3)]
    fill = float(sitk.GetArrayViewFromImage(lps).min())
    iso = sitk.Resample(lps, new_size, sitk.Transform(), sitk.sitkLinear, lps.GetOrigin(),
                        (SPACING,) * 3, lps.GetDirection(), fill, sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(iso)  # [z, y, x]  (LPS)
    arr = np.ascontiguousarray(arr[:, ::-1, ::-1])  # -> RAS ([S, A, R] index order), as nnU-Net saw it
    log(f"0.5mm volume {arr.shape}, range {arr.min():.0f}..{arr.max():.0f}")

    mean, std, lo, hi = _norm_props()
    arr = (np.clip(arr, lo, hi) - mean) / max(std, 1e-8)

    # pad to at least patch size
    pad = [max(0, PATCH[i] - arr.shape[i]) for i in range(3)]
    if any(pad):
        arr = np.pad(arr, [(0, p) for p in pad], constant_values=float(arr.min()))
    Z, Y, X = arr.shape

    gauss = compute_gaussian(PATCH, sigma_scale=1.0 / 8, value_scaling_factor=10, dtype=torch.float32, device=torch.device("cpu")).numpy()
    acc = np.zeros((Z, Y, X), np.float32)
    wsum = np.zeros((Z, Y, X), np.float32)
    s = _sess()
    zs, ys, xs = _starts(Z, PATCH[0], TILE_STEP), _starts(Y, PATCH[1], TILE_STEP), _starts(X, PATCH[2], TILE_STEP)
    total = len(zs) * len(ys) * len(xs)
    n = 0
    for z in zs:
        for y in ys:
            for x in xs:
                if is_cancelled and is_cancelled():
                    raise SegmentationCancelled()
                patch = arr[z:z + PATCH[0], y:y + PATCH[1], x:x + PATCH[2]][None, None].astype(np.float32)
                logits = s.run(None, {"input": patch})[0][0]
                m = logits - logits.max(0, keepdims=True)
                e = np.exp(m)
                prob = e[MANDIBLE_CH] / e.sum(0)
                acc[z:z + PATCH[0], y:y + PATCH[1], x:x + PATCH[2]] += prob * gauss
                wsum[z:z + PATCH[0], y:y + PATCH[1], x:x + PATCH[2]] += gauss
                n += 1
                log(f"patch {n}/{total}  ({time.time() - t0:.0f}s)")
                if on_progress:
                    on_progress(n, total, time.time() - t0)
    prob = acc / np.maximum(wsum, 1e-8)
    mask = (prob > 0.5)[: Z - pad[0], : Y - pad[1], : X - pad[2]].astype(np.uint8)

    # back to LPS array, then onto the original grid
    mask = np.ascontiguousarray(mask[:, ::-1, ::-1])
    m_img = sitk.GetImageFromArray(mask)
    m_img.CopyInformation(iso)
    full = sitk.Resample(m_img, orig, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    out = sitk.GetArrayFromImage(full).astype(np.uint8)
    log(f"mandible voxels {int(out.sum())} in {time.time() - t0:.0f}s")
    return out
