"""Apple-Silicon MPS inference for the original nnU-Net checkpoint.

ONNX Runtime's Core ML provider does not currently cover this model's 3D graph
reliably. On Apple Silicon we therefore run the same source checkpoint through
PyTorch/MPS, while keeping all interpolation and accumulation on the CPU.
"""
from __future__ import annotations

import json
import time
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from onnx_infer import PATCH, SPACING, SegmentationCancelled, _starts

HERE = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
_BUNDLED_ROOT = HERE / "models" / "nnunet"
_LEGACY_ROOT = Path.home() / ".totalsegmentator/nnunet/results/Dataset115_mandible"
ROOT = _BUNDLED_ROOT if _BUNDLED_ROOT.exists() else _LEGACY_ROOT
MODEL_DIR = next(ROOT.glob("nnUNetTrainer_DASegOrd0_NoMirroring__nnUNetPlans__3d_fullres"), None)
CHECKPOINT = MODEL_DIR / "fold_0" / "checkpoint_final.pth" if MODEL_DIR else Path()
_network: torch.nn.Module | None = None


def available() -> bool:
    return bool(MODEL_DIR and CHECKPOINT.exists() and torch.backends.mps.is_available())


def _model() -> torch.nn.Module:
    global _network
    if _network is None:
        if not available():
            raise RuntimeError("Apple MPS checkpoint is not available")
        plans = PlansManager(str(MODEL_DIR / "plans.json"))
        dataset = json.load(open(MODEL_DIR / "dataset.json"))
        configuration = plans.get_configuration("3d_fullres")
        network = get_network_from_plans(
            configuration.network_arch_class_name,
            configuration.network_arch_init_kwargs,
            configuration.network_arch_init_kwargs_req_import,
            input_channels=1,
            output_channels=len(dataset["labels"]),
            allow_init=True,
            deep_supervision=False,
        )
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        state = {key.replace("module.", "").replace("_orig_mod.", ""): value for key, value in checkpoint["network_weights"].items()}
        network.load_state_dict(state)
        _network = network.eval().to("mps")
    return _network


def _normalization() -> tuple[float, float, float, float]:
    assert MODEL_DIR
    props = json.load(open(MODEL_DIR / "plans.json"))["foreground_intensity_properties_per_channel"]["0"]
    return props["mean"], props["std"], props["percentile_00_5"], props["percentile_99_5"]


def segment_mandible(
    orig: sitk.Image,
    log=print,
    on_progress: Callable[[int, int, float], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> np.ndarray:
    if not available():
        raise RuntimeError("Apple MPS is unavailable")
    started = time.time()
    lps = sitk.DICOMOrient(orig, "LPS")
    size, spacing = lps.GetSize(), lps.GetSpacing()
    new_size = [max(1, int(round(size[i] * spacing[i] / SPACING))) for i in range(3)]
    fill = float(sitk.GetArrayViewFromImage(lps).min())
    iso = sitk.Resample(lps, new_size, sitk.Transform(), sitk.sitkLinear, lps.GetOrigin(), (SPACING,) * 3, lps.GetDirection(), fill, sitk.sitkFloat32)
    arr = np.ascontiguousarray(sitk.GetArrayFromImage(iso)[:, ::-1, ::-1])
    mean, std, low, high = _normalization()
    arr = (np.clip(arr, low, high) - mean) / max(std, 1e-8)
    pad = [max(0, PATCH[i] - arr.shape[i]) for i in range(3)]
    if any(pad): arr = np.pad(arr, [(0, value) for value in pad], constant_values=float(arr.min()))
    z_size, y_size, x_size = arr.shape
    gaussian = compute_gaussian(PATCH, sigma_scale=1 / 8, value_scaling_factor=10, dtype=torch.float32, device=torch.device("cpu")).numpy()
    acc, weights = np.zeros((z_size, y_size, x_size), np.float32), np.zeros((z_size, y_size, x_size), np.float32)
    zs, ys, xs = _starts(z_size, PATCH[0], 0.6), _starts(y_size, PATCH[1], 0.6), _starts(x_size, PATCH[2], 0.6)
    total, done, network = len(zs) * len(ys) * len(xs), 0, _model()
    with torch.inference_mode():
        for z in zs:
            for y in ys:
                for x in xs:
                    if is_cancelled and is_cancelled(): raise SegmentationCancelled()
                    patch = arr[z:z + PATCH[0], y:y + PATCH[1], x:x + PATCH[2]][None, None].astype(np.float32)
                    logits = network(torch.from_numpy(patch).to("mps"))
                    probability = torch.softmax(logits, dim=1)[0, 1].to("cpu").numpy()
                    acc[z:z + PATCH[0], y:y + PATCH[1], x:x + PATCH[2]] += probability * gaussian
                    weights[z:z + PATCH[0], y:y + PATCH[1], x:x + PATCH[2]] += gaussian
                    done += 1
                    log(f"MPS patch {done}/{total} ({time.time() - started:.0f}s)")
                    if on_progress: on_progress(done, total, time.time() - started)
    mask = (acc / np.maximum(weights, 1e-8) > 0.5)[:z_size - pad[0], :y_size - pad[1], :x_size - pad[2]].astype(np.uint8)
    mask = np.ascontiguousarray(mask[:, ::-1, ::-1])
    mask_image = sitk.GetImageFromArray(mask); mask_image.CopyInformation(iso)
    output = sitk.GetArrayFromImage(sitk.Resample(mask_image, orig, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)).astype(np.uint8)
    log(f"MPS mandible voxels {int(output.sum())} in {time.time() - started:.0f}s")
    return output
