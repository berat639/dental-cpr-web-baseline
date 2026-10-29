"""DentalSegmentator inference for local CBCT processing.

The model provides distinct upper- and lower-tooth labels.  Unlike the legacy
mandible-only model, its output can be used as direct arch evidence instead of
guessing teeth from vendor-dependent CBCT intensities.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from onnx_infer import SegmentationCancelled, _starts

HERE = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
ROOT = HERE / "models" / "dental-segmentator"
MODEL_DIR = next(ROOT.glob("**/nnUNetTrainer__nnUNetPlans__3d_fullres"), None)
CHECKPOINT = MODEL_DIR / "fold_0" / "checkpoint_final.pth" if MODEL_DIR else Path()
CONFIG = "3d_fullres"
TILE_STEP = float(os.environ.get("DENTAL_TILE_STEP", "0.65"))
_network: torch.nn.Module | None = None
_use_fp16: bool = False


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def provider_name() -> str:
    target = device().type
    return {"mps": "PyTorch MPS", "cuda": "PyTorch CUDA", "cpu": "PyTorch CPU"}[target]


def available() -> bool:
    return bool(MODEL_DIR and CHECKPOINT.exists())


def _model() -> tuple[torch.nn.Module, tuple[int, int, int], tuple[float, float, float], tuple[int, int, int]]:
    """Load the released nnU-Net v2.2 checkpoint with current nnU-Net support."""
    global _network, _use_fp16
    assert MODEL_DIR
    plans = PlansManager(str(MODEL_DIR / "plans.json"))
    configuration = plans.get_configuration(CONFIG)
    raw = configuration.configuration
    architecture = raw["architecture"]
    patch = tuple(int(v) for v in raw["patch_size"])
    spacing = tuple(float(v) for v in raw["spacing"])
    transpose = tuple(int(v) for v in json.load(open(MODEL_DIR / "plans.json"))["transpose_forward"])
    if _network is None:
        target = device()
        dataset = json.load(open(MODEL_DIR / "dataset.json"))
        net = get_network_from_plans(
            architecture["network_class_name"], architecture["arch_kwargs"], architecture["_kw_requires_import"],
            input_channels=1, output_channels=len(dataset["labels"]), allow_init=True, deep_supervision=False,
        ).eval()
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        state = {key.replace("module.", "").replace("_orig_mod.", ""): value for key, value in checkpoint["network_weights"].items()}
        net.load_state_dict(state)
        if target.type in {"mps", "cuda"}:
            # Run 3D convolutions in FP16 while keeping ConvTranspose3d in FP32 (required on Apple MPS).
            net = net.half()
            for module in net.modules():
                if isinstance(module, nn.ConvTranspose3d):
                    module.float()
                    module.register_forward_pre_hook(lambda mod, inp: (inp[0].float(),))
                    module.register_forward_hook(lambda mod, inp, out: out.half())
            _use_fp16 = True
        else:
            _use_fp16 = False
        _network = net.to(target)
    return _network, patch, spacing, transpose


def _normalization() -> tuple[float, float, float, float]:
    assert MODEL_DIR
    props = json.load(open(MODEL_DIR / "plans.json"))["foreground_intensity_properties_per_channel"]["0"]
    return props["mean"], props["std"], props["percentile_00_5"], props["percentile_99_5"]


def _resample_for_model(image: sitk.Image, spacing_after_transpose: tuple[float, float, float], transpose: tuple[int, int, int]) -> tuple[sitk.Image, np.ndarray]:
    """Return an RAS image and the model-axis input array.

    nnU-Net stores spacing after its transpose.  Convert it back to the
    SimpleITK x/y/z ordering before resampling, then apply the same transpose
    to the z/y/x array passed into the network.
    """
    ras = sitk.DICOMOrient(image, "RAS")
    inverse = [transpose.index(axis) for axis in range(3)]
    array_spacing = [spacing_after_transpose[inverse[axis]] for axis in range(3)]  # z, y, x
    sitk_spacing = tuple(reversed(array_spacing))
    size, old_spacing = ras.GetSize(), ras.GetSpacing()
    new_size = [max(1, int(round(size[i] * old_spacing[i] / sitk_spacing[i]))) for i in range(3)]
    fill = float(sitk.GetArrayViewFromImage(ras).min())
    iso = sitk.Resample(ras, new_size, sitk.Transform(), sitk.sitkLinear, ras.GetOrigin(), sitk_spacing, ras.GetDirection(), fill, sitk.sitkFloat32)
    array = np.ascontiguousarray(np.transpose(sitk.GetArrayFromImage(iso), transpose))
    return iso, array


def segment_dental(
    orig: sitk.Image,
    log=print,
    on_progress: Callable[[int, int, float], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Return DentalSegmentator labels 0..5 on the original image grid."""
    if not available():
        raise RuntimeError("DentalSegmentator model is not available")
    started = time.time()
    network, patch, spacing, transpose = _model()
    model_image, array = _resample_for_model(orig, spacing, transpose)
    mean, std, low, high = _normalization()
    dtype_np = np.float16 if _use_fp16 else np.float32
    dtype_torch = torch.float16 if _use_fp16 else torch.float32
    min_norm = float((low - mean) / max(std, 1e-8))
    array = ((np.clip(array, low, high) - mean) / max(std, 1e-8)).astype(dtype_np)
    padding = [max(0, patch[axis] - array.shape[axis]) for axis in range(3)]
    if any(padding):
        array = np.pad(array, [(0, value) for value in padding], constant_values=dtype_np(min_norm))
    shape = array.shape
    target = device()
    # Clamp Gaussian tail to 1e-2 before casting to FP16 so tile corners never underflow to zero.
    gaussian = compute_gaussian(patch, sigma_scale=1 / 8, value_scaling_factor=10, dtype=torch.float32, device=torch.device("cpu"))
    gaussian_dev = torch.clamp(gaussian, min=1e-2).to(device=target, dtype=dtype_torch)
    # Accumulate weighted class probabilities directly on the accelerator without per-tile 98 MB CPU copies.
    # Dividing all class scores by the same positive scalar weight per voxel does not change argmax.
    scores = torch.zeros((6, *shape), dtype=dtype_torch, device=target)
    array_dev = torch.from_numpy(array).to(target)
    zs, ys, xs = (_starts(shape[axis], patch[axis], TILE_STEP) for axis in range(3))
    total = len(zs) * len(ys) * len(xs)
    done = 0
    air_cutoff = min_norm + 1e-3
    with torch.inference_mode():
        for z in zs:
            for y in ys:
                for x in xs:
                    if is_cancelled and is_cancelled():
                        raise SegmentationCancelled()
                    tile_np = array[z:z + patch[0], y:y + patch[1], x:x + patch[2]]
                    if float(tile_np.max()) <= air_cutoff:
                        scores[0, z:z + patch[0], y:y + patch[1], x:x + patch[2]] += gaussian_dev
                    else:
                        tile = array_dev[z:z + patch[0], y:y + patch[1], x:x + patch[2]][None, None]
                        probability = torch.softmax(network(tile).float(), dim=1)[0].to(dtype_torch)
                        scores[:, z:z + patch[0], y:y + patch[1], x:x + patch[2]] += probability * gaussian_dev
                    done += 1
                    log(f"DentalSegmentator tile {done}/{total} ({time.time() - started:.0f}s)")
                    if on_progress:
                        on_progress(done, total, time.time() - started)
        labels = torch.argmax(scores, dim=0).to(torch.uint8).cpu().numpy()
    del scores, array_dev, gaussian_dev
    if target.type == "mps":
        torch.mps.empty_cache()
    cropped = labels[:shape[0] - padding[0], :shape[1] - padding[1], :shape[2] - padding[2]]
    # Undo nnU-Net's array transpose before returning to SimpleITK geometry.
    inverse = np.argsort(np.asarray(transpose))
    output_image = sitk.GetImageFromArray(np.ascontiguousarray(np.transpose(cropped, inverse)))
    output_image.CopyInformation(model_image)
    output = sitk.GetArrayFromImage(sitk.Resample(output_image, orig, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)).astype(np.uint8)
    log(f"DentalSegmentator labels {np.bincount(output.ravel(), minlength=6).tolist()} in {time.time() - started:.0f}s")
    return output
