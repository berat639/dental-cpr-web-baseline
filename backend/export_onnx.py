"""Export the TotalSegmentator craniofacial mandible nnU-Net (fold_0) to ONNX (fixed patch size).

Usage: .venv/bin/python export_onnx.py
Output: models/mandible.onnx
"""
import json
from pathlib import Path

import torch
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

MODEL_DIR = next(
    (Path.home() / ".totalsegmentator/nnunet/results/Dataset115_mandible").glob("nnUNetTrainer_DASegOrd0_NoMirroring__nnUNetPlans__3d_fullres")
)
OUT = Path(__file__).parent / "models" / "mandible.onnx"
OUT.parent.mkdir(exist_ok=True)

plans = PlansManager(str(MODEL_DIR / "plans.json"))
dataset = json.load(open(MODEL_DIR / "dataset.json"))
cfg = plans.get_configuration("3d_fullres")
num_classes = len(dataset["labels"])

network = get_network_from_plans(
    cfg.network_arch_class_name,
    cfg.network_arch_init_kwargs,
    cfg.network_arch_init_kwargs_req_import,
    input_channels=1,
    output_channels=num_classes,
    allow_init=True,
    deep_supervision=False,
)
ckpt = torch.load(MODEL_DIR / "fold_0" / "checkpoint_final.pth", map_location="cpu", weights_only=False)
state = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in ckpt["network_weights"].items()}
print(network.load_state_dict(state))
network.eval()

patch = list(cfg.patch_size)
print("patch", patch, "classes", num_classes)
dummy = torch.randn(1, 1, *patch)
with torch.no_grad():
    torch.onnx.export(
        network, dummy, str(OUT), opset_version=17, dynamo=False,
        input_names=["input"], output_names=["logits"],
    )
print("saved", OUT, OUT.stat().st_size // 1_000_000, "MB")
