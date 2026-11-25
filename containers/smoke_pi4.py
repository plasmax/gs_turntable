"""Import checks for the Pi4 image. Needs no network or models. FlexGEMM and Pi4 need an NVIDIA driver to import,
so without a GPU (GitHub runners) only their installation is checked; the full import runs where a GPU is visible.
"""
import importlib.metadata as md
import importlib.util
import sys
import cv2, numpy as np, PIL, safetensors, scipy, torch, triton

assert torch.version.cuda == "12.6", torch.version.cuda
for name in ("flex_gemm", "pi4"):
    assert importlib.util.find_spec(name), f"{name} is not installed"
assert md.version("triton") == "3.7.1"
if torch.cuda.is_available():
    from pi4 import Pi4
    from pi4.io import load_images
    assert hasattr(Pi4, "from_pretrained")
    print("full Pi4 import OK on", torch.cuda.get_device_name())
else:
    print("no GPU: Pi4 import not exercised")
print("pi4 image OK: torch", torch.__version__, "triton", triton.__version__, "python", sys.version.split()[0])
