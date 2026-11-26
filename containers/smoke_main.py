"""Import and CPU checks for the main image. Needs no GPU, network or models."""
import ctypes, glob, os, sys, tempfile
import cv2, imageio, kornia, lpips, numpy as np, plyfile, pycolmap, pymatting, scipy, timm, torch, torchvision, transformers
import gsplat

assert torch.version.cuda == "12.8", torch.version.cuda
archs = torch._C._cuda_getArchFlags().split()  # compiled-in kernels; get_arch_list() is empty without a GPU
for arch in ("sm_86", "sm_120"):  # A4000/A6000, RTX Pro 6000
    assert arch in archs, (arch, archs)
from gsplat import csrc  # the prebuilt CUDA extension must load; otherwise gsplat would try to compile one at runtime
import loma.loma as L
assert L.LoMaB and L.filter_matches
assert hasattr(cv2, "SIFT_create")
assert imageio.plugins.ffmpeg.get_exe() if hasattr(imageio.plugins, "ffmpeg") else True
import imageio_ffmpeg
assert os.path.exists(imageio_ffmpeg.get_ffmpeg_exe())
with tempfile.TemporaryDirectory() as d:
    db = pycolmap.Database(f"{d}/t.db")
    cam = pycolmap.Camera(model="SIMPLE_PINHOLE", width=64, height=64, params=[50, 32, 32])
    cam.camera_id = 1
    cam.has_prior_focal_length = True
    db.write_camera(cam, use_camera_id=True)
    db.close()
    r = pycolmap.Rigid3d()
a = np.zeros((32, 32), np.float64)
a[8:24, 8:24] = 1
pymatting.estimate_foreground_ml(np.random.rand(32, 32, 3), a)  # exercises numba
if torch.cuda.is_available():
    from gsplat import rasterization
    out, al, _ = rasterization(torch.zeros(1, 3).cuda(), torch.tensor([[1., 0, 0, 0]]).cuda(), torch.full((1, 3), .1).cuda(),
                               torch.ones(1).cuda(), torch.ones(1, 3).cuda(), torch.tensor(np.diag([1., 1, 1, 1]) + np.eye(4, k=3) * 0).float().cuda()[None] + torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2.], [0, 0, 0, 0]]).cuda(),
                               torch.tensor([[50., 0, 16], [0, 50, 16], [0, 0, 1]]).cuda()[None], 32, 32, packed=False)
    assert al.max() > .5
    print("GPU rasterization OK on", torch.cuda.get_device_name())
print("main image OK: torch", torch.__version__, "gsplat", gsplat.__version__, "pycolmap", pycolmap.__version__, "python", sys.version.split()[0])
