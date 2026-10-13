"""Writes normalized NCHW float32 det inputs (det_HxW.f32) from a document image: BGR, /255, ImageNet mean/std."""
import sys
import numpy as np
from PIL import Image

src, out_dir = sys.argv[1], sys.argv[2]
im = Image.open(src).convert('RGB')
for lim in (960, 1600):
    w, h = im.size
    s = lim / max(w, h)
    W, H = max(32, round(w * s / 32) * 32), max(32, round(h * s / 32) * 32)
    a = np.asarray(im.resize((W, H), Image.BILINEAR), dtype=np.float32)[:, :, ::-1] / 255.0
    a = (a - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    path = f'{out_dir}/det_{H}x{W}.f32'
    a.transpose(2, 0, 1).astype(np.float32).tofile(path)
    print(path)
