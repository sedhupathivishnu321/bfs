"""Frozen-backbone slice tokenisation.

Every kept series (Sagittal / Coronal / Axial, D=24 slices, 160x160) is passed
slice-by-slice through an ImageNet-pretrained ResNet-18 (timm RSB-A1 weights).
The final feature map (5x5x512) is average-pooled to a GxG grid (default 2x2),
giving D*G*G tokens per plane. Tokens are stored as float16:

    feats  : [N, P=3, D, G*G, 512]   float16
    mask   : [N, P]                  bool (plane present)

The backbone is frozen, so tokens are computed once and shared by all heads
(proposed model, baselines and ablations) -> identical inputs, fair comparison.
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import timm
import torch
import torch.nn.functional as F

PLANES = ["Sagittal", "Coronal", "Axial"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def build_backbone(weights: str) -> torch.nn.Module:
    m = timm.create_model("resnet18", pretrained=False, num_classes=0, global_pool="")
    sd = torch.load(weights, map_location="cpu")
    m.load_state_dict({k: v for k, v in sd.items() if not k.startswith("fc.")})
    return m.eval()


def load_study(path: str, depth: int, size: int):
    z = np.load(path)
    vol = np.zeros((len(PLANES), depth, size, size), np.uint8)
    mask = np.zeros(len(PLANES), bool)
    for key in z.files:
        plane = key.split("|")[0]
        if plane in PLANES:
            vol[PLANES.index(plane)] = z[key]
            mask[PLANES.index(plane)] = True
    return vol, mask


@torch.no_grad()
def encode(model, vol_u8: np.ndarray, grid: int, bs: int = 96) -> np.ndarray:
    """vol_u8: [P, D, H, W] uint8 -> [P, D, grid*grid, C] float16"""
    P, D, H, W = vol_u8.shape
    x = torch.from_numpy(vol_u8.reshape(P * D, 1, H, W)).float().div_(255).expand(-1, 3, -1, -1)
    x = (x - MEAN) / STD
    outs = []
    for i in range(0, x.shape[0], bs):
        f = model(x[i:i + bs])                                   # [b, 512, h, w]
        f = F.adaptive_avg_pool2d(f, grid).flatten(2).transpose(1, 2)  # [b, g*g, 512]
        outs.append(f)
    f = torch.cat(outs).reshape(P, D, grid * grid, -1)
    return f.half().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vols", default="/home/user/data/vols")
    ap.add_argument("--split", default="train")
    ap.add_argument("--weights", default="/home/user/data/r18a1.pth")
    ap.add_argument("--out", default="/home/user/data/feats")
    ap.add_argument("--grid", type=int, default=2)
    ap.add_argument("--depth", type=int, default=24)
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    files = sorted(glob.glob(os.path.join(a.vols, a.split, "*.npz")))
    ids = [os.path.basename(f)[:-4] for f in files]
    os.makedirs(a.out, exist_ok=True)
    model = build_backbone(a.weights)
    N = len(files)
    feats = np.lib.format.open_memmap(os.path.join(a.out, f"{a.split}_feats.npy"), "w+", np.float16,
                                      (N, len(PLANES), a.depth, a.grid * a.grid, 512))
    masks = np.zeros((N, len(PLANES)), bool)
    t0 = time.time()
    for i, f in enumerate(files):
        vol, m = load_study(f, a.depth, a.size)
        feats[i] = encode(model, vol, a.grid)
        masks[i] = m
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{N} {time.time() - t0:.0f}s", flush=True)
    feats.flush()
    np.save(os.path.join(a.out, f"{a.split}_mask.npy"), masks)
    with open(os.path.join(a.out, f"{a.split}_ids.txt"), "w") as fh:
        fh.write("\n".join(ids))
    print("done", N, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
