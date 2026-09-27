"""Kaggle-style inference: DICOM folders -> volumes -> frozen tokens -> MV-MoR 5-fold ensemble -> submission.csv

    python src/kneemor/infer.py --root <competition dir with test_series/ and test_series.csv> \
        --models results/models --key mvmor_s0 --out submission.csv
Uses exactly the training-time preprocessing (ingest.select_series / volume_from_dicoms, features.encode).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kneemor.features import PLANES, build_backbone, encode  # noqa: E402
from kneemor.ingest import select_series, volume_from_dicoms  # noqa: E402
from kneemor.models import build  # noqa: E402
from kneemor.report_labeler import LABELS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--models", default="results/models")
    ap.add_argument("--key", default="mvmor_s0")
    ap.add_argument("--weights", default="/home/user/data/r18a1.pth")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--int8", action="store_true",
                    help="INT8 post-training quantised backbone (~5x faster on CPU, no measured AUC loss; "
                         "calibrated on up to 8 test studies' images - unsupervised, no labels)")
    a = ap.parse_args()
    sdf = select_series(pd.read_csv(os.path.join(a.root, f"{a.split}_series.csv")))
    studies = pd.read_csv(os.path.join(a.root, f"{a.split}.csv")).StudyInstanceUID.tolist()
    bb = build_backbone(a.weights)
    vols = {}
    heads = []
    for f in sorted(glob.glob(os.path.join(a.models, f"{a.key}_f*.pt"))):
        h = build(a.key.rsplit("_s", 1)[0])
        h.load_state_dict(torch.load(f, map_location="cpu"))
        heads.append(h.eval())
    def load(st):
        if st in vols:
            return vols[st]
        vol = np.zeros((3, 24, 160, 160), np.uint8)
        mask = np.zeros(3, bool)
        for _, r in sdf[sdf.StudyInstanceUID == st].iterrows():
            files = glob.glob(os.path.join(a.root, f"{a.split}_series", st, r.SeriesInstanceUID, "*.dcm"))
            v, _ = volume_from_dicoms([open(f, "rb").read() for f in files], 24, 160)
            if v is not None:
                vol[PLANES.index(r.Anatomical_Plane)] = v
                mask[PLANES.index(r.Anatomical_Plane)] = True
        vols[st] = (vol, mask)
        return vol, mask

    if a.int8:
        import copy
        from torch.ao.quantization import get_default_qconfig_mapping
        from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
        from kneemor.features import MEAN, STD
        torch.backends.quantized.engine = "fbgemm"
        prep = prepare_fx(copy.deepcopy(bb), get_default_qconfig_mapping("fbgemm"), (torch.randn(8, 3, 160, 160),))
        with torch.no_grad():
            for st in studies[:8]:
                vol, _ = load(st)
                x = torch.from_numpy(vol.reshape(-1, 1, 160, 160)).float().div_(255).expand(-1, 3, -1, -1)
                prep((x - MEAN) / STD)
        bb = convert_fx(prep)
    rows = []
    for st in studies:
        vol, mask = load(st)
        vols.pop(st, None)
        if not mask.any():
            rows.append([st] + [0.5] * len(LABELS))
            continue
        x = torch.from_numpy(encode(bb, vol, 2).astype(np.float32))[None]
        m = torch.from_numpy(mask)[None]
        with torch.no_grad():
            p = np.mean([torch.sigmoid(h(x, m)).numpy()[0] for h in heads], 0)
        rows.append([st] + p.tolist())
    pd.DataFrame(rows, columns=["StudyInstanceUID"] + LABELS).to_csv(a.out, index=False)
    print(f"wrote {a.out} ({len(rows)} studies)")


if __name__ == "__main__":
    main()
