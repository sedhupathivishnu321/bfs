"""Shared split / window helpers (same logic as scripts/02 and 07) for analysis scripts that reload saved artifacts."""
from pathlib import Path
import numpy as np
from . import data as D

R = Path(__file__).resolve().parents[2]


def load_all():
    feats = {p.stem: D.load_recording(p)[0] for p in sorted((R / "data/processed/features").glob("*.npz"))}
    site = {c: c.split("_")[0] for c in feats}
    return feats, site, sorted(set(site.values()))


def windows(feats, codes, lo=0.0, hi=1.0, stride=4):
    parts = [D.make_windows(feats[c], int(lo * len(feats[c])), int(hi * len(feats[c])), stride) for c in codes]
    parts = [p for p in parts if p is not None]
    return tuple(np.concatenate(z) for z in zip(*parts))


def fold_data(feats, site, sites, test_site):
    others = [s for s in sites if s != test_site]; vs = others[sites.index(test_site) % len(others)]
    tr = windows(feats, [c for c in feats if site[c] not in (test_site, vs)])
    va = windows(feats, [c for c in feats if site[c] == vs]); te = windows(feats, [c for c in feats if site[c] == test_site])
    return tr, va, te
