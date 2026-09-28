"""Leakage-free streaming ingest of the RSNA Knee Abnormality Detection archive.

The Kaggle bundle is a 265 GB (compressed) zip of ~820k DICOM slices. That is
far larger than the compute node's disk, so we never materialise the archive:

1. The zip central directory is read once (HTTP range requests) and cached as
   ``index_full.pkl`` = [(name, header_offset, compress_size, file_size, method)].
2. Every series occupies a *contiguous* byte span in the archive (verified), so
   one ranged GET per series fetches all of its slices.
3. Slices are inflated in memory, decoded with pydicom, ordered along the
   slice normal, resampled to a fixed depth D and in-plane size S, intensity
   normalised per series (0.5/99.5 percentiles) and stored as uint8.

Output: ``<out>/<StudyInstanceUID>.npz`` with one (D,S,S) array per kept series
plus ``<out>/_series_meta.csv`` (per-series DICOM descriptors for QA).
"""
from __future__ import annotations

import argparse
import io
import os
import pickle
import struct
import sys
import time
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
import pydicom
import requests

LOCAL_HDR = struct.Struct("<IHHHHHIIIHH")  # zip local file header (30 bytes)


def select_series(series_df: pd.DataFrame, per_plane: int = 1) -> pd.DataFrame:
    """Pick at most `per_plane` series per anatomical plane per study.

    Fluid-sensitive (PD/T2 FS) sequences are preferred because 10 of the 12
    targets (tears, effusion, contusion, synovitis, cysts) are conspicuous on
    them. Ties are broken deterministically by SeriesInstanceUID. This choice
    depends only on DICOM-derived metadata available at test time (no labels).

    Data audit note (docs/data_audit.md): across all 24,371 series, Fluid_Sensitive
    == Fat_Suppression with zero exceptions in this cohort, so the two terms below
    are not independent evidence -- this is closer to weighting one signal 3x than
    combining two. Series-per-plane duplication is also common (roughly a third of
    study-plane pairs have >1 candidate series), so this tie-break does real work;
    it has not been shown to be the best available criterion, only a leakage-safe one.
    """
    df = series_df.copy()
    df["_rank"] = -df["Fluid_Sensitive"] * 2 - df["Fat_Suppression"]
    df = df.sort_values(["StudyInstanceUID", "Anatomical_Plane", "_rank", "SeriesInstanceUID"])
    return df.groupby(["StudyInstanceUID", "Anatomical_Plane"]).head(per_plane).drop(columns="_rank")


def fetch_span(url: str, start: int, end: int, session: requests.Session, retries: int = 5) -> bytes:
    for k in range(retries):
        try:
            r = session.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
            if r.status_code == 206 and len(r.content) == end - start + 1:
                return r.content
        except requests.RequestException:
            pass
        time.sleep(2 ** k)
    raise RuntimeError(f"range fetch failed {start}-{end}")


def inflate_members(buf: bytes, base: int, members) -> list[bytes]:
    out = []
    for _, off, csize, _, method in members:
        p = off - base
        sig, *_rest, nlen, xlen = LOCAL_HDR.unpack_from(buf, p)
        assert sig == 0x04034B50, "bad local header"
        d0 = p + 30 + nlen + xlen
        raw = buf[d0:d0 + csize]
        out.append(zlib.decompress(raw, -15) if method == 8 else raw)
    return out


def slice_position(ds) -> float:
    try:
        iop = np.array(ds.ImageOrientationPatient, dtype=float)
        ipp = np.array(ds.ImagePositionPatient, dtype=float)
        return float(np.dot(np.cross(iop[:3], iop[3:]), ipp))
    except Exception:
        return float(getattr(ds, "InstanceNumber", 0))


def pick_slice_indices(n: int, depth: int):
    """Choose `depth` slice indices out of `n` available, in [0, n).

    Replaces a plain ``np.linspace(0, n-1, depth).round()``, which is fine when
    n >> depth but can round two evenly-spaced target positions to the same
    integer index near the ends of a short series (docs/data_audit.md finding
    3: series range 11-320 slices, median 30, against a fixed depth of 16-24).
    * n >= depth: evenly spaced and **guaranteed unique** (any rounding
      collision is repaired by inserting the midpoint of the largest gap).
    * n < depth: every available slice is kept, then padded by repeating the
      *last* slice -- an explicit, symmetric policy, in place of whichever
      slices happened to fall on a rounding collision before.
    """
    if n >= depth:
        idx = np.unique(np.round(np.linspace(0, n - 1, depth)).astype(int))
        while len(idx) < depth:
            gaps = np.diff(idx)
            j = int(np.argmax(gaps))
            idx = np.unique(np.insert(idx, j + 1, (idx[j] + idx[j + 1]) // 2))
        return idx[:depth]
    return np.concatenate([np.arange(n), np.full(depth - n, n - 1, dtype=int)])


def volume_from_dicoms(blobs: list[bytes], depth: int, size: int):
    # pass 1: headers only (cheap) -> geometry; pass 2: decode just the `depth` selected slices
    slices = []
    for b in blobs:
        try:
            ds = pydicom.dcmread(io.BytesIO(b), force=True, stop_before_pixels=True)
            shp = (int(ds.Rows), int(ds.Columns))
        except Exception:
            continue
        slices.append((slice_position(ds), shp, ds, b))
    if not slices:
        return None, {}
    # keep the dominant in-plane shape (drops localisers / odd reconstructions)
    shp = pd.Series([s[1] for s in slices]).value_counts().index[0]
    slices = sorted([s for s in slices if s[1] == shp], key=lambda s: s[0])
    n = len(slices)
    idx = pick_slice_indices(n, depth)
    imgs = []
    for i in idx:
        try:
            img = pydicom.dcmread(io.BytesIO(slices[i][3]), force=True).pixel_array.astype(np.float32)
            if img.ndim != 2:
                raise ValueError("non-2D pixel data")
        except Exception:
            img = np.zeros(shp, np.float32)
        imgs.append(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA))
    vol = np.stack(imgs)
    lo, hi = np.percentile(vol, [0.5, 99.5])
    vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)
    ds0 = slices[n // 2][2]
    info = dict(n_slices=n, rows=shp[0], cols=shp[1],
                pixel_spacing=str(getattr(ds0, "PixelSpacing", "")),
                slice_thickness=getattr(ds0, "SliceThickness", np.nan),
                manufacturer=str(getattr(ds0, "Manufacturer", "")),
                field_strength=getattr(ds0, "MagneticFieldStrength", np.nan),
                series_desc=str(getattr(ds0, "SeriesDescription", "")))
    return (vol * 255).round().astype(np.uint8), info


def process_study(url, split, study, series_rows, members_by_series, out_dir, depth, size, session):
    path = os.path.join(out_dir, f"{study}.npz")
    if os.path.exists(path):
        return study, []
    arrays, metas = {}, []
    for _, row in series_rows.iterrows():
        sid = row.SeriesInstanceUID
        mem = sorted(members_by_series.get(sid, []), key=lambda m: m[1])
        if not mem:
            continue
        start = mem[0][1]
        end = mem[-1][1] + 30 + len(mem[-1][0].encode()) + 256 + mem[-1][2]
        buf = fetch_span(url, start, end, session)
        vol, info = volume_from_dicoms(inflate_members(buf, start, mem), depth, size)
        if vol is None:
            continue
        arrays[f"{row.Anatomical_Plane}|{sid}"] = vol
        metas.append(dict(StudyInstanceUID=study, SeriesInstanceUID=sid, split=split,
                          Anatomical_Plane=row.Anatomical_Plane,
                          Fluid_Sensitive=row.Fluid_Sensitive, Fat_Suppression=row.Fat_Suppression,
                          **info))
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    return study, metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/user/data")
    ap.add_argument("--out", default="/home/user/data/vols")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--per-plane", type=int, default=1)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()

    url = open(os.path.join(a.data, "url.txt")).read().strip()
    index = pickle.load(open(os.path.join(a.data, "index_full.pkl"), "rb"))
    members = defaultdict(list)
    for m in index:
        p = m[0].split("/")
        if len(p) == 4 and p[0] == f"{a.split}_series":
            members[p[2]].append(m)
    sdf = select_series(pd.read_csv(os.path.join(a.data, "meta", f"{a.split}_series.csv")), a.per_plane)
    studies = sorted(sdf.StudyInstanceUID.unique())
    studies = studies[a.shard::a.nshards]
    if a.limit:
        studies = studies[:a.limit]
    out_dir = os.path.join(a.out, a.split)
    os.makedirs(out_dir, exist_ok=True)
    groups = {s: g for s, g in sdf.groupby("StudyInstanceUID")}

    sess = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=a.workers, pool_maxsize=a.workers)
    sess.mount("https://", adapter)
    meta_path = os.path.join(out_dir, "_series_meta.csv")
    t0, done, all_meta = time.time(), 0, []
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(process_study, url, a.split, s, groups[s], members, out_dir, a.depth, a.size, sess)
                for s in studies]
        for f in as_completed(futs):
            try:
                _, metas = f.result()
                all_meta.extend(metas)
            except Exception as e:  # keep going; failures are reported and re-run is idempotent
                print("ERROR", repr(e), file=sys.stderr, flush=True)
            done += 1
            if done % 50 == 0 or done == len(studies):
                print(f"[{a.split}] {done}/{len(studies)} studies  {time.time() - t0:.0f}s", flush=True)
                pd.DataFrame(all_meta).to_csv(meta_path + f".part{a.shard}", index=False)
    pd.DataFrame(all_meta).to_csv(meta_path + f".part{a.shard}", index=False)


if __name__ == "__main__":
    main()
