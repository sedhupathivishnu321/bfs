"""Builds notebooks/rsna_knee_kaggle_v2.ipynb, a self-contained Kaggle notebook (GPU).

    python notebooks/build_kaggle_notebook_v2.py

v2 removes the fixed limits of rsna_knee_kaggle.ipynb (one series per plane, 16 slices, 224 px, a single small
backbone, a fixed epoch/fold plan) and adds the features of the public Raptor / ConvNeXt-MIL notebooks that
the analysis in docs/notebook_analysis.md found useful. The original builder and notebook are kept unchanged
because the README numbers were produced with them.
"""
import pathlib

import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parents[1]
LABELER_SRC = (ROOT / "src/kneemor/report_labeler.py").read_text()

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
C = []

C.append(md(r"""# RSNA Knee Abnormality Detection v2: slot-based 2.5-D MIL with per-finding attention

Self-contained Kaggle notebook: turn on a **GPU** (T4×2 or P100) and **Run All**.

## What changed against v1 (and where each idea comes from)

| Limit in v1 | v2 | Source of the idea |
|---|---|---|
| 1 series per plane, 16 slices, 224 px | **5 slots** (sag fluid, sag non-fluid, cor fluid, cor non-fluid, axial), 96 slices, 256–384 px, windows of 3 adjacent slices | Raptor CoAtNet notebooks |
| Resize by pixel count | **Physical crop** (`CROP_MM` mm around the centre, from `PixelSpacing`), series ordered along the slice normal, 2–98 % span, per-series percentile window | Raptor / ConvNeXt-MIL |
| Mean over slices per plane | **Per-finding attention over all windows** + slot and slice-position embeddings + a 2-layer transformer, plus v1's global branch and deep supervision | Raptor, ConvNeXt-MIL, v1 |
| One small backbone (EfficientNet-B0) | any timm backbone; **several arms** can be trained and fused | Velociraptor |
| No test-time augmentation | **Anatomical mirror TTA** (sagittal: reverse the 3 slice channels; coronal/axial: flip width) | Raptor |
| Plain weights | **EMA** of the weights (the role SWA plays in Raptor) | Raptor |
| Fixed folds/epochs | **Presets** and a **wall-clock budget**: training stops cleanly after the last fold that fits, and the submission uses the folds finished | Dinosaur v5 deadline guards |
| Fusion weights picked on the 58 expert studies | Fusion weights fitted on **out-of-fold** predictions (shrunk toward equal); the expert set is used only for reporting | fixes a weakness seen in the public notebooks |
| Fails silently | per-study fallbacks (missing series, unreadable DICOM, no series at all -> label prevalence), checkpoint SHA-256 in `config.json` | Raptor / Blend-Gold |

Kept from v1: the multilingual report labeler (+ optional local LLM labeler), leakage-safe protocol (the 58
expert-labelled studies are never trained on unless you set `INCLUDE_GOLD=True`), soft-label BCE, bootstrap CI,
leave-one-out thresholds for accuracy.

## Honest status
* The code was smoke-tested on **synthetic** DICOMs on CPU (shapes, masks, training step, TTA, fusion,
  submission format). **No accuracy has been measured for v2.** Whatever AUC/accuracy it reaches on the real data is
  printed by the notebook; nothing here promises a number.
* Run-time and memory estimates in the presets are estimates, not measurements.
* Label quality still caps the result: in the author's study the rule labels reach ≈0.735 AUC against the expert
  set. The LLM labeler (`USE_LLM_LABELS=True`) is the largest expected lever; the better of rules / LLM /
  average is chosen with the 58 expert studies (three candidates, so a small selection effect; the 95 % CI of a
  58-study AUC is about ±0.06).
* The public notebooks' ≈0.95 scores come from checkpoints trained on LLM labels plus extra data and tuned on
  the 58 expert studies and the public leaderboard; this notebook does not reproduce them.

**Modes:** `MODE="train"` (internet ON for pretrained weights) trains, evaluates and writes weights + a submission.
`MODE="infer"` (code-competition, internet OFF): set `WEIGHTS_DIR` to a Kaggle dataset holding `config.json` and the
`*_fold*.pt` files."""))

C.append(code(r"""# ============================== CONFIG ==============================
import os, glob, json, math, time, random, warnings, hashlib, shutil
warnings.filterwarnings("ignore")

class CFG:
    MODE = "train"                 # "train" (train+eval+submit) | "infer" (load WEIGHTS_DIR, submit only)
    PRESET = "balanced"            # "fast" | "balanced" | "max" | "custom" (keep the values below)
    SEED = 42
    TIME_BUDGET_H = 8.5            # stop starting new folds when the next one would not finish inside this budget
    INFER_RESERVE_MIN = 25         # time kept for test inference
    # ---- data ----
    DATA_DIR = None                # None -> auto-detect under /kaggle/input
    CACHE_DIR = "/kaggle/temp/cache" if os.path.isdir("/kaggle/temp") else "/kaggle/working/cache"
    OUT_DIR = "/kaggle/working"
    WEIGHTS_DIR = "/kaggle/input/knee-v2-weights"       # used when MODE == "infer"
    AUTO_FIT_DISK = True           # lower IMG automatically if the cache would not fit on disk
    # ---- input geometry (physical) ----
    # (plane, fluid_sensitive preference 1/0/-1=any, slices)  ->  96 slices, as in the Raptor notebooks
    SLOTS = [("Sagittal", 1, 26), ("Sagittal", 0, 22), ("Coronal", 1, 18), ("Coronal", 0, 12), ("Axial", -1, 18)]
    SPAN = (0.02, 0.98)            # fraction of each stack that is sampled (wide: collaterals / lateral meniscus)
    CROP_MM = 140.0                # square field of view around the image centre
    IMG = 256                      # cached in-plane size
    PCT = (2.0, 98.0)              # per-series intensity window (percentiles of the sampled slices)
    ALLOW_SLOT_REUSE = True        # a slot with no unused series of its plane reuses that plane's best series
    # ---- model ----
    BACKBONES = ["convnext_tiny.in12k_ft_in1k"]  # >1 entry = several arms fused on OOF; e.g. add "tf_efficientnet_b3.ns_jft_in1k"
    PRETRAINED = True              # needs internet in MODE="train"
    D_MODEL = 384
    TF_DEPTH = 2
    DROP = 0.2
    DROP_PATH = 0.1
    GRAD_CKPT = False              # gradient checkpointing in the encoder (saves memory, ~30 % slower)
    # ---- training ----
    FOLDS = 5
    TRAIN_FOLDS = [0, 1, 2, 3, 4]
    EPOCHS = 6
    BATCH = 2                      # studies per step
    TRAIN_WINDOWS = 32             # windows (3-slice stacks) sampled per study per step
    EVAL_STRIDE = 1                # 1 = every slice is a window centre at inference
    LR_BACKBONE = 1e-4
    LR_HEAD = 1e-3
    WD = 0.02
    AUX_W = 0.5                    # deep supervision of the global and the local branch
    EMA = 0.998                    # 0 = off; EMA weights are evaluated and saved
    CLIP = 2.0
    MIRROR_AUG = False             # also train on anatomically mirrored studies (Raptor uses mirror only at test time)
    MIRROR_TTA = True
    NUM_WORKERS = 4
    AMP = True
    INCLUDE_GOLD = False           # True: add the 58 expert studies to training (final-fit mode; gold metrics are then invalid)
    # ---- labels ----
    USE_LLM_LABELS = False         # True -> attach a Qwen2.5-Instruct model and set LLM_PATH
    LLM_PATH = "/kaggle/input/qwen2.5/transformers/3b-instruct/1"
    LLM_BATCH = 8
    # ---- output ----
    OUTPUT = "prob"                # "prob" | "rank" (per-column percentile rank of the ensemble)
    SMOKE = False
    MAX_TRAIN_STUDIES = None       # e.g. 300 for a quick trial

PRESETS = {   # ESTIMATES for a Kaggle T4x2, not measurements
    "fast":     dict(IMG=224, TRAIN_WINDOWS=24, EPOCHS=4, BACKBONES=["convnext_nano.in12k_ft_in1k"], D_MODEL=256),
    "balanced": dict(),
    "max":      dict(IMG=320, TRAIN_WINDOWS=48, EPOCHS=8, GRAD_CKPT=True,
                     BACKBONES=["convnext_tiny.in12k_ft_in1k", "tf_efficientnet_b3.ns_jft_in1k"]),
}
for k, v in PRESETS.get(CFG.PRESET, {}).items(): setattr(CFG, k, v)

if os.environ.get("KNEE_SMOKE") == "1":   # used only by the author's CPU smoke test on synthetic data
    CFG.SMOKE, CFG.DATA_DIR = True, os.environ["KNEE_DATA_DIR"]
    CFG.CACHE_DIR = CFG.OUT_DIR = os.environ["KNEE_OUT_DIR"]
    CFG.BACKBONES, CFG.PRETRAINED, CFG.IMG, CFG.CROP_MM, CFG.D_MODEL = ["resnet18", "resnet10t"], False, 64, 100.0, 64
    CFG.SLOTS = [("Sagittal", 1, 5), ("Sagittal", 0, 4), ("Coronal", 1, 4), ("Coronal", 0, 3), ("Axial", -1, 4)]
    CFG.FOLDS, CFG.TRAIN_FOLDS, CFG.EPOCHS, CFG.BATCH, CFG.TRAIN_WINDOWS = 2, [0, 1], 1, 2, 8
    CFG.NUM_WORKERS, CFG.AMP, CFG.EMA, CFG.TF_DEPTH, CFG.AUTO_FIT_DISK = 0, False, 0.9, 1, False
    CFG.MODE = os.environ.get("KNEE_MODE", "train"); CFG.WEIGHTS_DIR = os.environ.get("KNEE_WEIGHTS", CFG.OUT_DIR)
CFG.TOTAL = sum(s[2] for s in CFG.SLOTS)
os.makedirs(CFG.CACHE_DIR, exist_ok=True); os.makedirs(CFG.OUT_DIR, exist_ok=True)
print({k: v for k, v in vars(CFG).items() if not k.startswith("_")})"""))

C.append(code(r"""import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import cv2, pydicom, timm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(CFG.SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPU = torch.cuda.device_count()
print("device:", DEVICE, "| GPUs:", N_GPU, "| torch", torch.__version__, "| timm", timm.__version__)

def find_data_dir():
    if CFG.DATA_DIR: return CFG.DATA_DIR
    for p in sorted(glob.glob("/kaggle/input/*")) + sorted(glob.glob("/kaggle/input/*/*")) + sorted(glob.glob("/kaggle/input/*/*/*")):
        if os.path.exists(os.path.join(p, "test_series.csv")): return p
    raise FileNotFoundError("competition data not found under /kaggle/input")
DATA = find_data_dir(); print("data:", DATA)
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
T0 = time.time()
def elapsed(): return f"{(time.time() - T0) / 60:.1f} min"
def sha256(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(n): h.update(b)
    return h.hexdigest()"""))

C.append(md("## 1. Multilingual rule labeler (embedded, unchanged from v1)"))
C.append(code("%%writefile knee_labeler.py\n" + LABELER_SRC))

C.append(code(r"""import importlib, knee_labeler; importlib.reload(knee_labeler)
from knee_labeler import label_frame, evaluate as eval_labeler

def macro_auc(y, p, valid=None):
    a = []
    for k in range(y.shape[1]):
        v = np.ones(len(y), bool) if valid is None else valid[:, k]
        if 0 < y[v, k].sum() < v.sum(): a.append(roc_auc_score(y[v, k], p[v, k]))
    return float(np.mean(a)) if a else float("nan")

if CFG.MODE == "train":
    train = pd.read_csv(f"{DATA}/train.csv")
    gold_mask = train[LABELS].notna().all(axis=1).values
    print(f"studies: {len(train)} | expert-labelled: {gold_mask.sum()} "
          f"({'used for training' if CFG.INCLUDE_GOLD else 'held out'})")
    rule = label_frame(train).set_index("StudyInstanceUID").loc[train.StudyInstanceUID, LABELS].values
    ev = eval_labeler(train[gold_mask], pd.DataFrame(rule[gold_mask], columns=LABELS)
                      .assign(StudyInstanceUID=train.StudyInstanceUID[gold_mask].values))
    print(f"rule labeler vs expert labels: macro AUC {ev.auc.mean():.4f}")
    display(ev[["label", "n_pos", "auc", "sens", "spec"]].round(3))"""))

C.append(md(r"""## 2. Optional LLM labeler (the largest expected lever)

Set `USE_LLM_LABELS=True` and attach an instruction model (e.g. **Qwen2.5-3B/7B-Instruct** from Kaggle Models).
The model returns a probability per finding (hedged wording -> intermediate values, as in the public notebooks'
soft labels). Rules, LLM and their average are each scored on the 58 expert studies; the best one becomes the
training label. This is the only decision taken with the expert set."""))

C.append(code(r"""LLM_PROMPT = '''You are a musculoskeletal radiologist. Read the knee MRI report (any language) and rate each finding
as present in THIS knee on this exam. Answer ONLY with JSON, values between 0 and 1 (1 = definitely present,
0 = absent/not mentioned, 0.5 = equivocal, about 0.8 = suspected). Definitions:
ACL: ACL tear (partial or complete). MCL: MCL sprain/tear (any grade). Medial Meniscus / Lateral Meniscus: meniscal tear
(not degeneration alone). Medial OA / Lateral OA / PF OA: osteoarthritis of that compartment (cartilage loss,
osteophytes, degenerative joint disease). Effusion: joint effusion more than physiological. Synovitis: synovitis or
synovial thickening. Baker's: Baker/popliteal cyst. Contusion: bone contusion/bruise/traumatic marrow edema.
Fracture: any fracture.
Keys: "ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture".
REPORT:
'''

def llm_label(reports):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(CFG.LLM_PATH, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(CFG.LLM_PATH, torch_dtype=torch.float16, device_map="auto").eval()
    out = np.full((len(reports), len(LABELS)), np.nan, np.float32)
    for i in range(0, len(reports), CFG.LLM_BATCH):
        chunk = reports[i:i + CFG.LLM_BATCH]
        msgs = [tok.apply_chat_template([{"role": "user", "content": LLM_PROMPT + str(r)[:6000]}],
                                        tokenize=False, add_generation_prompt=True) for r in chunk]
        enc = tok(msgs, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=160, do_sample=False)
        for j, g in enumerate(gen):
            txt = tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            try:
                js = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
                out[i + j] = [float(js.get(k, np.nan)) for k in LABELS]
            except Exception:
                pass
        if i % (CFG.LLM_BATCH * 50) == 0: print(f"  LLM {i}/{len(reports)} {elapsed()}")
    del model; torch.cuda.empty_cache()
    return out

if CFG.MODE == "train":
    candidates = {"rules": rule}
    if CFG.USE_LLM_LABELS:
        llm = llm_label(train.Report.tolist())
        ok = ~np.isnan(llm).any(1); print(f"LLM parsed {ok.mean():.1%} of reports")
        llm = np.where(np.isnan(llm), rule, np.clip(llm, 0, 1))
        candidates["llm"] = llm
        candidates["llm+rules"] = (llm + rule) / 2
    Yg_all = train.loc[gold_mask, LABELS].values.astype(int)
    scores = {k: macro_auc(Yg_all, v[gold_mask]) for k, v in candidates.items()}
    print("labeler macro AUC vs expert labels:", {k: round(v, 4) for k, v in scores.items()})
    LABEL_SOURCE = max(scores, key=scores.get)
    SOFT = candidates[LABEL_SOURCE].astype(np.float32)
    print("-> training labels from:", LABEL_SOURCE)"""))

C.append(md(r"""## 3. DICOM → physically normalised slot volumes

Per study the cache holds `TOTAL` slices (96 by default) in fixed slots. For each slot:
1. choose a series of the slot's plane, preferring the slot's fluid-sensitivity (a series is used once; if none is
   left and `ALLOW_SLOT_REUSE`, the plane's best series is reused, so the slot is not empty);
2. order the files along the slice normal (`ImageOrientationPatient` × `ImagePositionPatient`);
3. take slices evenly from 2 % to 98 % of the stack;
4. apply the modality LUT (slope/intercept), invert MONOCHROME1, clip to the series' 2–98 percentiles;
5. centre-crop a `CROP_MM` square using `PixelSpacing` (a knee covers the same fraction of the frame at 0.3 and
   0.5 mm/px), resize to `IMG` with area interpolation; store uint8."""))
C.append(code(r"""def pick_series(rows, plane, fluid, used):
    cands = [r for r in rows if r["Anatomical_Plane"] == plane]
    new = [r for r in cands if r["SeriesInstanceUID"] not in used]
    if fluid in (0, 1):
        pref = [r for r in new if int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
        if pref: return pref[0], False
    if new: return new[0], False
    if CFG.ALLOW_SLOT_REUSE and cands:
        best = sorted(cands, key=lambda r: -(int(r.get("Fluid_Sensitive", 0) or 0) == fluid) if fluid in (0, 1) else 0)
        return best[0], True
    return None, False

def _hdr(f):
    try:
        ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
        o, p = ds.get("ImageOrientationPatient"), ds.get("ImagePositionPatient")
        if o is not None and p is not None and len(o) == 6:
            o = np.array(o, float); pos = float(np.dot(np.cross(o[:3], o[3:]), np.array(p, float)))
        else:
            pos = float(ds.get("InstanceNumber", 0) or 0)
        ps = ds.get("PixelSpacing"); ps = float(ps[0]) if ps is not None else 0.0
        return pos, (int(ds.Rows), int(ds.Columns)), ps, f
    except Exception:
        return None

def _read_px(f):
    ds = pydicom.dcmread(f, force=True)
    a = ds.pixel_array
    if a.ndim != 2: raise ValueError("multi-frame / colour")
    a = a.astype(np.float32)
    s, i = ds.get("RescaleSlope"), ds.get("RescaleIntercept")
    if s is not None and i is not None: a = a * float(s) + float(i)
    if str(ds.get("PhotometricInterpretation", "")) == "MONOCHROME1": a = a.max() - a
    return a

def mm_crop_resize(a, ps, size):
    h, w = a.shape
    if ps and ps > 0:
        c = int(round(CFG.CROP_MM / ps)); ch, cw = min(c, h), min(c, w)
    else:                                   # unknown spacing: assume the image spans about the field of view
        ch = cw = min(h, w)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    crop = np.ascontiguousarray(a[y0:y0 + ch, x0:x0 + cw])
    out = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return out

def series_slices(files, n, size, threads=8):
    with ThreadPoolExecutor(threads) as ex:
        hdr = [h for h in ex.map(_hdr, files) if h is not None]
    if not hdr: return None
    shp = pd.Series([h[1] for h in hdr]).value_counts().index[0]
    hdr = sorted([h for h in hdr if h[1] == shp], key=lambda h: h[0])
    ps_all = [h[2] for h in hdr if h[2] > 0]; ps = float(np.median(ps_all)) if ps_all else 0.0
    m = len(hdr); lo, hi = int(m * CFG.SPAN[0]), max(int(m * CFG.SPAN[1]) - 1, int(m * CFG.SPAN[0]))
    idx = np.linspace(lo, hi, n).round().astype(int) if m > 1 else np.zeros(n, int)
    uniq = sorted(set(idx.tolist()))
    def rd(i):
        try: return _read_px(hdr[i][3])
        except Exception: return None
    with ThreadPoolExecutor(threads) as ex:
        px = dict(zip(uniq, ex.map(rd, uniq)))
    good = [a for a in px.values() if a is not None]
    if not good: return None
    samp = np.concatenate([a[::4, ::4].ravel() for a in good])
    lo_v, hi_v = np.percentile(samp, CFG.PCT)
    out = []
    for i in idx:
        a = px[int(i)]
        a = np.zeros(shp, np.float32) if a is None else np.clip((a - lo_v) / max(hi_v - lo_v, 1e-6), 0, 1)
        out.append(mm_crop_resize(a, ps, size))
    return (np.stack(out) * 255).round().astype(np.uint8)

def study_volume(args):
    split, st, rows = args
    vol = np.zeros((CFG.TOTAL, CFG.IMG, CFG.IMG), np.uint8); smask = np.zeros(len(CFG.SLOTS), bool)
    used, pos = set(), 0
    for s, (plane, fluid, n) in enumerate(CFG.SLOTS):
        try:
            r, reused = pick_series(rows, plane, fluid, used)
            if r is not None:
                used.add(r["SeriesInstanceUID"])
                v = series_slices(glob.glob(f"{DATA}/{split}_series/{st}/{r['SeriesInstanceUID']}/*.dcm"), n, CFG.IMG)
                if v is not None: vol[pos:pos + n] = v; smask[s] = True
        except Exception as e:
            print(f"  [cache] {st[:12]} slot {s} failed: {type(e).__name__}: {str(e)[:80]}")
        pos += n
    return vol, smask

def plan_cache(n_train, n_test):
    global_img = CFG.IMG
    free = shutil.disk_usage(CFG.CACHE_DIR).free
    need = lambda img: (n_train + n_test) * CFG.TOTAL * img * img
    while CFG.AUTO_FIT_DISK and need(global_img) > 0.9 * free and global_img > 160:
        global_img -= 32
    if global_img != CFG.IMG:
        print(f"!! cache would need {need(CFG.IMG) / 1e9:.1f} GB, {free / 1e9:.1f} GB free -> IMG {CFG.IMG} -> {global_img}")
        CFG.IMG = global_img
    print(f"cache estimate: {need(CFG.IMG) / 1e9:.1f} GB at IMG={CFG.IMG} ({free / 1e9:.1f} GB free)")

def build_cache(split, studies):
    sig = hashlib.md5(json.dumps([CFG.SLOTS, CFG.SPAN, CFG.CROP_MM, CFG.IMG, CFG.PCT, CFG.ALLOW_SLOT_REUSE, len(studies)]).encode()).hexdigest()[:10]
    vp, mp = f"{CFG.CACHE_DIR}/{split}_{sig}_vol.npy", f"{CFG.CACHE_DIR}/{split}_{sig}_mask.npy"
    if os.path.exists(mp): return np.load(vp, mmap_mode="r"), np.load(mp)
    sdf = pd.read_csv(f"{DATA}/{split}_series.csv")
    g = {s: d.to_dict("records") for s, d in sdf.groupby("StudyInstanceUID")}
    vol = np.lib.format.open_memmap(vp, "w+", np.uint8, (len(studies), CFG.TOTAL, CFG.IMG, CFG.IMG))
    mask = np.zeros((len(studies), len(CFG.SLOTS)), bool)
    jobs = [(split, s, g.get(s, [])) for s in studies]
    t = time.time()
    if CFG.NUM_WORKERS == 0: it = map(study_volume, jobs)
    else: ex = ProcessPoolExecutor(max(1, os.cpu_count() or 1)); it = ex.map(study_volume, jobs, chunksize=2)
    for i, (v, m) in enumerate(it):
        vol[i], mask[i] = v, m
        if (i + 1) % 250 == 0: print(f"  {split}: {i + 1}/{len(studies)}  {time.time() - t:.0f}s")
    vol.flush(); np.save(mp, mask)
    print(f"cached {split}: {len(studies)} studies in {time.time() - t:.0f}s; slot coverage {mask.mean(0).round(3)}; "
          f"studies with no usable series: {int((mask.sum(1) == 0).sum())}")
    return np.load(vp, mmap_mode="r"), mask

if CFG.MODE == "train":
    if CFG.MAX_TRAIN_STUDIES:
        keep = sorted(np.where(gold_mask)[0].tolist() + [i for i in range(len(train)) if not gold_mask[i]][:CFG.MAX_TRAIN_STUDIES])
        train, rule, SOFT, gold_mask = train.iloc[keep].reset_index(drop=True), rule[keep], SOFT[keep], gold_mask[keep]
    studies = train.StudyInstanceUID.tolist()
    n_test = len(pd.read_csv(f"{DATA}/test.csv"))
    plan_cache(len(studies), n_test)
    VOL, MASK = build_cache("train", studies)
    print("train cache ready", elapsed())"""))

C.append(md(r"""## 4. Windows, augmentation and the model

* A **window** is 3 adjacent slices of the same slot (edges replicated, never crossing slot borders) used as RGB.
* Training samples `TRAIN_WINDOWS` windows per study (jittered even spacing over the valid windows); inference
  uses every `EVAL_STRIDE`-th slice as a centre.
* Augmentation on the GPU, shared within a slot of a study: small rotation / scale / shift, gamma and gain. **No
  naive flips** (they would swap medial/lateral). `MIRROR_AUG` applies the anatomical mirror instead.
* **Model.** timm CNN per window -> projection + slot embedding + slice-position embedding -> 2-layer transformer
  over all windows of the study -> **one attention distribution per finding** (local branch) ; a global branch
  (masked mean per slot -> linear) ; final logit = mean of both, each also supervised (deep supervision)."""))
C.append(code(r"""SLOT_OF = np.concatenate([[s] * n for s, (_, _, n) in enumerate(CFG.SLOTS)])           # slice -> slot
SLOT_START = np.cumsum([0] + [n for _, _, n in CFG.SLOTS])[:-1]
SAG_SLOTS = [s for s, (p, _, _) in enumerate(CFG.SLOTS) if p == "Sagittal"]
N_SLOT = len(CFG.SLOTS)

def window_table():
    # for every slice j (a window centre): the 3 slice indices (clipped to its slot) and its relative position in the slot
    idx = np.zeros((CFG.TOTAL, 3), np.int64); pos = np.zeros(CFG.TOTAL, np.float32)
    for s, (_, _, n) in enumerate(CFG.SLOTS):
        a = SLOT_START[s]
        for k in range(n):
            idx[a + k] = [a + max(k - 1, 0), a + k, a + min(k + 1, n - 1)]
            pos[a + k] = k / max(n - 1, 1)
    return idx, pos
WIN_IDX, WIN_POS = window_table()

class KneeDS(torch.utils.data.Dataset):
    # returns uint8 windows [K,3,H,W], slot id [K], position [K], window-valid [K], slot-valid [N_SLOT]
    def __init__(self, vol, mask, idx, y=None, train=False):
        self.vol, self.mask, self.idx, self.y, self.train = vol, mask, idx, y, train
        self.K = CFG.TRAIN_WINDOWS if train else len(range(0, CFG.TOTAL, CFG.EVAL_STRIDE))
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]; sm = self.mask[j].copy()
        valid_c = np.where(sm[SLOT_OF])[0]                                   # valid window centres
        K = self.K
        if self.train and len(valid_c):
            edges = np.linspace(0, len(valid_c), K + 1)
            sel = valid_c[np.minimum((edges[:-1] + np.random.rand(K) * np.diff(edges)).astype(int), len(valid_c) - 1)]
            wv = np.ones(K, bool)
        else:
            cen = np.arange(0, CFG.TOTAL, CFG.EVAL_STRIDE)
            sel = cen[sm[SLOT_OF[cen]]] if len(valid_c) else cen[:0]
            wv = np.zeros(K, bool); wv[:len(sel)] = True
            sel = np.pad(sel, (0, K - len(sel)))
        need = np.unique(WIN_IDX[sel]) if len(valid_c) else np.array([0])
        sl = np.zeros((CFG.TOTAL, CFG.IMG, CFG.IMG), np.uint8)
        if len(valid_c): sl[need] = self.vol[j][need]
        x = torch.from_numpy(sl[WIN_IDX[sel]])                               # K,3,H,W
        x[~torch.from_numpy(wv)] = 0
        out = {"x": x, "slot": torch.from_numpy(SLOT_OF[sel]).long(), "pos": torch.from_numpy(WIN_POS[sel]),
               "wv": torch.from_numpy(wv), "sm": torch.from_numpy(sm)}
        if self.y is not None: out["y"] = torch.from_numpy(self.y[i])
        return out

def mirror(x, slot):
    # anatomical mirror of [B,K,3,H,W]: sagittal windows reverse the 3 slice channels, coronal/axial flip the width
    is_sag = torch.isin(slot, torch.tensor(SAG_SLOTS, device=slot.device))[:, :, None, None, None]
    return torch.where(is_sag, x.flip(2), x.flip(-1))

def gpu_augment(x, slot):
    # x: B,K,3,H,W float in [0,1]; parameters are drawn per (study, slot) and shared by the windows of that slot
    B, K, _, H, W = x.shape; dev = x.device
    r = lambda lo, hi, *s: torch.empty(*s, device=dev).uniform_(lo, hi)
    g, gain, bias = r(0.7, 1.4, B, N_SLOT), r(0.85, 1.15, B, N_SLOT), r(-0.08, 0.08, B, N_SLOT)
    sc, ang, tx, ty = r(0.9, 1.1, B, N_SLOT), r(-0.15, 0.15, B, N_SLOT), r(-0.08, 0.08, B, N_SLOT), r(-0.08, 0.08, B, N_SLOT)
    take = lambda p: p.gather(1, slot)                                     # B,K
    gk, ak, sk = take(g), take(ang), take(sc)
    x = (x.clamp(0, 1) ** gk[..., None, None, None]) * take(gain)[..., None, None, None] + take(bias)[..., None, None, None]
    th = torch.zeros(B * K, 2, 3, device=dev)
    th[:, 0, 0] = (sk * torch.cos(ak)).flatten(); th[:, 0, 1] = (-sk * torch.sin(ak)).flatten()
    th[:, 1, 0] = (sk * torch.sin(ak)).flatten(); th[:, 1, 1] = (sk * torch.cos(ak)).flatten()
    th[:, 0, 2], th[:, 1, 2] = take(tx).flatten(), take(ty).flatten()
    grid = F.affine_grid(th, (B * K, 3, H, W), align_corners=False)
    x = F.grid_sample(x.reshape(B * K, 3, H, W), grid, align_corners=False, padding_mode="zeros")
    return x.reshape(B, K, 3, H, W).clamp(0, 1)

class Knee25DMIL(nn.Module):
    def __init__(self, backbone, pretrained, d=384, n=12, depth=2, drop=0.2, drop_path=0.1, grad_ckpt=False):
        super().__init__()
        kw = {"drop_path_rate": drop_path} if drop_path else {}
        self.enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0, in_chans=3, **kw)
        if grad_ckpt and hasattr(self.enc, "set_grad_checkpointing"): self.enc.set_grad_checkpointing(True)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        C = self.enc.num_features
        self.proj = nn.Sequential(nn.Dropout(drop), nn.Linear(C, d), nn.LayerNorm(d))
        self.slot_emb = nn.Embedding(N_SLOT, d)
        self.pos_emb = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, 8 if d % 8 == 0 else 4, 2 * d, 0.1, batch_first=True, norm_first=True)
        self.mix = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False) if depth else None
        self.att = nn.Sequential(nn.Linear(d, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n))   # one attention per finding
        self.w = nn.Parameter(torch.randn(n, d) * 0.02); self.b = nn.Parameter(torch.zeros(n))
        self.glob = nn.Sequential(nn.LayerNorm(N_SLOT * d), nn.Dropout(drop), nn.Linear(N_SLOT * d, n))
    def forward(self, x, slot, pos, wv):                 # x uint8-or-float [B,K,3,H,W] in [0,1]
        B, K = x.shape[:2]
        flat = x.flatten(0, 1)[wv.flatten()]
        if flat.shape[0]:
            f = self.enc(((flat - self.mean) / self.std).contiguous(memory_format=torch.channels_last))
        else:                                              # a batch made only of studies without series
            f = x.new_zeros(0, self.enc.num_features)
        feat = f.new_zeros(B * K, f.shape[-1]); feat[wv.flatten()] = f
        h = self.proj(feat.float().view(B, K, -1)) + self.slot_emb(slot) + self.pos_emb(pos.unsqueeze(-1).to(feat.dtype))
        pad = ~wv
        pad = pad & ~pad.all(1, keepdim=True)            # a study with no series attends to everything (output is overridden later)
        if self.mix is not None: h = self.mix(h, src_key_padding_mask=pad)
        a = self.att(h).float().masked_fill(pad.unsqueeze(-1), -1e4).softmax(1)             # B,K,12
        zl = (torch.einsum("bkc,bkd->bcd", a.to(h.dtype), h) * self.w).sum(-1) + self.b
        oh = F.one_hot(slot, N_SLOT).to(h.dtype) * wv.unsqueeze(-1).to(h.dtype)             # B,K,S
        sm = torch.einsum("bks,bkd->bsd", oh, h) / oh.sum(1).clamp_min(1).unsqueeze(-1)
        zg = self.glob(sm.flatten(1))
        return 0.5 * (zg + zl), zg, zl

def make_model(backbone, pretrained):
    return Knee25DMIL(backbone, pretrained, CFG.D_MODEL, 12, CFG.TF_DEPTH, CFG.DROP, CFG.DROP_PATH, CFG.GRAD_CKPT)

_m = make_model(CFG.BACKBONES[0], False)
print(f"{CFG.BACKBONES[0]}: {sum(p.numel() for p in _m.parameters()) / 1e6:.2f} M parameters | windows/study: train {CFG.TRAIN_WINDOWS}, eval {len(range(0, CFG.TOTAL, CFG.EVAL_STRIDE))}")
del _m"""))

C.append(md(r"""## 5. Training (expert-labelled studies held out; EMA weights; time-budget aware)

Each arm (backbone) is trained on the same stratified folds. Before each fold the notebook checks whether it still
fits inside `TIME_BUDGET_H` (using the time of the folds already done); if not, it stops and the ensemble uses the
completed folds."""))
C.append(code(r"""class EMA:
    def __init__(self, model, decay):
        self.decay, self.shadow = decay, {k: v.detach().clone().float() for k, v in model.state_dict().items()}
    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point: self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else: self.shadow[k].copy_(v)
    def state_dict(self, like): return {k: self.shadow[k].to(like[k].dtype) for k in like}

def forward_batch(model, b, train, mirror_flag=False):
    x = b["x"].to(DEVICE, non_blocking=True).float().div_(255); slot = b["slot"].to(DEVICE)
    pos, wv = b["pos"].to(DEVICE), b["wv"].to(DEVICE)
    if train: x = gpu_augment(x, slot)
    if mirror_flag: x = mirror(x, slot)
    with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=CFG.AMP and DEVICE.type == "cuda"):
        return model(x, slot, pos, wv)

def train_epoch(model, loader, opt, sched, scaler, ema):
    model.train(); tot, n = 0.0, 0
    for b in loader:
        y = b["y"].to(DEVICE)
        mir = CFG.MIRROR_AUG and random.random() < 0.5
        z, zg, zl = forward_batch(model, b, True, mir)
        bce = lambda t: F.binary_cross_entropy_with_logits(t.float(), y)
        loss = bce(z) + CFG.AUX_W * (bce(zg) + bce(zl))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.CLIP)
        scaler.step(opt); scaler.update(); sched.step()
        if ema is not None: ema.update(model.module if hasattr(model, "module") else model)
        tot += loss.item() * len(y); n += len(y)
    return tot / max(n, 1)

@torch.no_grad()
def predict(model, loader, tta=None):
    tta = CFG.MIRROR_TTA if tta is None else tta
    model.eval(); out = []
    for b in loader:
        p = torch.sigmoid(forward_batch(model, b, False)[0].float())
        if tta: p = 0.5 * (p + torch.sigmoid(forward_batch(model, b, False, True)[0].float()))
        p = p.cpu().numpy(); empty = ~b["sm"].any(1).numpy()
        if empty.any(): p[empty] = np.nan                    # no usable series: filled with the prevalence by the caller
        out.append(p)
    return np.concatenate(out)

def loader(idx, y=None, train=False):
    ds = KneeDS(VOL, MASK, idx, y, train)
    return torch.utils.data.DataLoader(ds, batch_size=CFG.BATCH if train else max(1, CFG.BATCH), shuffle=train,
                                       num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=train,
                                       persistent_workers=CFG.NUM_WORKERS > 0)

def fill_nan(P, prev): return np.where(np.isnan(P), prev[None], P)

if CFG.MODE == "train":
    trainable = np.ones(len(train), bool) if CFG.INCLUDE_GOLD else ~gold_mask
    pool = np.where(trainable)[0]; gold = np.where(gold_mask)[0]
    Ypool = SOFT[pool].copy()
    if CFG.INCLUDE_GOLD:                                       # expert labels replace the report labels for the gold studies
        Ypool[gold_mask[pool]] = train.loc[gold_mask, LABELS].values.astype(np.float32)   # pool and gold rows are both in ascending order
    Yg = train.loc[gold_mask, LABELS].values.astype(int)
    prev = Ypool.mean(0)
    strat = np.clip((Ypool >= 0.5).sum(1), 0, 5) * 2 + (Ypool[:, 0] >= 0.5)
    folds = np.zeros(len(pool), int)
    for f, (_, te) in enumerate(StratifiedKFold(CFG.FOLDS, shuffle=True, random_state=CFG.SEED).split(pool, strat)):
        folds[te] = f
    ARMS = {}                                                  # arm -> dict(oof, gold, files)
    fold_times = []
    stop = False
    for arm_i, backbone in enumerate(CFG.BACKBONES):
        arm = f"a{arm_i}"; ARMS[arm] = dict(backbone=backbone, oof=np.full(Ypool.shape, np.nan, np.float32), gold=[], files=[])
        for f in CFG.TRAIN_FOLDS:
            left = CFG.TIME_BUDGET_H * 3600 - (time.time() - T0) - CFG.INFER_RESERVE_MIN * 60
            if fold_times and left < 1.15 * np.mean(fold_times):
                print(f"!! time budget: {left / 60:.0f} min left < one fold ({np.mean(fold_times) / 60:.0f} min); stopping training"); stop = True; break
            t = time.time(); seed_all(CFG.SEED + f + 100 * arm_i)
            tr, va = pool[folds != f], pool[folds == f]
            model = make_model(backbone, CFG.PRETRAINED).to(DEVICE).to(memory_format=torch.channels_last)
            if N_GPU > 1: model = nn.DataParallel(model)
            core = model.module if hasattr(model, "module") else model
            enc_ids = {id(p) for p in core.enc.parameters()}
            opt = torch.optim.AdamW([{"params": list(core.enc.parameters()), "lr": CFG.LR_BACKBONE},
                                     {"params": [p for p in core.parameters() if id(p) not in enc_ids], "lr": CFG.LR_HEAD}],
                                    weight_decay=CFG.WD)
            dl = loader(tr, Ypool[folds != f], train=True)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[CFG.LR_BACKBONE, CFG.LR_HEAD], total_steps=CFG.EPOCHS * len(dl), pct_start=0.1)
            scaler = torch.amp.GradScaler(enabled=CFG.AMP and DEVICE.type == "cuda")
            ema = EMA(core, CFG.EMA) if CFG.EMA else None
            for ep in range(CFG.EPOCHS):
                loss = train_epoch(model, dl, opt, sched, scaler, ema)
                print(f"[{arm} {backbone}] fold {f} ep {ep + 1}/{CFG.EPOCHS} loss {loss:.4f}  {elapsed()}")
            if ema is not None: core.load_state_dict(ema.state_dict(core.state_dict()))   # evaluate and save the EMA weights
            pv = predict(model, loader(va)); yv = Ypool[folds == f]
            ARMS[arm]["oof"][folds == f] = fill_nan(pv, prev)
            print(f"== [{arm}] fold {f}: OOF macroAUC vs report labels {macro_auc((yv >= 0.5).astype(int), fill_nan(pv, prev), yv != 0.5):.4f}")
            if not CFG.INCLUDE_GOLD:
                gp = fill_nan(predict(model, loader(gold)), prev); ARMS[arm]["gold"].append(gp)
                print(f"   expert-set macroAUC (this fold model): {macro_auc(Yg, gp):.4f}")
            fn = f"{CFG.OUT_DIR}/{arm}_fold{f}.pt"; torch.save(core.state_dict(), fn); ARMS[arm]["files"].append(fn)
            fold_times.append(time.time() - t)
            del model, core, opt, dl; torch.cuda.empty_cache()
        if stop: break
    cfg_dump = {k: v for k, v in vars(CFG).items() if not k.startswith("_") and isinstance(v, (int, float, str, list, tuple, bool, type(None)))}
    cfg_dump["arms"] = {a: dict(backbone=v["backbone"], files=[os.path.basename(x) for x in v["files"]],
                                sha256=[sha256(x) for x in v["files"]]) for a, v in ARMS.items()}
    cfg_dump["prevalence"] = prev.tolist()"""))

C.append(md(r"""## 6. Arm fusion (fitted on out-of-fold predictions, never on the expert set) and evaluation

* With several arms, each finding gets a weight `w` for arm 1 vs arm 2 … chosen on the **OOF** AUC against the
  report labels over a grid and **shrunk 50 % toward equal weights**, because those labels are noisy. One arm = no fusion.
* Reported, all measured by this cell: OOF macro AUC; expert-set macro AUC with a 95 % bootstrap CI; expert-set
  accuracy at leave-one-out thresholds next to the "always negative" floor; a per-label table."""))
C.append(code(r"""def rankpct(P): return (pd.DataFrame(P).rank(method="average").values - 0.5) / len(P)

def fuse_weights(arms, Y):
    # returns W[arm, label]; coordinate search over a simplex grid on OOF, then shrink toward uniform
    names = list(arms); A = len(names)
    if A == 1: return {names[0]: np.ones(12)}
    R = {a: rankpct(arms[a]["oof"]) for a in names}
    W = np.full((A, 12), 1 / A)
    grid = np.linspace(0, 1, 11)
    for k in range(12):
        v = ~np.isnan(R[names[0]][:, k]) & (Y[:, k] != 0.5); y = (Y[v, k] >= 0.5).astype(int)
        if y.sum() == 0 or y.sum() == len(y): continue
        best, bw = -1, None
        for g in (grid if A == 2 else [None]):
            w = np.array([1 - g, g]) if A == 2 else np.full(A, 1 / A)
            s = roc_auc_score(y, sum(w[i] * R[names[i]][v, k] for i in range(A)))
            if s > best: best, bw = s, w
        W[:, k] = 0.5 * bw + 0.5 / A
    return {a: W[i] for i, a in enumerate(names)}

def ensemble(parts, W, key):
    names = list(parts); return sum(W[a][None] * rankpct(parts[a]) for a in names) if key == "rank" else sum(W[a][None] * parts[a] for a in names)

def loo_accuracy(y, p):
    n, L = y.shape; hit = np.zeros((n, L), bool)
    for k in range(L):
        cand = np.unique(np.r_[p[:, k], 1.01])
        for i in range(n):
            mm = np.arange(n) != i
            t = cand[np.argmax([((p[mm, k] >= c) == y[mm, k]).mean() for c in cand])]
            hit[i, k] = (p[i, k] >= t) == y[i, k]
    return hit.mean(), hit.mean(0)

if CFG.MODE == "train":
    done_arms = {a: v for a, v in ARMS.items() if v["files"]}
    W = fuse_weights(done_arms, Ypool)
    cfg_dump["fusion_weights"] = {a: w.tolist() for a, w in W.items()}
    oof_all = {a: v["oof"] for a, v in done_arms.items()}
    done = ~np.isnan(next(iter(oof_all.values()))).any(1)
    fused_oof = ensemble({a: o[done] for a, o in oof_all.items()}, W, "rank")
    yb = (Ypool[done] >= 0.5).astype(int)
    summary = {"label_source": LABEL_SOURCE, "labeler_gold_auc": scores[LABEL_SOURCE],
               "oof_macro_auc_fused": macro_auc(yb, fused_oof, Ypool[done] != 0.5),
               "oof_macro_auc_per_arm": {a: macro_auc(yb, o[done], Ypool[done] != 0.5) for a, o in oof_all.items()},
               "folds_trained_per_arm": {a: len(v["files"]) for a, v in done_arms.items()},
               "training_minutes": (time.time() - T0) / 60, "include_gold": CFG.INCLUDE_GOLD}
    if not CFG.INCLUDE_GOLD:
        Gp = {a: np.mean(v["gold"], 0) for a, v in done_arms.items()}
        G = ensemble(Gp, W, "rank")
        rng = np.random.default_rng(0); boots = []
        for _ in range(1000):
            i = rng.integers(0, len(Yg), len(Yg)); boots.append(macro_auc(Yg[i], G[i]))
        acc, acc_lab = loo_accuracy(Yg, G)
        summary.update({"gold_macro_auc_fused": macro_auc(Yg, G), "gold_auc_95ci": list(np.nanpercentile(boots, [2.5, 97.5])),
                        "gold_macro_auc_per_arm": {a: macro_auc(Yg, g) for a, g in Gp.items()},
                        "gold_accuracy_LOO_thresholds": acc, "gold_accuracy_all_negative": float((Yg == 0).mean())})
        per = pd.DataFrame({"gold_auc": [macro_auc(Yg[:, k:k + 1], G[:, k:k + 1]) for k in range(12)], "gold_acc_LOO": acc_lab,
                            "oof_auc": [macro_auc(yb[:, k:k + 1], fused_oof[:, k:k + 1], Ypool[done, k:k + 1] != 0.5) for k in range(12)]},
                           index=LABELS).round(4)
    else:
        print("INCLUDE_GOLD=True: the expert studies were trained on, so no expert-set metric is reported.")
        per = pd.DataFrame({"oof_auc": [macro_auc(yb[:, k:k + 1], fused_oof[:, k:k + 1], Ypool[done, k:k + 1] != 0.5) for k in range(12)]}, index=LABELS).round(4)
    json.dump(summary, open(f"{CFG.OUT_DIR}/metrics.json", "w"), indent=1, default=float)
    json.dump(cfg_dump, open(f"{CFG.OUT_DIR}/config.json", "w"), indent=1, default=float)
    print(json.dumps(summary, indent=1, default=float)); display(per)"""))

C.append(md(r"""## 7. Test inference → `submission.csv`

Every finished fold of every arm predicts the test studies (with mirror TTA); folds are averaged per arm, arms are
fused with the OOF weights. Studies with no readable series get the training prevalence. The output is checked
against `sample_submission.csv`."""))
C.append(code(r"""test = pd.read_csv(f"{DATA}/test.csv")
wdir = CFG.OUT_DIR if CFG.MODE == "train" else CFG.WEIGHTS_DIR
saved = json.load(open(f"{wdir}/config.json"))
if CFG.MODE == "infer":
    for k in ("SLOTS", "SPAN", "CROP_MM", "IMG", "PCT", "ALLOW_SLOT_REUSE", "D_MODEL", "TF_DEPTH", "TOTAL"):
        if k in saved: setattr(CFG, k, [tuple(s) for s in saved[k]] if k == "SLOTS" else saved[k])
    SLOT_OF = np.concatenate([[s] * n for s, (_, _, n) in enumerate(CFG.SLOTS)]); SLOT_START = np.cumsum([0] + [n for _, _, n in CFG.SLOTS])[:-1]
    SAG_SLOTS = [s for s, (p, _, _) in enumerate(CFG.SLOTS) if p == "Sagittal"]; N_SLOT = len(CFG.SLOTS); WIN_IDX, WIN_POS = window_table()
    CFG.AMP = CFG.AMP and DEVICE.type == "cuda"
prev = np.array(saved["prevalence"])
TVOL, TMASK = build_cache("test", test.StudyInstanceUID.tolist())
VOL, MASK = TVOL, TMASK
parts = {}
for arm, info in saved["arms"].items():
    ps = []
    for fn, h in zip(info["files"], info["sha256"]):
        path = f"{wdir}/{fn}"; assert sha256(path) == h, f"checkpoint changed: {fn}"
        model = make_model(info["backbone"], False).to(DEVICE).to(memory_format=torch.channels_last)
        model.load_state_dict(torch.load(path, map_location=DEVICE)); ps.append(fill_nan(predict(model, loader(np.arange(len(test)))), prev))
        del model; torch.cuda.empty_cache()
    parts[arm] = np.mean(ps, 0); print(f"arm {arm} ({info['backbone']}): {len(ps)} fold model(s)")
Wt = {a: np.array(w) for a, w in saved["fusion_weights"].items()}
if len(parts) > 1: P = ensemble(parts, Wt, "rank")             # weighted mean of per-column ranks, in (0, 1)
else: P = next(iter(parts.values()))
if CFG.OUTPUT == "rank": P = rankpct(P)
sub = pd.DataFrame(P, columns=LABELS); sub.insert(0, "StudyInstanceUID", test.StudyInstanceUID.values)
ss = pd.read_csv(f"{DATA}/sample_submission.csv"); sub = sub[ss.columns]
assert np.isfinite(sub[LABELS].values).all(), "non-finite predictions"
assert set(sub.StudyInstanceUID) == set(ss.StudyInstanceUID), "rows differ from sample_submission.csv"
sub.to_csv("submission.csv", index=False)
print(f"submission.csv: {sub.shape} | total {elapsed()}")
sub.head()"""))

nb = nbf.v4.new_notebook(cells=C)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["kaggle"] = {"accelerator": "gpu", "isInternetEnabled": True, "language": "python"}
out = ROOT / "notebooks/rsna_knee_kaggle_v2.ipynb"
nbf.write(nb, out)
print("wrote", out)
