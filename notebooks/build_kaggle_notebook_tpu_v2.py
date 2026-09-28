"""Builds notebooks/rsna_knee_kaggle_tpu_v2.ipynb, a self-contained Kaggle notebook (TPU v3-8).

    python notebooks/build_kaggle_notebook_tpu_v2.py

This is the TPU port of notebooks/rsna_knee_kaggle.ipynb (built by build_kaggle_notebook.py, GPU-only).
Same pipeline, same model, same evaluation; only what's needed to run on a single PyTorch/XLA TPU core
without erroring changed. Every change is explained in the notebook's own markdown ("What changed for
TPU") and in the inline comments next to the code it touches. The rule-based multilingual report labeler
is embedded verbatim from src/kneemor/report_labeler.py via %%writefile, so the notebook has no dependency
on this repository at run time.
"""
import pathlib

import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parents[1]
LABELER_SRC = (ROOT / "src/kneemor/report_labeler.py").read_text()

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
C = []

C.append(md(r"""# RSNA Knee Abnormality Detection: 2.5-D hybrid CNN (train + infer) — V2 (TPU)

This is a self-contained Kaggle notebook for the **TPU v3-8** accelerator: open **Settings -> Accelerator**
and pick **TPU VM v3-8**, turn **Internet** on, then **Run All**.

This is the TPU port of the GPU notebook `notebooks/rsna_knee_kaggle.ipynb` in the same repo. The pipeline,
model and evaluation are unchanged; only what's needed to run correctly on a single PyTorch/XLA TPU core
changed, listed below with the reasoning. If TPU isn't actually selected (or torch_xla can't reach a TPU),
the notebook falls back to GPU, then CPU, instead of crashing.

**Pipeline** (unchanged from the GPU version)
1. **Labels.** 58 training studies carry expert labels; the other ~4,350 have only a free-text report in
   one of eight languages. Two labelers turn those reports into training labels:
   * a multilingual rule labeler;
   * an *optional* local LLM labeler (attach a Qwen2.5-Instruct model) -- GPU/CPU only, see below.

   The notebook scores **both** on the 58 expert studies and trains on whichever agrees better.
2. **Preprocessing.** For each plane (sagittal, coronal, axial) it takes the best fluid-sensitive series,
   sorts it along the slice normal, resamples it to `SLICES` slices at `IMG` px, and caches the volume as
   uint8.
3. **Model: 2.5-D hybrid.**
   * An ImageNet-pretrained timm CNN runs on 3 adjacent slices as RGB and is **fine-tuned end-to-end**.
   * A **global branch** (per-plane mean -> linear) feeds the output directly.
   * A **local branch** mixes all slice tokens of all planes, then 12 label queries cross-attend to them.
     `CFG.HEAD_TYPE` picks the mixer:
     * `"more"` (**default, proposed**) -- **MV-MoRE**: one attention+FFN block, weight-shared and applied
       up to `MOR_RECURSIONS` times with MoR expert-choice depth routing (Bae et al., 2025), whose FFN is a
       top-k Mixture-of-Experts (`MOE_EXPERTS` experts, `MOE_TOPK` active per token; Shazeer 2017 / Switch,
       Fedus 2022 / ST-MoE router z-loss, Zoph 2022), rewritten below as a **dense, static-shape** dispatch
       for TPU (see "What changed for TPU").
     * `"transformer"` -- the original 2-layer unshared transformer. This is the architecture that produced
       the measured gold macro AUC 0.769 / OOF 0.834 result on GPU (see the repo README); set
       `CFG.HEAD_TYPE = "transformer"` to reproduce it, or leave `"more"` to run the proposed extension
       (its TPU numbers are **not yet measured** -- see "Targets vs. guarantees" below).
   * Each branch has its own auxiliary loss (plus the MoE load-balance/z-loss when `HEAD_TYPE="more"`).
4. **Training.** 5-fold CV on the report-labelled studies with AMP, a cosine schedule and soft BCE. The 58
   expert-labelled studies are **held out** and used only for evaluation.
5. **Evaluation.** OOF macro AUC; expert-set macro AUC with a bootstrap CI; expert-set **accuracy** with
   leave-one-out thresholds; per-label table; timings.
6. **Submission.** A fold ensemble writes `submission.csv`.

## What changed for TPU (and why)

PyTorch/XLA (the TPU backend) compiles a static computation graph -- it needs fixed tensor shapes -- and
lowers only a subset of ops. A few things in the GPU version either don't compile or aren't supported
there. Every change below is scoped to `DEVICE.type == "xla"`; running this same notebook on GPU/CPU is
unaffected and numerically identical to the GPU notebook:

* **Device selection.** `xm.xla_device()` is used when `torch_xla` is importable *and* a TPU actually
  answers a trivial op; otherwise it falls back to CUDA, then CPU. A missing/misconfigured accelerator
  degrades gracefully instead of crashing the whole run.
* **Single TPU core.** A v3-8 board exposes 8 cores; this notebook trains on **one** of them. Scaling to
  all 8 needs an `xmp.spawn`-based multi-process rewrite (per-core data sharding, gradient all-reduce,
  rank-0-only logging/checkpointing/labeling) -- a natural follow-up, but out of scope here: it multiplies
  the surface for hard-to-debug hangs, and this port's goal is a notebook that reliably finishes.
* **Sparse MoE -> dense, static-shape MoE.** The MV-MoRE FFN originally dispatched each token to its top-k
  experts with boolean-mask indexing (`out[mask] = ...`), whose output shape depends on the data (how many
  tokens land on each expert). XLA either can't compile that or has to recompile the graph every step --
  the documented reason real token-choice MoE-on-TPU implementations use fixed capacity. It's rewritten
  below as a dense weighted sum over **all** experts, gated by a static-shape matrix (built with `scatter`,
  not boolean masking) that is exactly zero for non-selected experts -- the same computation (same weights,
  same output) without a data-dependent shape. `HEAD_TYPE="transformer"` doesn't use this module at all, so
  its numbers are unaffected.
* **bf16 autocast, no loss-scaler.** TPUs compute natively in bfloat16, which (unlike fp16) doesn't need
  gradient scaling. Training uses `torch.autocast(device_type="xla", dtype=torch.bfloat16)` in place of the
  fp16 CUDA autocast. If that ever raises on an older `torch_xla`, the notebook catches it once, prints a
  warning, and continues in fp32 for the rest of the run instead of crashing.
* **No spatial (grid_sample) augmentation on TPU.** `grid_sample`/`affine_grid` have inconsistent XLA
  lowering and are reported to run far slower there than on GPU/CPU. On TPU the notebook keeps the
  intensity augmentation (gamma/brightness/contrast jitter -- all elementwise) and skips the small random
  affine warp; on GPU/CPU both apply, exactly as before.
* **Checkpoints via `xm.save` / `map_location="cpu"`.** `xm.save` moves XLA tensors to CPU before pickling
  (plain `torch.save` on live XLA tensors isn't a reliable round-trip); loading uses
  `torch.load(..., map_location="cpu")` followed by `.to(DEVICE)`, rather than mapping straight to an XLA
  device.
* **`DataLoader(num_workers=0)` on TPU.** Forking worker processes after the XLA/PJRT runtime has opened
  its connection to the TPU in the parent process is a known source of hangs. Each `Dataset.__getitem__`
  here is just a memmap read (the expensive DICOM decoding already happened in `build_cache`, via a plain
  `ProcessPoolExecutor` that never touches torch/XLA, so it isn't affected), so losing worker parallelism
  for this step costs little.
* **LLM labeler disabled on TPU.** `USE_LLM_LABELS` loads a Hugging Face causal LM with
  `device_map="auto"`, which places weights on CUDA/CPU, not XLA. It's auto-disabled when
  `DEVICE.type == "xla"`, falling back to the rule labeler (the default candidate anyway).
* **Paths fixed for Kaggle.** `DATA_DIR` / `CACHE_DIR` / `OUT_DIR` point at `/kaggle/input` (auto-detected)
  and `/kaggle/working`, matching how this notebook actually runs on Kaggle.

## More robust Kaggle input-path detection

`find_data_dir()` now checks **both** mount conventions Kaggle uses for attached data -- the classic
`/kaggle/input/<name>` and, for a dataset attached via the newer picker with an owner-qualified handle,
`/kaggle/input/datasets/<owner>/<name>` -- via a bounded-depth walk that steps around the huge per-study
DICOM trees (`train_series`/`test_series`) instead of descending into them. The same helper
(`find_kaggle_path`) is used as a fallback for `WEIGHTS_DIR` in `MODE="infer"`, so a weights dataset
attached under either convention is still found instead of requiring an exact path match. This generalises
the same idea used for asset discovery in other Kaggle inference notebooks; only the general technique
(check both mount shapes, walk with bounded depth, skip the big series folders) is reused here, not any
specific paths, checkpoints, or code from elsewhere.

**Targets vs. guarantees.** As before: nothing here *guarantees* a particular macro AUC or accuracy; the
notebook **measures and prints** what it achieves. The default MV-MoRE head's TPU numbers are **not yet
measured** by this notebook -- only the GPU run with `HEAD_TYPE="transformer"` has a reported gold result.
This port's job is to run the pipeline correctly and reproducibly on TPU, not to claim a new number.

Two effects bound what any score can mean:
* In the author's CPU study, rule labels agreed with the expert labels at about 0.735 AUC, which caps
  models trained on them. The LLM labeler exists to raise that cap (GPU/CPU only, see above).
* The expert set has only 58 studies (95% CI about +/-0.06 AUC).

**Two ways to run:**
* `MODE="train"` (internet ON): trains, evaluates and writes the weights and a submission.
* `MODE="infer"` (internet OFF): set `WEIGHTS_DIR` to a Kaggle dataset containing the saved `fold*.pt` and
  `config.json`. TPU is generally not available in a no-internet code-competition rerun, so use `MODE="train"`
  here on TPU to produce and validate the weights, and a GPU/CPU notebook with `MODE="infer"` for the actual
  submission run."""))

C.append(code(r"""# ============================== CONFIG ==============================
import os, glob, json, math, time, random, warnings
warnings.filterwarnings("ignore")

class CFG:
    MODE = "train"                 # "train" (train+eval+submit) | "infer" (load WEIGHTS_DIR, submit only)
    SEED = 42
    # data
    DATA_DIR = None                # None -> auto-detect under /kaggle/input
    CACHE_DIR = "/kaggle/working/cache"
    OUT_DIR = "/kaggle/working"
    WEIGHTS_DIR = "/kaggle/input/knee-hybrid-weights"   # used when MODE == "infer"
    PLANES = ["Sagittal", "Coronal", "Axial"]
    SLICES = 16                    # slices per plane
    IMG = 224                      # in-plane size
    # model
    BACKBONE = "tf_efficientnet_b0.ns_jft_in1k"         # alternatives: "convnext_nano.in12k_ft_in1k", "resnet34.a1_in1k"
    PRETRAINED = True              # needs internet in MODE="train"
    D_MODEL = 256
    HEAD_TYPE = "more"             # "more" (MV-MoRE: shared recursive block + dense top-k-gated MoE FFN, proposed)
                                    # | "transformer" (original 2-layer unshared transformer, kept for comparison:
                                    #   this is what the GPU notebook's metrics.json with HEAD_TYPE="transformer" measured)
    MOR_RECURSIONS = 3             # MoR: max weight-shared recursions per token (expert-choice depth routing)
    MOE_EXPERTS = 4                # MoE: experts in the shared block's FFN
    MOE_TOPK = 2                   # MoE: experts each token is dispatched to (dense-gated for TPU; see model cell)
    # training
    FOLDS = 5
    TRAIN_FOLDS = [0, 1, 2, 3, 4]  # subset to save time, e.g. [0]
    EPOCHS = 12
    BATCH = 4                      # studies per step (each study = 3*SLICES images)
    LR_BACKBONE = 2e-4
    LR_HEAD = 1e-3
    WD = 1e-2
    AUX_W = 0.5                    # deep-supervision weight for each branch
    NUM_WORKERS = 4                # forced to 0 on TPU/XLA in the next cell -- see "What changed for TPU"
    AMP = True
    # labels
    USE_LLM_LABELS = False         # True -> attach a Qwen2.5-Instruct model and set LLM_PATH (GPU/CPU only)
    LLM_PATH = "/kaggle/input/qwen2.5/transformers/3b-instruct/1"
    LLM_BATCH = 8
    # debug / smoke
    SMOKE = False                  # tiny run to check the pipeline end-to-end
    MAX_TRAIN_STUDIES = None       # e.g. 300 for a quick trial

if os.environ.get("KNEE_SMOKE") == "1":   # used only by the author's CPU smoke test
    CFG.SMOKE, CFG.DATA_DIR = True, os.environ["KNEE_DATA_DIR"]
    CFG.CACHE_DIR = CFG.OUT_DIR = os.environ["KNEE_OUT_DIR"]
    CFG.BACKBONE, CFG.PRETRAINED, CFG.SLICES, CFG.IMG = "resnet18", False, 6, 64
    CFG.FOLDS, CFG.TRAIN_FOLDS, CFG.EPOCHS, CFG.BATCH, CFG.NUM_WORKERS, CFG.AMP = 2, [0, 1], 1, 2, 0, False
    CFG.MODE = os.environ.get("KNEE_MODE", "train"); CFG.WEIGHTS_DIR = os.environ.get("KNEE_WEIGHTS", CFG.OUT_DIR)
os.makedirs(CFG.CACHE_DIR, exist_ok=True); os.makedirs(CFG.OUT_DIR, exist_ok=True)
print({k: v for k, v in vars(CFG).items() if not k.startswith("_")})"""))

C.append(code(r"""import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import cv2, pydicom, timm
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(CFG.SEED)

# ------------------------- accelerator: TPU (XLA) > GPU (CUDA) > CPU -------------------------
# Falls back gracefully if "TPU VM v3-8" wasn't selected in Settings (or torch_xla can't reach a
# TPU), so the same notebook still runs -- just not on TPU.
xm = None
def setup_device():
    global xm
    try:
        import torch_xla.core.xla_model as _xm
    except ImportError:
        try:
            import subprocess, sys
            ver = torch.__version__.split("+")[0]
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"torch_xla[tpu]=={ver}",
                            "-f", "https://storage.googleapis.com/libtpu-releases/index.html"],
                           check=True, timeout=600)
            import torch_xla.core.xla_model as _xm
        except Exception as e:
            print(f"torch_xla not available ({e!r}); using GPU/CPU instead.")
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        dev = _xm.xla_device()
        _ = (torch.ones(1, device=dev) + 1).cpu()   # make sure the TPU actually answers
        xm = _xm
        return dev
    except Exception as e:
        print(f"TPU hardware not reachable ({e!r}); falling back to GPU/CPU. "
              f"Select Settings -> Accelerator -> TPU VM v3-8 to use a TPU.")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = setup_device()
N_GPU = torch.cuda.device_count()
print("device:", DEVICE, "| GPUs:", N_GPU, "| torch", torch.__version__, "| timm", timm.__version__)

if DEVICE.type == "xla" and CFG.NUM_WORKERS != 0:
    print("NUM_WORKERS -> 0 on TPU/XLA: each __getitem__ is a cheap memmap read (DICOM decoding already "
          "happened in build_cache), and forking DataLoader workers after the XLA runtime is initialized "
          "in this process is a known source of hangs.")
    CFG.NUM_WORKERS = 0
if CFG.USE_LLM_LABELS and DEVICE.type == "xla":
    print("USE_LLM_LABELS=True is not supported on TPU (transformers device_map='auto' targets CUDA/CPU); "
          "disabling for this run and using the rule labeler.")
    CFG.USE_LLM_LABELS = False

def _kaggle_roots():
    # Kaggle mounts attached data at either "/kaggle/input/<name>" (classic) or, for a dataset attached
    # via the newer picker with an owner-qualified handle, "/kaggle/input/datasets/<owner>/<name>".
    # Checking both conventions -- rather than assuming one -- means the notebook isn't only working
    # because of how a particular dataset happened to get attached last time.
    return sorted(glob.glob("/kaggle/input/*")) + sorted(glob.glob("/kaggle/input/datasets/*/*"))

def _kaggle_walk(root, max_depth=4, skip=("train_series", "test_series", "train_images", "test_images")):
    # Bounded-depth walk that steps around the huge per-study DICOM trees: irrelevant when looking for a
    # handful of top-level marker/weight files, and walking them in full would be slow.
    level = [root]
    for _ in range(max_depth):
        nxt = []
        for d in level:
            try:
                entries = sorted(os.scandir(d), key=lambda e: e.name)
            except OSError:
                continue
            yield d, [e.name for e in entries if e.is_file()]
            nxt += [e.path for e in entries if e.is_dir() and e.name not in skip]
        level = nxt

def find_data_dir():
    if CFG.DATA_DIR: return CFG.DATA_DIR
    for root in _kaggle_roots():
        for d, files in _kaggle_walk(root):
            if "test_series.csv" in files: return d
    raise FileNotFoundError("competition data (test_series.csv) not found under /kaggle/input")
DATA = find_data_dir(); print("data:", DATA)

def find_kaggle_path(*name_globs):
    # Search both Kaggle mount conventions (see _kaggle_roots) for a file or directory matching any of
    # name_globs. Used as a fallback when a configured path (WEIGHTS_DIR, LLM_PATH) doesn't exist exactly
    # as given -- e.g. because a dataset got attached under the owner-qualified path this run.
    import fnmatch
    for root in _kaggle_roots():
        for d, files in _kaggle_walk(root):
            if any(fnmatch.fnmatch(os.path.basename(d), pat) for pat in name_globs): return d
            if any(fnmatch.fnmatch(f, pat) for f in files for pat in name_globs): return d
    return None

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
T0 = time.time()
def elapsed(): return f"{(time.time() - T0) / 60:.1f} min" """))

C.append(md("## 1. Multilingual rule labeler (embedded)"))
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
    print(f"studies: {len(train)} | expert-labelled (held out): {gold_mask.sum()}")
    rule = label_frame(train).set_index("StudyInstanceUID").loc[train.StudyInstanceUID, LABELS].values
    ev = eval_labeler(train[gold_mask], pd.DataFrame(rule[gold_mask], columns=LABELS)
                      .assign(StudyInstanceUID=train.StudyInstanceUID[gold_mask].values))
    RULE_GOLD_AUC = float(ev.auc.mean())
    print(f"rule labeler vs expert labels: macro AUC {RULE_GOLD_AUC:.4f}")
    display(ev[["label", "n_pos", "auc", "sens", "spec"]].round(3))"""))

C.append(md(r"""## 2. Optional LLM labeler (the biggest expected lever)

Set `USE_LLM_LABELS=True` and attach an instruction model, for example **Qwen2.5-3B-Instruct** or 7B from
Kaggle Models, then point `LLM_PATH` at it. **GPU/CPU only** -- `device_map="auto"` doesn't place weights on
an XLA device, so this is force-disabled when the notebook is running on TPU (see the device-setup cell);
it falls back to the rule labeler, which is the default candidate anyway.

* The model reads each report in its original language and returns JSON with a probability per finding.
* It is scored on the 58 expert studies **exactly like the rule labeler**.
* The better of the two is used for training. If the LLM wins, its labels are averaged with the rule
  labels (a simple ensemble), and that average is scored too.

The labeler choice is the only decision made using the expert set, and there are only three candidates."""))

C.append(code(r"""LLM_PROMPT = '''You are a musculoskeletal radiologist. Read the knee MRI report (any language) and rate each finding
as present in THIS knee on this exam. Answer ONLY with JSON, values between 0 and 1 (1 = definitely present,
0 = absent/not mentioned, 0.5 = equivocal). Definitions:
ACL: ACL tear (partial or complete). MCL: MCL sprain/tear (any grade). Medial Meniscus / Lateral Meniscus: meniscal tear
(not degeneration alone). Medial OA / Lateral OA / PF OA: osteoarthritis of that compartment (cartilage loss,
osteophytes, degenerative joint disease). Effusion: joint effusion more than physiological. Synovitis: synovitis or
synovial thickening. Baker's: Baker/popliteal cyst. Contusion: bone contusion/bruise/traumatic marrow edema.
Fracture: any fracture.
Keys: "ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture".
REPORT:
'''

import re

def parse_llm_json(txt):
    '''Strict JSON first; on failure, fall back to regex "key": value pairs so a model that
    emits near-JSON (trailing commas, stray text, a missing brace) still contributes labels
    instead of the whole study silently falling back to the rule labeler.'''
    try:
        js = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
        return [float(js.get(k, np.nan)) for k in LABELS]
    except Exception:
        pass
    out = {}
    for k in LABELS:
        m = re.search(re.escape(k) + r'"?\s*[:=]\s*"?(-?[0-9]*\.?[0-9]+)', txt)
        if m:
            try: out[k] = float(m.group(1))
            except ValueError: pass
    return [out.get(k, np.nan) for k in LABELS]

def llm_label(reports):
    '''Returns None (never raises) if the LLM cannot be loaded, so a bad LLM_PATH degrades
    gracefully to the rule labeler instead of failing the whole training run.'''
    if not CFG.LLM_PATH or not os.path.exists(CFG.LLM_PATH):
        print(f"USE_LLM_LABELS=True but LLM_PATH not found ({CFG.LLM_PATH!r}); skipping LLM labels, using rules only.")
        return None
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(CFG.LLM_PATH, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(CFG.LLM_PATH, torch_dtype=torch.float16, device_map="auto").eval()
    except Exception as e:
        print(f"LLM load failed ({e!r}); skipping LLM labels, using rules only.")
        return None
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
            vals = parse_llm_json(txt)
            if not all(np.isnan(vals)):
                out[i + j] = vals
        if i % (CFG.LLM_BATCH * 50) == 0: print(f"  LLM {i}/{len(reports)} {elapsed()}")
    del model; torch.cuda.empty_cache()
    return out

if CFG.MODE == "train":
    candidates = {"rules": rule}
    if CFG.USE_LLM_LABELS:
        llm = llm_label(train.Report.tolist())
        if llm is not None:
            ok = ~np.isnan(llm).any(1); print(f"LLM parsed {ok.mean():.1%} of reports")
            llm = np.where(np.isnan(llm), rule, np.clip(llm, 0, 1))
            candidates["llm"] = llm
            candidates["llm+rules"] = (llm + rule) / 2
    Yg = train.loc[gold_mask, LABELS].values.astype(int)
    scores = {k: macro_auc(Yg, v[gold_mask]) for k, v in candidates.items()}
    print("labeler macro AUC vs expert labels:", {k: round(v, 4) for k, v in scores.items()})
    LABEL_SOURCE = max(scores, key=scores.get)
    SOFT = candidates[LABEL_SOURCE].astype(np.float32)
    print("-> training labels from:", LABEL_SOURCE)"""))

C.append(md("## 3. DICOM -> cached volumes"))
C.append(code(r"""def select_series(sdf):
    df = sdf.copy()
    df["_r"] = -df.Fluid_Sensitive * 2 - df.Fat_Suppression
    df = df.sort_values(["StudyInstanceUID", "Anatomical_Plane", "_r", "SeriesInstanceUID"])
    return df.groupby(["StudyInstanceUID", "Anatomical_Plane"]).head(1)

def _pos(ds):
    try:
        o = np.array(ds.ImageOrientationPatient, float); p = np.array(ds.ImagePositionPatient, float)
        return float(np.dot(np.cross(o[:3], o[3:]), p))
    except Exception:
        return float(getattr(ds, "InstanceNumber", 0))

def series_volume(files, S, size):
    hdr = []
    for f in files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
            hdr.append((_pos(ds), (int(ds.Rows), int(ds.Columns)), f))
        except Exception:
            pass
    if not hdr: return None
    shp = pd.Series([h[1] for h in hdr]).value_counts().index[0]
    hdr = sorted([h for h in hdr if h[1] == shp], key=lambda h: h[0])
    idx = np.linspace(0, len(hdr) - 1, S).round().astype(int)
    imgs = []
    for i in idx:
        try:
            a = pydicom.dcmread(hdr[i][2], force=True).pixel_array.astype(np.float32)
            if a.ndim != 2: raise ValueError
        except Exception:
            a = np.zeros(shp, np.float32)
        imgs.append(cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA))
    v = np.stack(imgs); lo, hi = np.percentile(v, [0.5, 99.5])
    return (np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)

def study_volume(args):
    split, st, rows = args
    vol = np.zeros((len(CFG.PLANES), CFG.SLICES, CFG.IMG, CFG.IMG), np.uint8); mask = np.zeros(len(CFG.PLANES), bool)
    for plane, sid in rows:
        if plane not in CFG.PLANES: continue
        v = series_volume(glob.glob(f"{DATA}/{split}_series/{st}/{sid}/*.dcm"), CFG.SLICES, CFG.IMG)
        if v is not None:
            vol[CFG.PLANES.index(plane)] = v; mask[CFG.PLANES.index(plane)] = True
    return vol, mask

def build_cache(split, studies):
    tag = f"{split}_{CFG.SLICES}x{CFG.IMG}_{len(studies)}"
    vp, mp = f"{CFG.CACHE_DIR}/{tag}_vol.npy", f"{CFG.CACHE_DIR}/{tag}_mask.npy"
    if os.path.exists(mp):
        return np.load(vp, mmap_mode="r"), np.load(mp)
    sel = select_series(pd.read_csv(f"{DATA}/{split}_series.csv"))
    g = {s: list(zip(d.Anatomical_Plane, d.SeriesInstanceUID)) for s, d in sel.groupby("StudyInstanceUID")}
    vol = np.lib.format.open_memmap(vp, "w+", np.uint8, (len(studies), len(CFG.PLANES), CFG.SLICES, CFG.IMG, CFG.IMG))
    mask = np.zeros((len(studies), len(CFG.PLANES)), bool)
    jobs = [(split, s, g.get(s, [])) for s in studies]
    workers = max(1, os.cpu_count() or 1)
    t = time.time()
    if CFG.NUM_WORKERS == 0:
        it = map(study_volume, jobs)
    else:
        ex = ProcessPoolExecutor(workers); it = ex.map(study_volume, jobs, chunksize=4)
    for i, (v, m) in enumerate(it):
        vol[i], mask[i] = v, m
        if (i + 1) % 250 == 0: print(f"  {split}: {i + 1}/{len(studies)}  {time.time() - t:.0f}s")
    vol.flush(); np.save(mp, mask)
    print(f"cached {split}: {len(studies)} studies in {time.time() - t:.0f}s; planes present: {mask.mean(0).round(3)}")
    return np.load(vp, mmap_mode="r"), mask

if CFG.MODE == "train":
    studies = train.StudyInstanceUID.tolist()
    if CFG.MAX_TRAIN_STUDIES:
        keep = np.where(gold_mask)[0].tolist() + [i for i in range(len(train)) if not gold_mask[i]][:CFG.MAX_TRAIN_STUDIES]
        keep = sorted(keep); train, rule, SOFT, gold_mask = train.iloc[keep].reset_index(drop=True), rule[keep], SOFT[keep], gold_mask[keep]
        studies = train.StudyInstanceUID.tolist()
    VOL, MASK = build_cache("train", studies)
    print("train cache ready", elapsed())"""))

C.append(md("## 4. Dataset & 2.5-D hybrid model"))
C.append(code(r"""class KneeDS(torch.utils.data.Dataset):
    def __init__(self, vol, mask, idx, y=None, train=False):
        self.vol, self.mask, self.idx, self.y, self.train = vol, mask, idx, y, train
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        v = torch.from_numpy(np.array(self.vol[j]))                # P,S,H,W uint8
        m = torch.from_numpy(self.mask[j].copy())
        if self.train:
            if random.random() < 0.15 and m.sum() > 1:             # plane dropout (missing-sequence robustness)
                k = random.choice(torch.where(m)[0].tolist()); m[k] = False; v[k] = 0
        out = {"x": v, "m": m}
        if self.y is not None: out["y"] = torch.from_numpy(self.y[i])
        return out


def gpu_augment(x):
    # x: B,P,S,H,W float in [0,1]. Intensity + (GPU/CPU only) small affine; NO flips (would swap medial/
    # lateral or ant/post). grid_sample/affine_grid have inconsistent XLA lowering and are reported far
    # slower on TPU than on GPU/CPU (see "What changed for TPU"), so the spatial warp is skipped there and
    # only the (elementwise, always-safe) intensity jitter applies.
    B, P = x.shape[:2]
    g = torch.empty(B, P, 1, 1, 1, device=x.device).uniform_(0.7, 1.4)
    x = x.clamp(0, 1) ** g
    x = x * torch.empty(B, P, 1, 1, 1, device=x.device).uniform_(0.85, 1.15) + torch.empty(B, P, 1, 1, 1, device=x.device).uniform_(-0.08, 0.08)
    x = x.clamp(0, 1)
    if DEVICE.type == "xla":
        return x
    th = torch.zeros(B * P, 2, 3, device=x.device)
    sc = torch.empty(B * P, device=x.device).uniform_(0.9, 1.1); ang = torch.empty(B * P, device=x.device).uniform_(-0.15, 0.15)
    th[:, 0, 0] = sc * torch.cos(ang); th[:, 0, 1] = -sc * torch.sin(ang); th[:, 1, 0] = sc * torch.sin(ang); th[:, 1, 1] = sc * torch.cos(ang)
    th[:, :, 2] = torch.empty(B * P, 2, device=x.device).uniform_(-0.08, 0.08)
    S, H, W = x.shape[2:]
    grid = F.affine_grid(th, (B * P, S, H, W), align_corners=False)
    x = F.grid_sample(x.reshape(B * P, S, H, W), grid, align_corners=False, padding_mode="zeros")
    return x.reshape(B, P, S, H, W).clamp(0, 1)

class MoEFFN(nn.Module):
    '''Top-k Mixture-of-Experts FFN (Shazeer 2017 / Switch, Fedus 2022 / ST-MoE, Zoph 2022), with a dense,
    static-shape dispatch: every expert runs on every token, gated by a matrix built with `scatter` (not
    boolean-mask indexing) that is exactly zero for non-selected experts. This computes the same thing as
    sparse top-k gather-dispatch (same weights, same output) but with a shape that never depends on which
    tokens route where -- boolean-mask gather has a data-dependent output shape, which XLA/TPU either can't
    compile or has to recompile every step for (the documented reason real token-choice MoE-on-TPU
    implementations use fixed capacity). The extra compute (all `n_experts` run, vs `top_k`) is minor here
    since the MoE FFN is tiny next to the CNN backbone. Exposes aux_loss (load-balancing) and z_loss
    (router-logit penalty, the documented cause of MoE training instability) for the caller to add to the
    task loss. Ported from src/kneemor/models.py MoEFFN so the notebook stays self-contained on Kaggle.'''
    def __init__(self, d, ffn, drop, n_experts=4, top_k=2):
        super().__init__()
        assert 1 <= top_k <= n_experts
        self.n_experts, self.top_k = n_experts, top_k
        self.router = nn.Linear(d, n_experts)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Dropout(drop),
                                                     nn.Linear(ffn, d)) for _ in range(n_experts)])
        self.aux_loss = torch.zeros(()); self.z_loss = torch.zeros(())
    def forward(self, x):
        shape = x.shape; flat = x.reshape(-1, shape[-1])
        logits = self.router(flat); probs = logits.softmax(-1)
        topv, topi = probs.topk(self.top_k, dim=-1)
        topv = topv / topv.sum(-1, keepdim=True).clamp_min(1e-9)
        gate = torch.zeros(flat.shape[0], self.n_experts, dtype=flat.dtype, device=flat.device)
        gate = gate.scatter(1, topi, topv.to(flat.dtype))            # static-shape gate, zero off top-k
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            out = out + gate[:, e:e + 1] * expert(flat)
        importance = probs.mean(0)
        frac = (gate > 0).float().mean(0)
        self.aux_loss = self.n_experts * (importance * frac).sum()          # switch load-balance loss
        self.z_loss = (torch.logsumexp(logits, dim=-1) ** 2).mean()         # ST-MoE router z-loss
        return out.reshape(shape)

class MoREBlock(nn.Module):
    '''Pre-norm attention + MoE-FFN block; one instance is reused every recursion (MoR weight sharing).'''
    def __init__(self, d, heads, ffn, drop, n_experts, top_k):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ffn = MoEFFN(d, ffn, drop, n_experts, top_k)
        self.drop = nn.Dropout(drop)
    def residual(self, h, pad):
        x = self.n1(h)
        a = self.attn(x, x, x, key_padding_mask=pad, need_weights=False)[0]
        u = h + self.drop(a)
        y = self.ffn(self.n2(u))
        return u + self.drop(y) - h

class MoRELocal(nn.Module):
    '''MV-MoRE local mixer: one shared MoREBlock applied R times with MoR expert-choice depth
    routing (Bae et al., 2025) -- replaces the plain unshared 2-layer transformer used for
    HEAD_TYPE="transformer". Adds per-token Mixture-of-Experts capacity at near-zero extra FLOPs
    relative to the CNN backbone (the backbone is 3-4 orders of magnitude larger in compute; see
    README "Proposed extension: MV-MoRE"). All indexing here (topk with a static k, gather, scatter_add) is
    already static-shape and XLA-safe; only MoEFFN's internal dispatch needed rewriting for TPU (see above).
    aux_loss (load-balance + router z-loss, averaged over the recursion steps run) is exposed for the
    training loop to add to the task loss.'''
    def __init__(self, d, heads=4, ffn_mult=2, recursions=3, capacity=(1.0, 0.5, 0.25), n_experts=4, top_k=2, drop=0.1):
        super().__init__()
        self.block = MoREBlock(d, heads, ffn_mult * d, drop, n_experts, top_k)
        self.R, self.capacity = recursions, list(capacity)
        self.routers = nn.ModuleList([nn.Linear(d, 1) for _ in range(recursions)])
        self.aux_loss = torch.zeros(())
    def forward(self, h, pad):
        B, T, d = h.shape
        active = ~pad
        moe_losses = []
        for r in range(self.R):
            score = self.routers[r](h).squeeze(-1)
            k = max(1, min(T, math.ceil(self.capacity[r] * T)))
            masked = score.masked_fill(~active, float("-inf"))
            idx = masked.topk(k, dim=1).indices
            sel_valid = torch.gather(active, 1, idx)
            hs = torch.gather(h, 1, idx[..., None].expand(-1, -1, d))
            g = torch.sigmoid(torch.gather(score, 1, idx)).unsqueeze(-1)
            upd = self.block.residual(hs, ~sel_valid) * g * sel_valid.unsqueeze(-1).float()
            moe_losses.append(0.01 * self.block.ffn.aux_loss + 0.001 * self.block.ffn.z_loss)
            h = h.scatter_add(1, idx[..., None].expand(-1, -1, d), upd)
            active = torch.zeros_like(active).scatter(1, idx, sel_valid)
        self.aux_loss = torch.stack(moe_losses).mean()
        return h

class Knee25DHybrid(nn.Module):
    '''2.5-D CNN (3 adjacent slices -> RGB) + global branch + local mixer + label-query branch.
    CFG.HEAD_TYPE picks the local mixer: "more" (MV-MoRE, proposed) or "transformer" (original
    baseline, kept so the two are a true apples-to-apples ablation on identical tokens/training).'''
    def __init__(self, backbone, pretrained, P, S, d=256, n=12, head_type="more",
                 recursions=3, n_experts=4, top_k=2):
        super().__init__()
        self.enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0, in_chans=3)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        C = self.enc.num_features
        self.glob = nn.Sequential(nn.LayerNorm(P * C), nn.Dropout(0.2), nn.Linear(P * C, n))
        self.proj = nn.Sequential(nn.LayerNorm(C), nn.Linear(C, d))
        self.plane = nn.Parameter(torch.zeros(P, 1, d)); self.pos = nn.Parameter(torch.zeros(P, S, d))
        nn.init.trunc_normal_(self.plane, std=0.02); nn.init.trunc_normal_(self.pos, std=0.02)
        self.head_type = head_type
        if head_type == "more":
            self.mix = MoRELocal(d, heads=4, ffn_mult=2, recursions=recursions, n_experts=n_experts, top_k=top_k, drop=0.1)
        else:
            layer = nn.TransformerEncoderLayer(d, 4, 2 * d, 0.1, batch_first=True, norm_first=True)
            self.mix = nn.TransformerEncoder(layer, 2)
        self.q = nn.Parameter(torch.randn(n, d) * 0.02)
        self.xattn = nn.MultiheadAttention(d, 4, dropout=0.1, batch_first=True)
        self.nq, self.nk = nn.LayerNorm(d), nn.LayerNorm(d)
        self.w = nn.Parameter(torch.randn(n, d) * 0.02); self.b = nn.Parameter(torch.zeros(n))
    def forward(self, x, m):                        # x: B,P,S,H,W in [0,1]; m: B,P bool
        B, P, S, H, W = x.shape
        xp = torch.cat([x[:, :, :1], x, x[:, :, -1:]], 2)             # replicate-pad slices
        rgb = torch.stack([xp[:, :, i:i + S] for i in range(3)], 3)  # B,P,S,3,H,W
        f = self.enc((rgb.reshape(-1, 3, H, W) - self.mean) / self.std)      # B*P*S, C
        f = f.float().view(B, P, S, -1)
        mf = m[..., None].float()
        zg = self.glob((f.mean(2) * mf).flatten(1))
        t = (self.proj(f) + self.plane + self.pos).reshape(B, P * S, -1)
        pad = (~m)[:, :, None].expand(B, P, S).reshape(B, P * S)
        if self.head_type == "more":
            t = self.mix(t, pad)
            moe_aux = self.mix.aux_loss.expand(B)          # [B]: DataParallel-safe (concatenable across GPUs)
        else:
            t = self.mix(t, src_key_padding_mask=pad)
            moe_aux = torch.zeros(B, device=x.device)
        q = self.nq(self.q).unsqueeze(0).expand(B, -1, -1)
        a, _ = self.xattn(q, self.nk(t), self.nk(t), key_padding_mask=pad)
        zl = ((q + a) * self.w).sum(-1) + self.b
        return 0.5 * (zg + zl), zg, zl, moe_aux

def make_model(pretrained):
    return Knee25DHybrid(CFG.BACKBONE, pretrained, len(CFG.PLANES), CFG.SLICES, CFG.D_MODEL,
                         head_type=CFG.HEAD_TYPE, recursions=CFG.MOR_RECURSIONS,
                         n_experts=CFG.MOE_EXPERTS, top_k=CFG.MOE_TOPK)

_m = make_model(False); print(f"params: {sum(p.numel() for p in _m.parameters()) / 1e6:.2f} M | head_type: {CFG.HEAD_TYPE}"); del _m"""))

C.append(md("## 5. 5-fold training (expert-labelled studies held out)"))
C.append(code(r"""AMP_DTYPE = torch.bfloat16 if DEVICE.type == "xla" else torch.float16  # TPUs are native bf16; no fp16 loss-scale headaches

def run_epoch(model, loader, opt=None, sched=None, scaler=None):
    train = opt is not None
    model.train(train); preds, tot, n = [], 0.0, 0
    for b in loader:
        x = b["x"].to(DEVICE, non_blocking=True).float().div_(255); m = b["m"].to(DEVICE)
        if train: x = gpu_augment(x)
        def fwd():
            with torch.set_grad_enabled(train):
                return model(x, m)
        amp_on = CFG.AMP and DEVICE.type in ("cuda", "xla")
        if amp_on:
            try:
                with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=True):
                    z, zg, zl, moe_aux = fwd()
            except Exception as e:
                # Self-heal instead of crashing the run: an older torch_xla may not register "xla" as an
                # autocast device_type. Disable AMP for the rest of the run and redo this batch in fp32.
                print(f"autocast unavailable on {DEVICE.type} ({e!r}); disabling AMP for the rest of the run.")
                CFG.AMP = False
                z, zg, zl, moe_aux = fwd()
        else:
            z, zg, zl, moe_aux = fwd()
        if train:
            y = b["y"].to(DEVICE)
            loss = F.binary_cross_entropy_with_logits(z.float(), y) + CFG.AUX_W * (
                F.binary_cross_entropy_with_logits(zg.float(), y) + F.binary_cross_entropy_with_logits(zl.float(), y)
            ) + moe_aux.float().mean()          # MoE load-balance + router z-loss (no-op, 0, for HEAD_TYPE="transformer")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt); scaler.update(); sched.step()
            if DEVICE.type == "xla": xm.mark_step()     # execute the queued graph and advance the TPU
            tot += loss.item() * len(y); n += len(y)
        else:
            preds.append(torch.sigmoid(z.float()).cpu().numpy())
    return tot / max(n, 1) if train else np.concatenate(preds)

def loader(idx, y=None, train=False):
    ds = KneeDS(VOL, MASK, idx, y, train)
    return torch.utils.data.DataLoader(ds, batch_size=CFG.BATCH if train else CFG.BATCH * 2, shuffle=train,
                                       num_workers=CFG.NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"), drop_last=train)

def save_state(state_dict, path):
    (xm.save if DEVICE.type == "xla" else torch.save)(state_dict, path)   # xm.save moves XLA tensors to CPU first

if CFG.MODE == "train":
    pool = np.where(~gold_mask)[0]; gold = np.where(gold_mask)[0]
    Yp = SOFT[pool]; Yg = train.loc[gold_mask, LABELS].values.astype(int)
    strat = np.clip((Yp >= 0.5).sum(1), 0, 5) * 2 + (Yp[:, 0] >= 0.5)
    folds = np.zeros(len(pool), int)
    for f, (_, te) in enumerate(StratifiedKFold(CFG.FOLDS, shuffle=True, random_state=CFG.SEED).split(pool, strat)):
        folds[te] = f
    oof = np.full(Yp.shape, np.nan, np.float32); gold_preds = []; fold_times = []
    for f in CFG.TRAIN_FOLDS:
        t = time.time(); seed_all(CFG.SEED + f)
        tr, va = pool[folds != f], pool[folds == f]
        model = make_model(CFG.PRETRAINED).to(DEVICE)
        if N_GPU > 1: model = nn.DataParallel(model)
        core = model.module if hasattr(model, "module") else model
        enc_p = list(core.enc.parameters()); enc_ids = {id(p) for p in enc_p}
        head_p = [p for p in core.parameters() if id(p) not in enc_ids]
        opt = torch.optim.AdamW([{"params": enc_p, "lr": CFG.LR_BACKBONE}, {"params": head_p, "lr": CFG.LR_HEAD}],
                                weight_decay=CFG.WD)
        dl = loader(tr, Yp[folds != f], train=True)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[CFG.LR_BACKBONE, CFG.LR_HEAD],
                                                    total_steps=CFG.EPOCHS * len(dl), pct_start=0.1)
        scaler = torch.amp.GradScaler(enabled=CFG.AMP and DEVICE.type == "cuda")   # bf16/XLA needs no loss scaling
        for ep in range(CFG.EPOCHS):
            loss = run_epoch(model, dl, opt, sched, scaler)
            msg = f"fold {f} ep {ep + 1}/{CFG.EPOCHS} loss {loss:.4f}"
            if ep == CFG.EPOCHS - 1 or (ep + 1) % 4 == 0:
                pv = run_epoch(model, loader(va))
                yv = Yp[folds == f]; msg += f" | val macroAUC(report labels) {macro_auc((yv >= 1).astype(int), pv, yv != 0.5):.4f}"
            print(msg, elapsed())
        oof[folds == f] = run_epoch(model, loader(va))
        gold_preds.append(run_epoch(model, loader(gold)))
        save_state(core.state_dict(), f"{CFG.OUT_DIR}/fold{f}.pt")
        fold_times.append(time.time() - t)
        print(f"== fold {f}: gold macroAUC {macro_auc(Yg, gold_preds[-1]):.4f}")
        del model, opt; torch.cuda.empty_cache()
    json.dump({k: v for k, v in vars(CFG).items() if not k.startswith("_") and isinstance(v, (int, float, str, list, bool, type(None)))},
              open(f"{CFG.OUT_DIR}/config.json", "w"), indent=1)"""))

C.append(md(r"""## 6. Evaluation: macro AUC, accuracy, per-label metrics (all measured)

* **OOF.** Scored against the training labels, excluding uncertain (0.5) entries.
* **Expert set (58 studies).** Uses the fold-ensemble predictions:
  * macro AUC, with a 95% bootstrap CI;
  * **accuracy** at per-label thresholds fitted by leave-one-out (fit on 57 studies, apply to the 58th);
  * the "always negative" accuracy, as a floor for reference."""))
C.append(code(r"""def loo_accuracy(y, p):
    n, L = y.shape; hit = np.zeros((n, L), bool)
    for k in range(L):
        cand = np.unique(np.r_[p[:, k], 1.01])
        for i in range(n):
            mm = np.arange(n) != i
            t = cand[np.argmax([((p[mm, k] >= c) == y[mm, k]).mean() for c in cand])]
            hit[i, k] = (p[i, k] >= t) == y[i, k]
    return hit.mean(), hit.mean(0)

if CFG.MODE == "train":
    done = ~np.isnan(oof).any(1)
    oof_auc = macro_auc((Yp[done] >= 1).astype(int), oof[done], Yp[done] != 0.5)
    G = np.mean(gold_preds, 0)
    g_auc = macro_auc(Yg, G)
    rng = np.random.default_rng(0); boots = []
    for _ in range(1000):
        i = rng.integers(0, len(Yg), len(Yg)); boots.append(macro_auc(Yg[i], G[i]))
    acc, acc_lab = loo_accuracy(Yg, G)
    per = pd.DataFrame({"gold_auc": [macro_auc(Yg[:, k:k + 1], G[:, k:k + 1]) for k in range(12)],
                        "gold_acc_LOO": acc_lab,
                        "oof_auc": [macro_auc((Yp[done, k:k + 1] >= 1).astype(int), oof[done, k:k + 1], Yp[done, k:k + 1] != 0.5) for k in range(12)]},
                       index=LABELS).round(4)
    summary = {"label_source": LABEL_SOURCE, "labeler_gold_auc": scores[LABEL_SOURCE],
               "oof_macro_auc": oof_auc, "gold_macro_auc": g_auc,
               "gold_auc_95ci": list(np.nanpercentile(boots, [2.5, 97.5])),
               "gold_accuracy_LOO_thresholds": acc, "gold_accuracy_all_negative": float((Yg == 0).mean()),
               "folds_trained": CFG.TRAIN_FOLDS, "device": str(DEVICE)}
    json.dump(summary, open(f"{CFG.OUT_DIR}/metrics.json", "w"), indent=1, default=float)
    print(json.dumps(summary, indent=1, default=float)); display(per)"""))

C.append(md("## 7. Test inference -> `submission.csv`"))
C.append(code(r"""t_inf = time.time()
test = pd.read_csv(f"{DATA}/test.csv")
wdir = CFG.OUT_DIR if CFG.MODE == "train" else CFG.WEIGHTS_DIR
if CFG.MODE == "infer" and not os.path.exists(f"{wdir}/config.json"):
    found = find_kaggle_path(os.path.basename(wdir.rstrip("/")), "fold*.pt")   # WEIGHTS_DIR didn't exist as configured
    if found: wdir = found; print("WEIGHTS_DIR not found as configured; auto-detected:", wdir)
if CFG.MODE == "infer" and os.path.exists(f"{wdir}/config.json"):
    for k, v in json.load(open(f"{wdir}/config.json")).items():
        if k in ("BACKBONE", "SLICES", "IMG", "D_MODEL", "PLANES"): setattr(CFG, k, v)
TVOL, TMASK = build_cache("test", test.StudyInstanceUID.tolist())
paths = sorted(glob.glob(f"{wdir}/fold*.pt")); assert paths, f"no fold*.pt in {wdir}"
VOL, MASK = TVOL, TMASK
preds = []
for p in paths:
    model = make_model(False)
    model.load_state_dict(torch.load(p, map_location="cpu"))   # always land on CPU first, then move to DEVICE
    model = model.to(DEVICE); model.eval()
    preds.append(run_epoch(model, loader(np.arange(len(test)))))
    del model
P = np.mean(preds, 0)
sub = pd.DataFrame(P, columns=LABELS); sub.insert(0, "StudyInstanceUID", test.StudyInstanceUID.values)
ss = pd.read_csv(f"{DATA}/sample_submission.csv"); sub = sub[ss.columns]
sub.to_csv("submission.csv", index=False)
print(f"submission.csv: {sub.shape} | {len(paths)} fold models")
sub.head()"""))

nb = nbf.v4.new_notebook(cells=C)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
# Best-effort hint for the web editor; Kaggle's TPU accelerator selection via API/metadata is not fully
# reliable (see notebook markdown), so Settings -> Accelerator -> TPU VM v3-8 must still be picked by hand.
nb.metadata["kaggle"] = {"accelerator": "tpu", "isInternetEnabled": True, "language": "python"}
out = ROOT / "notebooks/rsna_knee_kaggle_tpu_v2.ipynb"
nbf.write(nb, out)
print("wrote", out)
