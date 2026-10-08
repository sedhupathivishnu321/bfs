"""Builds notebooks/rsna_knee_kaggle_tpu_v2.ipynb, a self-contained Kaggle notebook (TPU v3-8).

    python notebooks/build_kaggle_notebook_tpu_v2.py

Model: an enhanced logistic-regression classifier (one per label) on frozen, pretrained-CNN features,
in place of the earlier MV-MoRE/transformer hybrid head -- see the notebook's own "Model" markdown for
why and what "enhanced" means concretely. The TPU/XLA device handling, Kaggle input-path detection and
labeling pipeline are carried over unchanged from the prior revision of this notebook; a frozen-feature
forward pass has none of the backward-pass dynamic-shape concerns that motivated most of the earlier
TPU-specific rewrites, so that part of the story is now much simpler. The rule-based multilingual report
labeler is embedded verbatim from src/kneemor/report_labeler.py via %%writefile, so the notebook has no
dependency on this repository at run time.
"""
import pathlib

import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parents[1]
LABELER_SRC = (ROOT / "src/kneemor/report_labeler.py").read_text()

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
C = []

C.append(md(r"""# RSNA Knee Abnormality Detection: enhanced logistic regression on frozen CNN features — V2 (TPU)

This is a self-contained Kaggle notebook for the **TPU v3-8** accelerator: open **Settings -> Accelerator**
and pick **TPU VM v3-8**, turn **Internet** on, then **Run All**.

**Pipeline**
1. **Labels.** 58 training studies carry expert labels; the other ~4,350 have only a free-text report in
   one of eight languages. Two labelers turn those reports into training labels:
   * a multilingual rule labeler;
   * an *optional* local LLM labeler (attach a Qwen2.5-Instruct model) -- GPU/CPU only, see below.

   The notebook scores **both** on the 58 expert studies and trains on whichever agrees better.
2. **Preprocessing.** For each plane (sagittal, coronal, axial) it takes the best fluid-sensitive series,
   sorts it along the slice normal, resamples it to `SLICES` slices at `IMG` px, and caches the volume as
   uint8.
3. **Model: enhanced logistic regression on frozen CNN features.** See "Model" below for the full design
   and why it replaced an earlier fine-tuned 2.5-D hybrid CNN with an MV-MoRE/transformer head.
4. **Fitting.** 5-fold CV on the report-labelled studies. The 58 expert-labelled studies are **held out**
   and used only for evaluation.
5. **Evaluation.** OOF macro AUC; expert-set macro AUC with a bootstrap CI; expert-set **accuracy** with
   leave-one-out thresholds; per-label table; timings.
6. **Submission.** A fold ensemble writes `submission.csv`.

## Model: why logistic regression, and what "enhanced" means here

An earlier revision of this notebook fine-tuned a 2.5-D CNN end-to-end under an MV-MoRE/transformer head.
This revision replaces that head with **logistic regression** -- deliberately simpler, and a better fit
for what this dataset actually supports: only 58 studies carry expert labels, the rest are weak/noisy
report-derived labels, and a model with hundreds of thousands of trainable parameters fit on that signal is
exactly the regime where a much smaller, regularised linear classifier tends to generalise at least as
well. The repo's own `results/summary.csv` bears this out: the existing plain logistic-regression baseline
lands within a few points of gold macro AUC of every deep variant tried, including the fine-tuned hybrid
CNN. "Enhanced" here means genuinely improving the *existing* classical baseline
(`src/kneemor/baseline_lr.py`: per-plane mean-pooled features -> `StandardScaler` -> a single fixed-`C`
`LogisticRegression` per label), not just renaming it:

* **Frozen features, extracted once.** The ImageNet-pretrained CNN (`CFG.BACKBONE`) is *not* fine-tuned --
  its weights never change, so unlike per-fold backprop training, features are extracted **once** for
  every study and cached (`extract_features`, keyed by a cache tag), then reused across all 5 folds. This
  is a large efficiency win, and it also removes an entire class of TPU complexity: a forward-only pass has
  no backward-pass dynamic-shape or gradient-dtype concerns at all.
* **Richer pooling.** Each study is pooled to one feature vector per plane as **mean *and* standard
  deviation** of the per-(3-slice-window) embedding over the plane's slices (`2 x C` per plane, `3 x 2 x C`
  total), instead of mean alone -- the spread across slices carries information the mean discards (e.g. a
  focal finding visible on only a few slices raises the std without moving the mean much). A missing plane
  still contributes exactly zero, as elsewhere in this notebook.
* **Class-balanced, per-label regularisation search.** Each of the 12 labels gets its own
  `LogisticRegressionCV` (`scikit-learn`) with `class_weight="balanced"` (several labels are rare) and an
  L2 strength `C` chosen from `CFG.LR_C_GRID` by that label's **own inner cross-validation on the training
  fold only** -- the search never sees the validation or gold studies, so nothing about picking `C` can
  leak across the evaluation boundary. `StandardScaler` is still applied first, as in the baseline.
* **Uncertain (0.5) labels excluded per label at fit time**, the same convention the evaluation already
  uses, rather than forcing them to a side with a `>=0.5` threshold as the plain baseline implicitly did.
* **Crash-safe fallback.** If a training fold has too few examples of one class for its label (common for
  the rarest findings, especially in a small or `MAX_TRAIN_STUDIES`-limited run), the inner CV is shrunk to
  fit, or -- if a fold has only one class for that label at all -- the label falls back to predicting that
  fold's observed base rate rather than raising `scikit-learn`'s "needs samples of at least 2 classes" error.

## What changed for TPU (and why)

The only TPU-relevant step left is `extract_features`' forward pass through the frozen encoder -- there is
no training loop, no optimizer, no gradients; logistic regression itself always runs on `scikit-learn`/CPU.
Every change below is scoped to `DEVICE.type == "xla"`; running on GPU/CPU is unaffected:

* **Device selection.** `xm.xla_device()` is used when `torch_xla` is importable *and* a TPU actually
  answers a trivial op; otherwise it falls back to CUDA, then CPU. A missing/misconfigured accelerator
  degrades gracefully instead of crashing the whole run.
* **bf16 autocast for the forward pass, with a self-heal.** `torch.autocast(device_type="xla",
  dtype=torch.bfloat16)` speeds up feature extraction on TPU (bf16 is its native compute type). Since this
  is inference-only, there's no gradient-scaling question at all; if autocast still raises on an older
  `torch_xla`, the notebook catches it once, prints a warning, and continues in fp32.
* **No checkpoint-serialisation concerns at all.** The fitted per-label classifiers are plain
  `scikit-learn` objects (CPU `numpy` arrays under the hood) pickled with Python's `pickle` -- there's no
  XLA tensor to move off-device first, unlike a `torch.save`/`xm.save` question for a trained `nn.Module`.
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

## What this model can honestly achieve (please read before trusting any number below)

Nothing in this notebook *guarantees* a particular accuracy or AUC; it **measures and prints** what it
achieves, and that measurement has a hard, dataset-set ceiling that no architecture change -- logistic
regression included -- moves:

* The rule labeler that produces most training labels agrees with the 58 expert-labelled studies at about
  **0.735 AUC**. Models trained on its output cannot legitimately score better than that on labels drawn
  from the same noisy process; the optional LLM labeler exists to try to raise this cap, not to guarantee it.
* The expert ("gold") set used for the headline evaluation has only **58 studies** (a 95% bootstrap CI of
  roughly +/-0.06 AUC on the macro score).
* Across *every* model architecture measured so far in this repository -- the fine-tuned hybrid CNN, its
  MV-MoRE and transformer heads, mean-pool MLP, ABMIL, gradient boosting, and the existing plain
  logistic-regression baseline -- measured **gold macro accuracy has landed in the ~0.53-0.63 range and
  per-label gold AUC in ~0.50-0.82** (see `results/summary.csv`, `results/per_label_auc.csv`). None of them
  are close to 99%, and there is no honest way to get there on this benchmark: not by swapping in logistic
  regression, and not by any other architecture change. A number that high on this task would mean the
  evaluation leaked (e.g. the same studies used for both fitting and scoring, or a threshold tuned on the
  set it's then measured on) -- this notebook's folds and the expert-set holdout are built specifically to
  rule that out, and it reports whatever macro AUC/accuracy actually comes out the other end, not a target.

**Two ways to run:**
* `MODE="train"` (internet ON): fits, evaluates and writes the classifiers and a submission.
* `MODE="infer"` (internet OFF): set `WEIGHTS_DIR` to a Kaggle dataset containing the saved `fold*.pkl` and
  `config.json`."""))

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
    # frozen feature extractor (NOT fine-tuned -- see "Model" markdown above)
    BACKBONE = "tf_efficientnet_b0.ns_jft_in1k"         # alternatives: "convnext_nano.in12k_ft_in1k", "resnet34.a1_in1k"
    PRETRAINED = True              # needs internet in MODE="train"
    FEAT_BATCH = 8                 # studies per forward pass through the frozen encoder
    # classifier: enhanced logistic regression, one independent binary model per label
    FOLDS = 5
    TRAIN_FOLDS = [0, 1, 2, 3, 4]  # subset to save time, e.g. [0]
    LR_C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]   # inverse L2-strength candidates (LogisticRegressionCV)
    LR_INNER_CV = 3                # folds for the leakage-safe C search, done within the training fold only
    LR_MAX_ITER = 4000
    NUM_WORKERS = 4                # forced to 0 on TPU/XLA in the next cell -- see "What changed for TPU"
    AMP = True                     # bf16/fp16 autocast for the frozen-encoder forward pass (no gradients -> always safe)
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
    CFG.FOLDS, CFG.TRAIN_FOLDS, CFG.FEAT_BATCH, CFG.NUM_WORKERS = 2, [0, 1], 2, 0
    CFG.LR_C_GRID, CFG.LR_INNER_CV = [0.1, 1.0], 2
    CFG.MODE = os.environ.get("KNEE_MODE", "train"); CFG.WEIGHTS_DIR = os.environ.get("KNEE_WEIGHTS", CFG.OUT_DIR)
os.makedirs(CFG.CACHE_DIR, exist_ok=True); os.makedirs(CFG.OUT_DIR, exist_ok=True)
print({k: v for k, v in vars(CFG).items() if not k.startswith("_")})"""))

C.append(code(r"""import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import cv2, pydicom, timm, pickle
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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

C.append(md("## 4. Frozen CNN features & enhanced logistic regression"))
C.append(code(r"""class KneeDS(torch.utils.data.Dataset):
    # Deterministic: the encoder is frozen (not fine-tuned), so every study's features only need
    # computing once, the same way for train/val/gold/test -- no train-time augmentation flag needed.
    def __init__(self, vol, mask, idx):
        self.vol, self.mask, self.idx = vol, mask, idx
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        v = torch.from_numpy(np.array(self.vol[j]))                # P,S,H,W uint8
        m = torch.from_numpy(self.mask[j].copy())
        return {"x": v, "m": m, "i": i}


AMP_DTYPE = torch.bfloat16 if DEVICE.type == "xla" else torch.float16  # TPUs are native bf16

MEAN_RGB = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
STD_RGB = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)

def make_encoder():
    # Frozen: an ImageNet-pretrained CNN used purely as a feature extractor. Nothing here is trained --
    # see the "Model" markdown above for why a fine-tuned CNN was replaced by this + logistic regression.
    enc = timm.create_model(CFG.BACKBONE, pretrained=CFG.PRETRAINED, num_classes=0, in_chans=3).to(DEVICE).eval()
    for p in enc.parameters(): p.requires_grad_(False)
    return enc

@torch.no_grad()
def extract_features(enc, vol, mask, tag):
    '''Per study, per plane: mean AND std (over the plane's slices) of the 3-slice-window embedding --
    richer than mean alone (see "Model" markdown). Cached to CFG.CACHE_DIR by `tag` since the encoder
    never changes across folds -- unlike backprop training, there is nothing fold-specific to recompute.'''
    path = f"{CFG.CACHE_DIR}/{tag}_feat.npy"
    if os.path.exists(path):
        return np.load(path)
    P, S, C = len(CFG.PLANES), CFG.SLICES, enc.num_features
    feats = np.zeros((len(vol), P, 2 * C), np.float32)
    dl = torch.utils.data.DataLoader(KneeDS(vol, mask, np.arange(len(vol))), batch_size=CFG.FEAT_BATCH,
                                     shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"))
    amp_on = CFG.AMP and DEVICE.type in ("cuda", "xla")
    t = time.time()
    for b in dl:
        x = b["x"].to(DEVICE, non_blocking=True).float().div_(255); m = b["m"].to(DEVICE); idx = b["i"].numpy()
        B, P_, S_, H, W = x.shape
        xp = torch.cat([x[:, :, :1], x, x[:, :, -1:]], 2)                     # replicate-pad slices
        rgb = torch.stack([xp[:, :, i2:i2 + S_] for i2 in range(3)], 3)       # B,P,S,3,H,W
        rgb = (rgb.reshape(-1, 3, H, W) - MEAN_RGB) / STD_RGB
        def fwd(): return enc(rgb).float().view(B, P_, S_, -1)
        if amp_on:
            try:
                with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=True):
                    f = fwd()
            except Exception as e:
                # Self-heal instead of crashing the run: an older torch_xla may not register "xla" as an
                # autocast device_type. Disable AMP for the rest of the run and redo this batch in fp32.
                print(f"autocast unavailable on {DEVICE.type} ({e!r}); extracting features in fp32 instead.")
                CFG.AMP, amp_on = False, False
                f = fwd()
        else:
            f = fwd()
        fm = f * m[..., None].float()[..., None]                              # zero out missing planes
        pooled = torch.cat([fm.mean(2), fm.std(2)], -1)                       # B,P,2C
        feats[idx] = pooled.cpu().numpy()
        if DEVICE.type == "xla": xm.mark_step()
    feats = feats.reshape(len(vol), -1)
    np.save(path, feats)
    print(f"  features cached: {tag} {feats.shape} in {time.time() - t:.0f}s")
    return feats

def predict_proba_pos(clf, X):
    return clf.predict_proba(X)[:, 1]

class ConstantProba:
    '''Fallback for a training fold with only one class present for a label (too rare to fit a real
    classifier on): predicts that fold's observed base rate for every study, rather than letting
    sklearn raise "This solver needs samples of at least 2 classes" and crash the whole run.'''
    def __init__(self, p): self.p = float(p)
    def predict_proba(self, X):
        return np.tile([1 - self.p, self.p], (len(X), 1))

def fit_label_classifier(Xtr, ytr):
    '''Enhanced logistic regression for one label: StandardScaler -> class-balanced LogisticRegressionCV,
    with its L2 strength chosen from CFG.LR_C_GRID by cross-validation *within this training fold only*
    (never touching validation/gold -- see "Model" markdown). Falls back to a fixed-C LogisticRegression
    if the inner CV can't run (too few examples of one class for the requested number of folds), and to
    ConstantProba if the fold has only one class for this label at all.'''
    ytr = np.asarray(ytr)
    if len(np.unique(ytr)) < 2:
        return ConstantProba(ytr.mean())
    n_pos, n_neg = int(ytr.sum()), int(len(ytr) - ytr.sum())
    cv = max(2, min(CFG.LR_INNER_CV, n_pos, n_neg))
    try:
        clf = make_pipeline(StandardScaler(), LogisticRegressionCV(
            Cs=CFG.LR_C_GRID, cv=cv, class_weight="balanced", max_iter=CFG.LR_MAX_ITER, scoring="roc_auc"))
        clf.fit(Xtr, ytr)
        return clf
    except Exception as e:
        print(f"LogisticRegressionCV failed ({e!r}); falling back to a fixed-C logistic regression.")
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=CFG.LR_MAX_ITER))
        clf.fit(Xtr, ytr)
        return clf

def save_models(models, path):
    with open(path, "wb") as f: pickle.dump(models, f)

def load_models(path):
    with open(path, "rb") as f: return pickle.load(f)

_e = make_encoder(); print(f"frozen encoder params: {sum(p.numel() for p in _e.parameters()) / 1e6:.2f} M (not trained -- see Model markdown)"); del _e"""))

C.append(md("## 5. 5-fold logistic-regression fitting (expert-labelled studies held out)"))
C.append(code(r"""if CFG.MODE == "train":
    pool = np.where(~gold_mask)[0]; gold = np.where(gold_mask)[0]
    Yp = SOFT[pool]; Yg = train.loc[gold_mask, LABELS].values.astype(int)
    strat = np.clip((Yp >= 0.5).sum(1), 0, 5) * 2 + (Yp[:, 0] >= 0.5)
    folds = np.zeros(len(pool), int)
    for f, (_, te) in enumerate(StratifiedKFold(CFG.FOLDS, shuffle=True, random_state=CFG.SEED).split(pool, strat)):
        folds[te] = f

    enc = make_encoder()
    FEAT = extract_features(enc, VOL, MASK, f"train_{CFG.SLICES}x{CFG.IMG}_{len(VOL)}")
    del enc
    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    print("features ready", FEAT.shape, elapsed())
    Fp, Fg = FEAT[pool], FEAT[gold]

    oof = np.full(Yp.shape, np.nan, np.float32); gold_preds = []; fold_times = []
    for f in CFG.TRAIN_FOLDS:
        t = time.time(); seed_all(CFG.SEED + f)
        tr, va = folds != f, folds == f
        gp = np.zeros(Yg.shape, np.float32); models = {}
        for k, lab in enumerate(LABELS):
            mtr = Yp[tr, k] != 0.5                                     # exclude uncertain labels from fitting
            ytr = (Yp[tr, k][mtr] >= 1).astype(int)
            clf = fit_label_classifier(Fp[tr][mtr], ytr)
            oof[va, k] = predict_proba_pos(clf, Fp[va])
            gp[:, k] = predict_proba_pos(clf, Fg)
            models[lab] = clf
        gold_preds.append(gp)
        save_models(models, f"{CFG.OUT_DIR}/fold{f}.pkl")
        fold_times.append(time.time() - t)
        print(f"fold {f}: gold macroAUC {macro_auc(Yg, gp):.4f}", elapsed())
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
    found = find_kaggle_path(os.path.basename(wdir.rstrip("/")), "fold*.pkl")   # WEIGHTS_DIR didn't exist as configured
    if found: wdir = found; print("WEIGHTS_DIR not found as configured; auto-detected:", wdir)
if CFG.MODE == "infer" and os.path.exists(f"{wdir}/config.json"):
    for k, v in json.load(open(f"{wdir}/config.json")).items():
        if k in ("BACKBONE", "SLICES", "IMG", "PLANES"): setattr(CFG, k, v)
TVOL, TMASK = build_cache("test", test.StudyInstanceUID.tolist())
paths = sorted(glob.glob(f"{wdir}/fold*.pkl")); assert paths, f"no fold*.pkl in {wdir}"
enc = make_encoder()
Ftest = extract_features(enc, TVOL, TMASK, f"test_{CFG.SLICES}x{CFG.IMG}_{len(TVOL)}")
del enc
if DEVICE.type == "cuda": torch.cuda.empty_cache()
preds = []
for p in paths:
    models = load_models(p)
    pf = np.zeros((len(test), len(LABELS)), np.float32)
    for k, lab in enumerate(LABELS):
        pf[:, k] = predict_proba_pos(models[lab], Ftest)
    preds.append(pf)
P = np.mean(preds, 0)
sub = pd.DataFrame(P, columns=LABELS); sub.insert(0, "StudyInstanceUID", test.StudyInstanceUID.values)
ss = pd.read_csv(f"{DATA}/sample_submission.csv"); sub = sub[ss.columns]
sub.to_csv("submission.csv", index=False)
print(f"submission.csv: {sub.shape} | {len(paths)} fold models | {time.time() - t_inf:.0f}s")
sub.head()"""))

nb = nbf.v4.new_notebook(cells=C)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
# Best-effort hint for the web editor; Kaggle's TPU accelerator selection via API/metadata is not fully
# reliable (see notebook markdown), so Settings -> Accelerator -> TPU VM v3-8 must still be picked by hand.
nb.metadata["kaggle"] = {"accelerator": "tpu", "isInternetEnabled": True, "language": "python"}
out = ROOT / "notebooks/rsna_knee_kaggle_tpu_v2.ipynb"
nbf.write(nb, out)
print("wrote", out)
