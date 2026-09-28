"""Builds notebooks/rsna_knee_kaggle_tpu_v2.ipynb, a self-contained Kaggle notebook (TPU v3-8).

    python notebooks/build_kaggle_notebook_tpu_v2.py

This is the TPU port of notebooks/build_kaggle_notebook.py / rsna_knee_kaggle.ipynb: same pipeline, model
and evaluation, with only what's needed for PyTorch/XLA changed (see the intro cell below for the full
list). The rule-based multilingual report labeler is embedded verbatim from src/kneemor/report_labeler.py
via %%writefile, so the notebook has no dependency on this repository at run time.
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
import cv2, pydicom, timm, socket
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

# ------------------------- pretrained-weight download: check once, degrade gracefully -------------------------
# timm.create_model(..., pretrained=True) fetches weights from huggingface.co. Kaggle TPU VM sessions have
# a separate worker VM whose outbound path to arbitrary hosts is less reliable than on GPU/CPU, even with
# Settings -> Internet -> On, and can fail with a DNS error ("Temporary failure in name resolution"); a
# retry bug in some huggingface_hub versions then turns that into a confusing "client has been closed"
# RuntimeError instead of a clean failure. Checked once, up front, instead of inside the 5-fold loop in
# cell 13, so a bad connection degrades to a random-init backbone (still trains and finishes) rather than
# crashing the run -- and rather than paying the connect timeout on every fold.
def hf_reachable(host="huggingface.co", timeout=5):
    try:
        socket.create_connection((host, 443), timeout=timeout).close()
        return True
    except OSError:
        return False
if CFG.PRETRAINED and not hf_reachable():
    print("huggingface.co unreachable from this session (DNS/network) -- Kaggle TPU sessions often can't "
          "reach arbitrary hosts even with Internet ON. Falling back to CFG.PRETRAINED = False (random-init "
          "backbone) so the run finishes; gold/OOF AUC from this run will reflect a random-init backbone, "
          "not the pretrained one. For real pretrained-weight numbers on TPU, cache the timm checkpoint "
          "into a Kaggle Dataset and point HF_HOME at it instead of relying on a live download.")
    CFG.PRETRAINED = False

def find_data_dir():
    if CFG.DATA_DIR: return CFG.DATA_DIR
    for p in sorted(glob.glob("/kaggle/input/*")) + sorted(glob.glob("/kaggle/input/*/*")):
        if os.path.exists(os.path.join(p, "test_series.csv")): return p
    raise FileNotFoundError("competition data not found under /kaggle/input")
DATA = find_data_dir(); print("data:", DATA)
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
T0 = time.time()
def elapsed(): return f"{(time.time() - T0) / 60:.1f} min" """))

# ============================== remaining cells are added incrementally, cell by cell ==============================

nb = nbf.v4.new_notebook(cells=C)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["kaggle"] = {"accelerator": "tpu", "isInternetEnabled": True, "language": "python"}
out = ROOT / "notebooks/rsna_knee_kaggle_tpu_v2.ipynb"
nbf.write(nb, out)
print("wrote", out)
