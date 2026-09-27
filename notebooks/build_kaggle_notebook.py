"""Builds notebooks/rsna_knee_kaggle.ipynb, a self-contained Kaggle notebook (GPU).

    python notebooks/build_kaggle_notebook.py

The rule-based multilingual report labeler is embedded verbatim from src/kneemor/report_labeler.py
via %%writefile, so the notebook has no dependency on this repository at run time.
"""
import pathlib

import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parents[1]
LABELER_SRC = (ROOT / "src/kneemor/report_labeler.py").read_text()

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
C = []

C.append(md(r"""# RSNA Knee Abnormality Detection: 2.5-D hybrid CNN (train + infer)

This is a self-contained Kaggle notebook: turn on a **GPU** (T4×2 or P100) and **Run All**.

**Pipeline**
1. **Labels.** 58 training studies carry expert labels; the other ~4,350 have only a free-text report in
   one of eight languages. Two labelers turn those reports into training labels:
   * a multilingual rule labeler;
   * an *optional* local LLM labeler (attach a Qwen2.5-Instruct model).

   The notebook scores **both** on the 58 expert studies and trains on whichever agrees better.
2. **Preprocessing.** For each plane (sagittal, coronal, axial) it takes the best fluid-sensitive series,
   sorts it along the slice normal, resamples it to `SLICES` slices at `IMG` px, and caches the volume as
   uint8.
3. **Model: 2.5-D hybrid.**
   * An ImageNet-pretrained timm CNN runs on 3 adjacent slices as RGB and is **fine-tuned end-to-end**.
   * A **global branch** (per-plane mean → linear) feeds the output directly.
   * A **local branch** (2-layer transformer over all slice tokens of all planes, then 12 label queries
     with cross-attention) feeds it too.
   * Each branch has its own auxiliary loss.
4. **Training.** 5-fold CV on the report-labelled studies with AMP, a cosine schedule and soft BCE. The 58
   expert-labelled studies are **held out** and used only for evaluation.
5. **Evaluation.** OOF macro AUC; expert-set macro AUC with a bootstrap CI; expert-set **accuracy** with
   leave-one-out thresholds; per-label table; timings.
6. **Submission.** A fold ensemble writes `submission.csv`.

**Targets vs. guarantees.** The goal is as high a macro AUC and accuracy as possible at low runtime. Nothing
here *guarantees* a particular number such as 98% accuracy; the notebook **measures and prints** what it
achieves.

Two effects bound what any score can mean:
* In the author's CPU study, rule labels agreed with the expert labels at about 0.735 AUC, which caps
  models trained on them. The LLM labeler exists to raise that cap.
* The expert set has only 58 studies (95% CI about ±0.06 AUC).

**Two ways to run:**
* `MODE="train"` (internet ON): trains, evaluates and writes the weights and a submission.
* `MODE="infer"` (for the code-competition submission, internet OFF): set `WEIGHTS_DIR` to a Kaggle dataset
  containing the saved `fold*.pt` and `config.json`."""))

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
    # training
    FOLDS = 5
    TRAIN_FOLDS = [0, 1, 2, 3, 4]  # subset to save time, e.g. [0]
    EPOCHS = 12
    BATCH = 4                      # studies per step (each study = 3*SLICES images)
    LR_BACKBONE = 2e-4
    LR_HEAD = 1e-3
    WD = 1e-2
    AUX_W = 0.5                    # deep-supervision weight for each branch
    NUM_WORKERS = 4
    AMP = True
    # labels
    USE_LLM_LABELS = False         # True -> attach a Qwen2.5-Instruct model and set LLM_PATH
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPU = torch.cuda.device_count()
print("device:", DEVICE, "| GPUs:", N_GPU, "| torch", torch.__version__, "| timm", timm.__version__)

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
Kaggle Models, then point `LLM_PATH` at it.

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
    Yg = train.loc[gold_mask, LABELS].values.astype(int)
    scores = {k: macro_auc(Yg, v[gold_mask]) for k, v in candidates.items()}
    print("labeler macro AUC vs expert labels:", {k: round(v, 4) for k, v in scores.items()})
    LABEL_SOURCE = max(scores, key=scores.get)
    SOFT = candidates[LABEL_SOURCE].astype(np.float32)
    print("-> training labels from:", LABEL_SOURCE)"""))

C.append(md("## 3. DICOM → cached volumes"))
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
    # x: B,P,S,H,W float in [0,1]. Intensity + small affine; NO flips (would swap medial/lateral or ant/post).
    B, P = x.shape[:2]
    g = torch.empty(B, P, 1, 1, 1, device=x.device).uniform_(0.7, 1.4)
    x = x.clamp(0, 1) ** g
    x = x * torch.empty(B, P, 1, 1, 1, device=x.device).uniform_(0.85, 1.15) + torch.empty(B, P, 1, 1, 1, device=x.device).uniform_(-0.08, 0.08)
    th = torch.zeros(B * P, 2, 3, device=x.device)
    sc = torch.empty(B * P, device=x.device).uniform_(0.9, 1.1); ang = torch.empty(B * P, device=x.device).uniform_(-0.15, 0.15)
    th[:, 0, 0] = sc * torch.cos(ang); th[:, 0, 1] = -sc * torch.sin(ang); th[:, 1, 0] = sc * torch.sin(ang); th[:, 1, 1] = sc * torch.cos(ang)
    th[:, :, 2] = torch.empty(B * P, 2, device=x.device).uniform_(-0.08, 0.08)
    S, H, W = x.shape[2:]
    grid = F.affine_grid(th, (B * P, S, H, W), align_corners=False)
    x = F.grid_sample(x.reshape(B * P, S, H, W), grid, align_corners=False, padding_mode="zeros")
    return x.reshape(B, P, S, H, W).clamp(0, 1)

class Knee25DHybrid(nn.Module):
    '''2.5-D CNN (3 adjacent slices -> RGB) + global branch + local transformer/label-query branch.'''
    def __init__(self, backbone, pretrained, P, S, d=256, n=12):
        super().__init__()
        self.enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0, in_chans=3)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        C = self.enc.num_features
        self.glob = nn.Sequential(nn.LayerNorm(P * C), nn.Dropout(0.2), nn.Linear(P * C, n))
        self.proj = nn.Sequential(nn.LayerNorm(C), nn.Linear(C, d))
        self.plane = nn.Parameter(torch.zeros(P, 1, d)); self.pos = nn.Parameter(torch.zeros(P, S, d))
        nn.init.trunc_normal_(self.plane, std=0.02); nn.init.trunc_normal_(self.pos, std=0.02)
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
        t = self.mix(t, src_key_padding_mask=pad)
        q = self.nq(self.q).unsqueeze(0).expand(B, -1, -1)
        a, _ = self.xattn(q, self.nk(t), self.nk(t), key_padding_mask=pad)
        zl = ((q + a) * self.w).sum(-1) + self.b
        return 0.5 * (zg + zl), zg, zl

def make_model(pretrained):
    return Knee25DHybrid(CFG.BACKBONE, pretrained, len(CFG.PLANES), CFG.SLICES, CFG.D_MODEL)

_m = make_model(False); print(f"params: {sum(p.numel() for p in _m.parameters()) / 1e6:.2f} M"); del _m"""))

C.append(md("## 5. 5-fold training (expert-labelled studies held out)"))
C.append(code(r"""def run_epoch(model, loader, opt=None, sched=None, scaler=None):
    train = opt is not None
    model.train(train); preds, tot, n = [], 0.0, 0
    for b in loader:
        x = b["x"].to(DEVICE, non_blocking=True).float().div_(255); m = b["m"].to(DEVICE)
        if train: x = gpu_augment(x)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=CFG.AMP and DEVICE.type == "cuda"):
            with torch.set_grad_enabled(train):
                z, zg, zl = model(x, m)
        if train:
            y = b["y"].to(DEVICE)
            loss = F.binary_cross_entropy_with_logits(z.float(), y) + CFG.AUX_W * (
                F.binary_cross_entropy_with_logits(zg.float(), y) + F.binary_cross_entropy_with_logits(zl.float(), y))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(y); n += len(y)
        else:
            preds.append(torch.sigmoid(z.float()).cpu().numpy())
    return tot / max(n, 1) if train else np.concatenate(preds)

def loader(idx, y=None, train=False):
    ds = KneeDS(VOL, MASK, idx, y, train)
    return torch.utils.data.DataLoader(ds, batch_size=CFG.BATCH if train else CFG.BATCH * 2, shuffle=train,
                                       num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=train)

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
        scaler = torch.amp.GradScaler(enabled=CFG.AMP and DEVICE.type == "cuda")
        for ep in range(CFG.EPOCHS):
            loss = run_epoch(model, dl, opt, sched, scaler)
            msg = f"fold {f} ep {ep + 1}/{CFG.EPOCHS} loss {loss:.4f}"
            if ep == CFG.EPOCHS - 1 or (ep + 1) % 4 == 0:
                pv = run_epoch(model, loader(va))
                yv = Yp[folds == f]; msg += f" | val macroAUC(report labels) {macro_auc((yv >= 1).astype(int), pv, yv != 0.5):.4f}"
            print(msg, elapsed())
        oof[folds == f] = run_epoch(model, loader(va))
        gold_preds.append(run_epoch(model, loader(gold)))
        torch.save(core.state_dict(), f"{CFG.OUT_DIR}/fold{f}.pt")
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
               "folds_trained": CFG.TRAIN_FOLDS}
    json.dump(summary, open(f"{CFG.OUT_DIR}/metrics.json", "w"), indent=1, default=float)
    print(json.dumps(summary, indent=1, default=float)); display(per)"""))

C.append(md("## 7. Test inference → `submission.csv`"))
C.append(code(r"""t_inf = time.time()
test = pd.read_csv(f"{DATA}/test.csv")
wdir = CFG.OUT_DIR if CFG.MODE == "train" else CFG.WEIGHTS_DIR
if CFG.MODE == "infer" and os.path.exists(f"{wdir}/config.json"):
    for k, v in json.load(open(f"{wdir}/config.json")).items():
        if k in ("BACKBONE", "SLICES", "IMG", "D_MODEL", "PLANES"): setattr(CFG, k, v)
TVOL, TMASK = build_cache("test", test.StudyInstanceUID.tolist())
paths = sorted(glob.glob(f"{wdir}/fold*.pt")); assert paths, f"no fold*.pt in {wdir}"
VOL, MASK = TVOL, TMASK
preds = []
for p in paths:
    model = make_model(False).to(DEVICE); model.load_state_dict(torch.load(p, map_location=DEVICE)); model.eval()
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
nb.metadata["kaggle"] = {"accelerator": "gpu", "isInternetEnabled": True, "language": "python"}
out = ROOT / "notebooks/rsna_knee_kaggle.ipynb"
nbf.write(nb, out)
print("wrote", out)
