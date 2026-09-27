"""Gold-calibrated report labeler (v2).

v1 (report_labeler.py) is a fixed rule set: 0.735 macro AUC vs the 58 expert labels.
Its errors are systematic: it misses OA that is described as cartilage loss or chondropathy,
it over-calls trivial effusions, and it misses synovitis that is implied by other findings.
v2 therefore learns the *mapping* from report evidence to the expert label convention:

  features (fixed a priori, written from domain knowledge, not by reading gold reports)
    - the 12 v1 rule outputs
    - affirmed-mention counts for: cartilage/chondral lesion per compartment (medial/lateral/PF),
      osteophytes, bone-marrow oedema, effusion size words (small / moderate-large),
      meniscal degeneration/signal, Hoffa / synovial thickening, ligament signal/thickening, cysts
    - log report length
  model: per label, L2 logistic regression (C=0.3) on standardised features, fitted on the 58 gold studies.

Measurement is leakage-safe: leave-one-out over the gold studies (the held-out study's
report never influences its own prediction). The final model, refitted on all 58, is applied ONLY
to the 4,349 non-gold reports to produce v2 silver labels. Gold images/labels never enter image training.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kneemor.report_labeler import (COMP, LABELS, label_report, polarity, rx,  # noqa: E402
                                    sentences)

CART = rx(r"chondr", r"cartilag", r"condr", r"knorpel", r"kraakbeen", r"hrskav", r"kikird", r"хрущ", r"chondral")
CART_BAD = rx(r"loss", r"thinn", r"defect", r"fissur", r"ulcer", r"denud", r"erosi", r"perdida", r"adelgaz",
              r"ulcera", r"verlust", r"ausdunn", r"defekt", r"verlies", r"dunn", r"stanjen", r"gubit", r"defek",
              r"incelme", r"kayb", r"изтън", r"загуб", r"amincis", r"perte", r"grade (3|4|iii|iv)",
              r"grad (3|4|iii|iv)", r"full[- ]thickness", r"chondropath", r"chondromalac", r"condromalac",
              r"condropat", r"hondromalac", r"kondromalaz", r"хондромалац")
OSTEO = rx(r"osteophyt", r"osteofit", r"остеофит", r"spur")
BME = rx(r"marrow (o)?edema", r"bone (o)?edema", r"edema oseo", r"medula osea", r"knochenmark\w*odem",
         r"knochenodem", r"beenmerg", r"kostan\w* edem", r"kemik iligi", r"костно\w* едем", r"oedeme osseux",
         r"medular\w* edem", r"kontuz", r"contusi", r"bruise")
EFF = rx(r"effusion", r"derrame", r"erguss", r"hydrops", r"effusie", r"vocht", r"izljev", r"efuzyon",
         r"(mayi|sivi) artis", r"излив", r"epanchement", r"fluid")
SMALL = rx(r"small", r"minor", r"mild", r"minimal", r"trace", r"pequen", r"leve", r"escaso", r"gering", r"klein",
           r"wenig", r"beperkt", r"manji", r"mali", r"blag", r"minimal", r"az miktar", r"hafif", r"малък", r"лек",
           r"minime", r"petit", r"discret")
LARGE = rx(r"moderate", r"large", r"marked", r"significant", r"moderad", r"abundant", r"importante", r"gross",
           r"massiv", r"deutlich", r"ausgepr", r"fors", r"matig", r"groot", r"umjeren", r"velik", r"izrazit",
           r"orta", r"belirgin", r"fazla", r"умерен", r"голям", r"изразен", r"modere", r"abondant", r"important")
MEN_DEG = rx(r"menisc\w*\W+(\w+\W+)?(degenera|signal|mucoid|myxoid|intrasubstan)",
             r"(degenera|mucoid|myxoid)\w*\W+(\w+\W+){0,3}menisc", r"menisk\w*\W+(\w+\W+)?(degenera|signal|mukoid)",
             r"(degenera|mukoid)\w*\W+(\w+\W+){0,3}menisk", r"мениск\w*\W+(\w+\W+)?дегенер", r"menisq\w*\W+(\w+\W+)?degener")
SYN = rx(r"synovi", r"sinovi", r"hoffa", r"синови", r"plica", r"pannus", r"sinoviy")
LIG = rx(r"ligament\w*\W+(\w+\W+)?(thicken|signal|edema|laxity)", r"(thicken|signal|edema)\w*\W+(\w+\W+){0,2}ligament",
         r"band\w*\W+(\w+\W+)?(verdick|signal|odem)", r"sveza\w*\W+(\w+\W+)?(zadeb|edem|signal)",
         r"bag\w*\W+(\w+\W+)?(kalin|odem|sinyal)", r"връзк\w*\W+(\w+\W+)?(удебел|едем|сигнал)")
CYST = rx(r"cyst", r"quiste", r"zyste", r"cyste", r"cist", r"kist", r"кист", r"kyste", r"ganglion")

EXTRA = ["cart_med", "cart_lat", "cart_pf", "cart_any", "osteo", "bme", "eff_small", "eff_large", "eff_any",
         "men_deg", "syn", "lig", "cyst", "log_len"]


def extra_features(text: str) -> dict[str, float]:
    f = dict.fromkeys(EXTRA, 0.0)
    if not isinstance(text, str):
        return f
    f["log_len"] = np.log1p(len(text))
    for s in sentences(text):
        lead_neg = False
        for m in CART.finditer(s):
            if polarity(s, m.start(), lead_neg) > 0 and CART_BAD.search(s):
                f["cart_any"] += 1
                for key, loc in (("cart_med", "Medial OA"), ("cart_lat", "Lateral OA"), ("cart_pf", "PF OA")):
                    if COMP[loc].search(s):
                        f[key] += 1
        for name, pat in (("osteo", OSTEO), ("bme", BME), ("men_deg", MEN_DEG), ("syn", SYN), ("lig", LIG),
                          ("cyst", CYST)):
            for m in pat.finditer(s):
                if polarity(s, m.start(), lead_neg) > 0:
                    f[name] += 1
        for m in EFF.finditer(s):
            if polarity(s, m.start(), lead_neg) > 0:
                f["eff_any"] += 1
                f["eff_small"] += bool(SMALL.search(s))
                f["eff_large"] += bool(LARGE.search(s))
    for k in EXTRA[:-1]:
        f[k] = np.log1p(f[k])
    return f


def featurize(df: pd.DataFrame, variant: str = "full") -> np.ndarray:
    v1 = pd.DataFrame([label_report(t) for t in df.Report])[LABELS].values
    if variant == "rules":
        return v1
    ex = pd.DataFrame([extra_features(t) for t in df.Report])[EXTRA].values
    return np.hstack([v1, ex])


# per-label evidence groups, fixed by anatomy (a feature may only inform labels it is clinically related to)
GROUPS = {
    "ACL": ["lig", "bme"], "MCL": ["lig", "bme"],
    "Medial Meniscus": ["men_deg", "cyst"], "Lateral Meniscus": ["men_deg", "cyst"],
    "Medial OA": ["cart_med", "cart_any", "osteo"], "Lateral OA": ["cart_lat", "cart_any", "osteo"],
    "PF OA": ["cart_pf", "cart_any", "osteo"],
    "Effusion": ["eff_any", "eff_small", "eff_large", "syn"], "Synovitis": ["syn", "eff_any", "eff_large"],
    "Baker's": ["cyst"], "Contusion": ["bme"], "Fracture": ["bme"],
}


def group_cols(k: int) -> list[int]:
    """column indices for label k in the 'full' feature matrix: its own v1 output + its evidence group."""
    return [k] + [12 + EXTRA.index(f) for f in GROUPS[LABELS[k]]]


def make_model(C: float = 0.3):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def loo_predict(F: np.ndarray, Y: np.ndarray, C: float = 0.3, grouped: bool = False) -> np.ndarray:
    P = np.zeros(Y.shape)
    for i in range(len(Y)):
        tr = np.arange(len(Y)) != i
        for k in range(Y.shape[1]):
            c = group_cols(k) if grouped else slice(None)
            P[i, k] = make_model(C).fit(F[tr][:, c], Y[tr, k]).predict_proba(F[i:i + 1][:, c])[0, 1]
    return P


def per_label_auc(Y, P):
    return np.array([roc_auc_score(Y[:, k], P[:, k]) for k in range(Y.shape[1])])


def main(train_csv="/home/user/data/meta/train.csv", out="/home/user/data/meta/silver_labels_v2.csv"):
    df = pd.read_csv(train_csv)
    gm = df[LABELS].notna().all(axis=1)
    gold, pool = df[gm].reset_index(drop=True), df[~gm].reset_index(drop=True)
    Yg = gold[LABELS].values.astype(int)
    rows = {}
    v1 = pd.DataFrame([label_report(t) for t in gold.Report])[LABELS].values
    rows["v1 rules (no fitting)"] = per_label_auc(Yg, v1)
    for variant in ("rules", "full"):
        F = featurize(gold, variant)
        rows[f"v2 {variant} (LOO)"] = per_label_auc(Yg, loo_predict(F, Yg))
    rows["v2 grouped (LOO)"] = per_label_auc(Yg, loo_predict(featurize(gold, "full"), Yg, grouped=True))
    res = pd.DataFrame(rows, index=LABELS).T
    res["macro"] = res.mean(axis=1)
    os.makedirs("results", exist_ok=True)
    res.round(4).to_csv("results/labeler_v2_gold_loo.csv")
    print(res.round(3).to_string())

    # final v2 = grouped model refitted on all 58 gold, applied only to the non-gold pool
    Fg, Fp = featurize(gold, "full"), featurize(pool, "full")
    Pp = np.column_stack([make_model().fit(Fg[:, group_cols(k)], Yg[:, k]).predict_proba(Fp[:, group_cols(k)])[:, 1]
                          for k in range(12)])
    silver = pd.DataFrame(Pp, columns=LABELS)
    silver.insert(0, "StudyInstanceUID", pool.StudyInstanceUID.values)
    # keep gold rows present (with v1 values) so downstream code finds every id; they are never trained on
    g_v1 = pd.DataFrame(v1, columns=LABELS)
    g_v1.insert(0, "StudyInstanceUID", gold.StudyInstanceUID.values)
    pd.concat([silver, g_v1]).to_csv(out, index=False)
    print("wrote", out, "| v2 mean positive prob:", silver[LABELS].mean().round(3).to_dict())


if __name__ == "__main__":
    main()
