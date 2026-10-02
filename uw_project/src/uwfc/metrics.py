import numpy as np
from sklearn.metrics import roc_auc_score


def reg_metrics(Yp, Y, y_s, names):
    """Errors in physical units (de-normalised by y_s)."""
    e = (Yp - Y) * y_s
    return {f"MAE_{n}": float(np.abs(e[:, j]).mean()) for j, n in enumerate(names)} | \
           {f"RMSE_{n}": float(np.sqrt((e[:, j] ** 2).mean())) for j, n in enumerate(names)}


def event_metrics(Yp, Y, y_s, col, thr):
    """Degradation event: true gain change at horizon `col` < thr dB. Score = -predicted change."""
    yt = (Y[:, col] * y_s[col] < thr).astype(int)
    sc = -Yp[:, col] * y_s[col]
    if yt.min() == yt.max(): return {}
    pred = (-sc < thr).astype(int)
    tp = ((pred == 1) & (yt == 1)).sum(); tn = ((pred == 0) & (yt == 0)).sum()
    fp = ((pred == 1) & (yt == 0)).sum(); fn = ((pred == 0) & (yt == 1)).sum()
    P = tp / max(tp + fp, 1); Rc = tp / max(tp + fn, 1)
    return dict(AUC=float(roc_auc_score(yt, sc)), precision=float(P), recall_sens=float(Rc), specificity=float(tn / max(tn + fp, 1)),
                F1=float(2 * P * Rc / max(P + Rc, 1e-9)), accuracy=float((pred == yt).mean()), event_rate=float(yt.mean()))


def coverage(Yp, lv, Y, y_s):
    """Empirical coverage of the nominal 90% Gaussian interval (z=1.645) and its mean width (physical units)."""
    if lv is None: return {}
    sd = np.exp(0.5 * lv)
    inside = np.abs(Y - Yp) <= 1.645 * sd
    return dict(cov90=float(inside.mean()), width90=float((2 * 1.645 * sd * y_s).mean()))
