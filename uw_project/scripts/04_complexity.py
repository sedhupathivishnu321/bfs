"""Parameters, FLOPs (torch FlopCounter, 2*MAC), CPU latency (batch 1 and batch 256, 1 thread-set = 4 cores) and parameter memory."""
import sys
from pathlib import Path
import torch, pandas as pd
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R / "src"))
from uwfc import train as T, data as D
from uwfc.models import make_model
fin = 3 * D.NF
Ls = D.L
def analytic_mflops(n):
    # analytic 2*MAC counts; torch's FlopCounter misses fused RNN / attention kernels, so these are authoritative for LSTM/GRU/Transformer
    if n == "LSTM": mac = Ls * 4 * 64 * (fin + 64) + 64 * 6
    elif n == "GRU": mac = Ls * 3 * 64 * (fin + 64) + 64 * 6
    elif n == "Transformer": d, ff = 32, 64; mac = Ls * fin * d + 2 * (4 * Ls * d * d + 2 * Ls * Ls * d + 2 * Ls * d * ff) + d * 6
    else: return float('nan')
    return 2 * mac / 1e6
rows = []
for n in ("LSTM", "GRU", "TCN", "Transformer", "PCT"):
    m = make_model(n, fin, 6, L=D.L, n_own=D.NF).eval()
    x1, xb = torch.randn(1, D.L, fin), torch.randn(256, D.L, fin)
    rows.append(dict(model=n, params=T.count_params(m), param_MB=T.count_params(m) * 4 / 2**20, MFLOPs_torch_counter=T.flops(m, x1) / 1e6, MFLOPs_analytic=analytic_mflops(n),
                     latency_ms_b1=T.latency_ms(m, x1), latency_ms_b256=T.latency_ms(m, xb, 50)))
df = pd.DataFrame(rows).round(4); df.to_csv(R / "results/complexity.csv", index=False); print(df.to_string(index=False))
