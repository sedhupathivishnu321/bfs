"""Compute scalability of each forecaster vs number of nodes (receivers) N: every node is one independent window batch (one forecast per node).
Measured on this machine's CPU (4 vCPU, no GPU) with random inputs -> timing/memory only, no accuracy. Run on an otherwise idle machine."""
import sys, time, resource, tracemalloc
from pathlib import Path
import numpy as np, pandas as pd, torch
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R / "src"))
from uwfc import data as D, train as T
from uwfc.models import make_model
torch.set_num_threads(4); fin = 3 * D.NF + 0 + D.NF
fin = 2 * D.NF + D.NF   # own rel + ctx rel + (log-vol replaces nothing) + abs  -> matches nrm.x width (3*NF+... ) approximately
fin = 3 * D.NF + D.NF   # own, ctx, logvol, abs
rows = []
for n in ("LSTM", "GRU", "TCN", "Transformer", "PCT"):
    for N in (1, 4, 16, 64, 256, 1024):
        m = make_model(n, fin, 6, L=D.L, n_own=D.NF).eval(); x = torch.randn(N, D.L, fin)
        with torch.no_grad():
            for _ in range(5): m(x)
            reps = max(3, int(200 / N)); t = time.perf_counter()
            for _ in range(reps): m(x)
            lat = (time.perf_counter() - t) / reps * 1e3
        from torch.profiler import profile, ProfilerActivity
        with torch.no_grad(), profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof: m(x)
        peak = sum(max(e.self_cpu_memory_usage, 0) for e in prof.key_averages()) / 2**20      # MB of tensor memory allocated in one forward pass (upper bound on activations)
        mt = make_model(n, fin, 6, L=D.L, n_own=D.NF); opt = torch.optim.AdamW(mt.parameters()); xt = torch.randn(N * 4, D.L, fin); yt = torch.randn(N * 4, 6)
        t = time.perf_counter()
        for _ in range(3):
            for i in range(0, len(xt), 256):
                loss = ((mt(xt[i:i + 256])[0] - yt[i:i + 256]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        tr = (time.perf_counter() - t) / 3
        rows.append(dict(model=n, nodes=N, infer_ms_total=lat, infer_ms_per_node=lat / N, act_alloc_MB=peak, train_s_per_epoch=tr))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(R / "results/scalability.csv", index=False)
