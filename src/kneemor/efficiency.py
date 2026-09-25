"""Parameters, FLOPs (torch FlopCounterMode, measured on the actual routed graph),
CPU latency and peak RSS for each head and for the frozen backbone."""
from __future__ import annotations

import json
import os
import resource
import sys
import time

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kneemor.features import build_backbone  # noqa: E402
from kneemor.models import build, n_params  # noqa: E402

HEADS = ["mvmor", "mvmor_norouting", "mvmor_r1", "mvmor_meandec", "mvmor_noplane", "mvmor_unshared",
         "transformer", "abmil", "meanmlp"]


def flops(fn):
    with FlopCounterMode(display=False) as fc:
        fn()
    return fc.get_total_flops()


def latency(fn, reps=30, warm=5):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1000), float(np.percentile(ts, 90) * 1000)


def main(out="results/efficiency.json", threads=1):
    torch.set_num_threads(threads)
    x = torch.randn(1, 3, 24, 4, 512)
    m = torch.ones(1, 3, dtype=torch.bool)
    res = {}
    with torch.no_grad():
        for h in HEADS:
            net = build(h).eval()
            f = flops(lambda: net(x, m))
            lat = latency(lambda: net(x, m))
            res[h] = dict(params=n_params(net), gflops=f / 1e9, latency_ms_median=lat[0], latency_ms_p90=lat[1])
            print(h, res[h], flush=True)
        bb = build_backbone("/home/user/data/r18a1.pth")
        vol = torch.randn(72, 3, 160, 160)
        f = flops(lambda: bb(vol))
        lat = latency(lambda: bb(vol), reps=5, warm=1)
        res["backbone_resnet18_72slices"] = dict(params=sum(p.numel() for p in bb.parameters()), gflops=f / 1e9,
                                                 latency_ms_median=lat[0], latency_ms_p90=lat[1])
        print(res["backbone_resnet18_72slices"])
    res["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    res["threads"] = threads
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
