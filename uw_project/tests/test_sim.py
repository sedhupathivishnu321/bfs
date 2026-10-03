import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from uwfc import linksim as S, ctrl as C, dt as DT


def test_hybrid_latency_is_acoustic_when_optical_fails():
    """Regression test for the bug where HYB latency was min(lat_a, lat_o) even if the optical packet was lost."""
    p = S.Params(); d = np.full((2, 2), 800.0)
    sa = np.full((2, 2), 20.0); so = np.full((2, 2), -30.0)               # acoustic fine, optical dead
    m = S.link_metrics(sa, so, np.full((2, 2), 4), p=p, dist=d)            # HYB low
    ac = S.link_metrics(sa, so, np.full((2, 2), 0), p=p, dist=d)
    assert np.allclose(m["latency"], ac["latency"], atol=1e-6) and m["latency"].min() > 0.5


def test_pdr_ber_monotone_and_modes():
    p = S.Params(); s = np.linspace(-5, 30, 50)
    assert np.all(np.diff(S.ber_bpsk(s)) <= 0) and np.all(np.diff(S.ber_qam64(s + 5)) <= 0)
    sa = np.full((1,), 12.0); so = np.full((1,), 25.0); d = np.array([10.0])
    r = {a: S.link_metrics(sa, so, np.array([a]), p=p, dist=d) for a in range(6)}
    assert r[4]["goodput"] > r[0]["goodput"] and r[4]["energy"] > r[0]["energy"]       # HYB uses both links
    assert r[1]["energy"] > r[0]["energy"] and r[1]["pdr"] >= r[0]["pdr"]              # high power costs more, never lowers PDR


def test_scenarios_use_real_traces_and_are_leak_free():
    tr = S.Traces(["blue", "red"]); rng = np.random.default_rng(0); sc = S.make_scenarios(tr, 4, 6, rng, sites=["blue"])
    assert sc["is_real"].all() and sc["gain"].shape == (4, 6, S.TOT) and np.isfinite(sc["gain"]).all()
    d = DT.build(sc, S.Params(), rng)
    # target at decision step must lie after the last observed step by obs_delay
    assert d["X"].shape[2] == S.HIST and d["tgt"].shape[-1] == 4 and np.isfinite(d["tgt"]).all()


def test_battery_death_and_rollout_shapes():
    p = S.Params(); tr = S.Traces(["blue"]); rng = np.random.default_rng(1); sc = S.make_scenarios(tr, 3, 4, rng, sites=["blue"]); d = DT.build(sc, p, rng); ep = C.Episode(d, 3, p)
    out = C.rollout(ep, C.Fixed(1), p)                                      # AC high power every step exhausts a 45 % battery
    assert out["reward"].shape == (ep.T, 3, 4) and (out["energy"] == 0).any() and out["viol"][-1].min() == 1.0
    s = C.summarize(out); assert 0 <= s["pdr"] <= 1 and abs(s["mode_ac"] - 1.0) < 1e-9
