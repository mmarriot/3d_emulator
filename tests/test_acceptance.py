"""Acceptance tests on a real v3 ntuple (V3_TRUTH_AND_METRICS.md section 12), run when TICLTUNE_NTUPLE points to one:
    TICLTUNE_NTUPLE=/path/job_000.root python3 -m pytest -q tests/test_acceptance.py
"""
import os

import numpy as np
import pytest

from ticltune import clue, closure, data, metrics, truth

PATH = os.environ.get("TICLTUNE_NTUPLE")
pytestmark = pytest.mark.skipif(not PATH, reason="set TICLTUNE_NTUPLE to a v3 ntuple")


@pytest.fixture(scope="module")
def events():
    return data.load(PATH, max_events=int(os.environ.get("TICLTUNE_NEV", "2")))


def summary(ev, labels, level):
    return metrics.summary(metrics.score(ev, labels, level))


def test_truth_is_consistent(events):
    for ev in events:
        t = truth.build(ev)
        inter = ev.un["bx"] * 1_000_000 + ev.un["evt"]
        assert all(len(np.unique(inter[t.unit_target == k])) == 1 for k in np.unique(t.unit_target))
        E = np.asarray(ev.rh["E"], float)
        assert np.isclose(t.E.sum() + t.no_truth.sum(), E.sum(), rtol=1e-5)
        per_rh = np.asarray(t.T.sum(axis=1)).ravel()
        assert np.allclose(per_rh + t.no_truth, E, rtol=1e-4, atol=1e-6)
        assert ((t.completeness > t.frac) & (t.purity > t.frac) | t.unreachable).all()


def test_level_closure(events):
    # a layer-cluster clustering scored through the rechit fractions = scored on layer clusters with projected truth
    for ev in events:
        t = truth.build(ev)
        lab = clue.cluster(clue.points(ev, "lc"))
        L = metrics.objects_from_labels(lab, ev.n_lc)
        S_rh = (metrics.objects_from_lc_labels(ev, lab) @ t.T).toarray()
        S_lc = (L @ (ev.lc_to_rh() @ t.T)).toarray()
        assert np.allclose(S_rh, S_lc, rtol=1e-9, atol=1e-9)


def test_reference_points(events):
    for ev in events:
        t = truth.build(ev)
        ideal = summary(ev, truth.ideal_labels(ev, t, "rh"), "rh")
        assert ideal["Phi_sig"] == 0
        assert summary(ev, truth.ideal_labels(ev, t, "lc"), "lc")["Phi_sig"] == 0
        shatter = summary(ev, np.arange(ev.n_rh), "rh")                       # one object per rechit
        assert shatter["eps_sig"] == pytest.approx(ideal["eps_sig"])
        assert shatter["K_sig"] == pytest.approx(ideal["K_sig"])
        assert shatter["Phi_sig"] > 0.5
        blob = summary(ev, (ev.rh["z"] > 0).astype(np.int64), "rh")          # one object per endcap
        assert blob["Phi_sig"] == 0 and blob["K_sig"] > 0.9
        empty = summary(ev, np.full(ev.n_rh, -1), "rh")
        assert (empty["eps_sig"], empty["K_sig"], empty["Phi_sig"]) == (0.0, 1.0, 0.0)


def test_accounting_and_ranges(events):
    for ev in events:
        for level in ("lc", "rh"):
            pts = clue.points(ev, level)
            p = clue.DEFAULT if level == "lc" else clue.DEFAULT.with_(outlierDistance=1.5, seedingDistance=1.4,
                                                                       sigmaT=(0.0023, 0.0046, 0.024), layerScale=1.5,
                                                                       rhoc=0.1)
            s = metrics.score(ev, clue.cluster(pts, p), level)
            assert np.allclose(s["e_num"] + s["lost_other"] + s["lost_none"], s["e_den"])
            m = metrics.summary(s)
            assert all(0 <= m[k] <= 1 for k in ("K_sig", "eps_sig", "Phi_sig"))


def test_emulator_closure_with_cmssw(events):
    if events[0].clue_assignment is None:
        pytest.skip("no CMSSW CLUEstering assignment in this ntuple")
    res = closure.run(events)
    assert res["raw"]["lc_agree_frac"] > 0.999
