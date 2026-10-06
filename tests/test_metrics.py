"""The metrics (V3_TRUTH_AND_METRICS.md section 6): each test is one clustering of a toy event with the values the
definitions give by hand."""
import numpy as np
import pytest
import scipy.sparse as sp

from ticltune import metrics, truth
from ticltune.data import Tracksters

PU = dict(signal=0, evt=1)


def run(ev, labels, level="rh"):
    return metrics.summary(metrics.score(ev, np.asarray(labels), level))


def same(a, b):
    """Two summaries are equal (nan == nan)."""
    return a.keys() == b.keys() and all(np.isclose(a[k], b[k], equal_nan=True, rtol=1e-12) for k in a)


def two_particles(mk):
    # signal a (rechits 0, 1: 2 + 2 GeV), signal b (rechits 2, 3: 1 + 3 GeV), both hadrons
    return mk(4, [(0, 0, 2.0), (1, 0, 2.0), (2, 1, 1.0), (3, 1, 3.0)])


def test_perfect_clustering(mk):
    s = run(two_particles(mk), [0, 0, 1, 1])
    assert (s["eps_sig"], s["K_sig"], s["Phi_sig"]) == pytest.approx((1, 0, 0))


def test_pure_fragments_keep_efficiency_and_are_charged_as_surplus(mk):
    # a in two pieces (2 + 2), b in one: c = (4, 4), N = (2, 1) -> Phi = 1 - 8 / (8 + 4) = 1/3
    s = run(two_particles(mk), [0, 1, 2, 2])
    assert (s["eps_sig"], s["K_sig"]) == pytest.approx((1, 0)) and s["Phi_sig"] == pytest.approx(1 / 3)


def test_shattering_reaches_the_ideal_values_and_maximal_fragmentation(mk):
    s = run(two_particles(mk), [0, 1, 2, 3])
    assert (s["eps_sig"], s["K_sig"]) == pytest.approx((1, 0)) and s["Phi_sig"] == pytest.approx(0.5)


def test_one_blob_credits_only_its_best_target(mk):
    # everything in one object: best target a (4 of 8), b gets nothing
    s = run(two_particles(mk), [0, 0, 0, 0])
    assert s["K_sig"] == pytest.approx(0.5) and s["eps_sig"] == pytest.approx(0.5) and s["Phi_sig"] == 0
    assert s["lost_other_HAD"] == pytest.approx(0.5)


def test_empty_clustering(mk):
    s = run(two_particles(mk), [-1, -1, -1, -1])
    assert (s["eps_sig"], s["K_sig"], s["Phi_sig"]) == (0.0, 1.0, 0.0)
    assert s["lost_none_HAD"] == pytest.approx(1)


def test_pileup_in_a_signal_object_is_contamination(mk):
    # object 0: signal a (3) + pileup (1) -> K = 1/4; a fully collected
    ev = mk(2, [(0, 0, 3.0), (1, 1, 1.0)], units=[{}, PU])
    s = run(ev, [0, 0])
    assert s["K_sig"] == pytest.approx(0.25) and s["eps_sig"] == pytest.approx(1)


def test_no_truth_energy_is_contamination(mk):
    ev = mk(2, [(0, 0, 3.0)], no_truth=[0.0, 1.0])
    assert run(ev, [0, 0])["K_sig"] == pytest.approx(0.25)


def test_assignment_is_by_plurality(mk):
    # one rechit: signal a 0.3, three pileup particles 0.25 / 0.25 / 0.2 (different collisions, never merged):
    # a has the most -> a signal object with K = 0.7
    ev = mk(1, [(0, 0, 0.3), (0, 1, 0.25), (0, 2, 0.25), (0, 3, 0.2)],
            units=[{}, dict(signal=0, evt=1), dict(signal=0, evt=2), dict(signal=0, evt=3)])
    s = run(ev, [0])
    assert s["K_sig"] == pytest.approx(0.7) and s["signal_objects_per_event"] == 1


def test_signal_in_a_pileup_object_is_lost_and_counted_in_K_pu(mk):
    # object: pileup 0.6 + signal a 0.4 -> a pileup object: a's 0.4 is an efficiency loss, K_pu = 0.4
    ev = mk(2, [(0, 1, 0.6), (1, 0, 0.4), (1, 1, 0.0)], units=[{}, PU])
    s = run(ev, [0, 0])
    assert s["eps_sig"] == 0 and s["K_pu"] == pytest.approx(0.4) and s["K_sig"] == 1.0
    assert s["pileup_objects_per_event"] == 1


def test_noise_objects_are_unassigned_and_change_nothing(mk):
    ev = mk(3, [(0, 0, 2.0), (1, 0, 2.0)], no_truth=[0, 0, 5.0])
    a, b = run(ev, [0, 0, -1]), run(ev, [0, 0, 1])
    assert b["noise_objects_per_event"] == 1
    assert (a["eps_sig"], a["K_sig"], a["Phi_sig"]) == (b["eps_sig"], b["K_sig"], b["Phi_sig"])


def test_ties_go_to_the_larger_target_then_the_lower_index(mk):
    # one object holds 1 GeV of a (deposited 1) and 1 GeV of b (pileup, deposited 3): b wins
    ev = mk(3, [(0, 0, 1.0), (0, 1, 1.0), (1, 1, 2.0)], units=[{}, PU])
    best, own, _ = metrics.assign(metrics.objects(ev, np.array([0, -1, -1]), "rh"), truth.build(ev))
    assert list(best) == [1] and own[0] == pytest.approx(1)
    ev = mk(1, [(0, 0, 1.0), (0, 1, 1.0)], units=[dict(signal=0, evt=1), dict(signal=0, evt=2)])
    assert list(metrics.assign(metrics.objects(ev, np.array([0]), "rh"), truth.build(ev))[0]) == [0]


def test_classes_are_averaged(mk):
    # an EM target perfectly clustered, a hadron half collected: eps = (1 + 0.5) / 2
    ev = mk(3, [(0, 0, 2.0), (1, 1, 1.0), (2, 1, 1.0)], units=[dict(pdg=22), {}])
    s = run(ev, [0, 1, -1])
    assert s["eps_EM"] == 1 and s["eps_HAD"] == pytest.approx(0.5) and s["eps_sig"] == pytest.approx(0.75)


def test_cell_floor_of_contamination(mk):
    # a rechit shared 3 (a) / 1 (pileup): any object holding it carries 1 GeV of contamination
    ev = mk(2, [(0, 0, 3.0), (0, 1, 1.0), (1, 0, 4.0)], units=[{}, PU])
    s = run(ev, [0, 0])
    assert s["K_HAD"] == pytest.approx(1 / 8) and s["K_floor_HAD"] == pytest.approx(1 / 8)


def test_layer_cluster_level_equals_rechit_level(mk):
    # layer clusters with shared rechits: scoring a layer-cluster clustering through the fractions equals scoring
    # the same rechit fractions directly
    lcs = [[(0, 1.0), (1, 0.7)], [(1, 0.3), (2, 1.0)], [(3, 1.0)]]
    ev = mk(4, [(0, 0, 2.0), (1, 0, 1.0), (1, 1, 1.0), (2, 1, 2.0), (3, 2, 1.0)], units=[{}, {}, PU], lcs=lcs)
    t = truth.build(ev)
    lab = np.array([0, 1, 1])
    W_lc = metrics.objects(ev, lab, "lc")
    W = sp.csr_matrix(np.array([[1, 0.7, 0, 0], [0, 0.3, 1, 1]]))
    a, b = metrics.evaluate(ev, W_lc, t), metrics.evaluate(ev, W, t)
    assert all(np.allclose(a[k], b[k]) for k in metrics.KEYS)
    # with whole rechits, a layer-cluster clustering equals the rechit clustering it implies
    ev2 = mk(4, [(0, 0, 2.0), (1, 0, 1.0), (2, 1, 2.0), (3, 1, 1.0)],
             lcs=[[(0, 1.0), (1, 1.0)], [(2, 1.0)], [(3, 1.0)]])
    a, b = run(ev2, [0, 1, -1], "lc"), run(ev2, [0, 0, 1, -1], "rh")
    assert same(a, b)


def test_cmssw_tracksters_use_the_vertex_multiplicity(mk):
    # layer cluster 1 shared by two tracksters (multiplicity 2): each takes half of it
    ev = mk(3, [(0, 0, 2.0), (1, 0, 1.0), (1, 1, 1.0), (2, 1, 2.0)])
    ts = Tracksters(np.array([0, 2, 4]), np.array([0, 1, 1, 2]), np.array([1, 0.5, 0.5, 1]))
    W = metrics.objects_from_tracksters(ev, ts)
    assert W.toarray() == pytest.approx(np.array([[1, 0.5, 0], [0, 0.5, 1]]))
    s = metrics.summary(metrics.evaluate(ev, W, truth.build(ev)))
    assert s["K_sig"] == pytest.approx(0.5 / 3) and s["eps_sig"] == pytest.approx(2.5 / 3)


def test_accounting_identity(mk):
    rng = np.random.default_rng(1)
    for _ in range(20):
        ev, lab = random_event(mk, rng)
        s = metrics.evaluate(ev, metrics.objects(ev, lab, "rh"), truth.build(ev))
        assert np.allclose(s["e_num"] + s["lost_other"] + s["lost_none"], s["e_den"])


def random_event(mk, rng, n_rh=30, n_un=6):
    units = [dict(signal=int(rng.random() < 0.6), evt=int(rng.integers(0, 3)), pdg=int(rng.choice([22, 211])))
             for _ in range(n_un)]
    for u in units:
        u["evt"] = 0 if u["signal"] else u["evt"] + 1
    truth_ = [(int(c), int(u), float(rng.uniform(0.1, 3))) for c in range(n_rh)
              for u in rng.choice(n_un, size=rng.integers(1, 4), replace=False)]
    ev = mk(n_rh, truth_, units=units, no_truth=rng.uniform(0, 0.3, n_rh))
    return ev, rng.integers(-1, 6, n_rh)


def test_splitting_never_lowers_collection_nor_raises_contamination(mk):
    # why the fragmentation metric is needed: metrics 1 and 2 alone reward shattering
    rng = np.random.default_rng(2)
    for _ in range(30):
        ev, lab = random_event(mk, rng)
        t = truth.build(ev)
        W = metrics.objects(ev, lab, "rh")
        best, own, _ = metrics.assign(W, t)
        k = int(rng.integers(0, lab.max() + 1))
        split = lab.copy()
        members = np.nonzero(lab == k)[0]
        split[members[: len(members) // 2]] = lab.max() + 1
        best2, own2, _ = metrics.assign(metrics.objects(ev, split, "rh"), t)
        E1 = metrics.objects(ev, lab, "rh") @ ev.rh["E"]
        E2 = metrics.objects(ev, split, "rh") @ ev.rh["E"]
        assert own2.sum() >= own.sum() - 1e-9                                           # credit never drops
        assert (E2 - own2)[best2 >= 0].sum() <= (E1 - own)[best >= 0].sum() + 1e-9     # contamination never rises
        assert (best2 >= 0).sum() >= (best >= 0).sum()                                  # objects never fewer


def test_metrics_are_in_the_unit_interval(mk):
    rng = np.random.default_rng(3)
    for _ in range(30):
        ev, lab = random_event(mk, rng)
        s = run(ev, lab)
        for k in ("K_sig", "eps_sig", "Phi_sig"):
            assert 0 <= s[k] <= 1


def test_sums_are_additive_over_events(mk):
    rng = np.random.default_rng(4)
    evs = [random_event(mk, rng) for _ in range(5)]
    sums = [metrics.evaluate(ev, metrics.objects(ev, lab, "rh"), truth.build(ev)) for ev, lab in evs]
    whole = metrics.summary(metrics.combine(sums))
    parts = metrics.summary(metrics.combine([metrics.combine(sums[:2]), metrics.combine(sums[2:])]))
    assert same(whole, parts) and whole["events"] == 5


def test_samples_are_averaged_with_equal_weight():
    a = dict(K_sig=0.2, eps_sig=0.6, Phi_sig=0.1, events=10)
    b = dict(K_sig=0.4, eps_sig=0.8, Phi_sig=0.3, events=100)
    m = metrics.mean_over_samples([a, b])
    assert m == pytest.approx(dict(K_sig=0.3, eps_sig=0.7, Phi_sig=0.2, events=110))
