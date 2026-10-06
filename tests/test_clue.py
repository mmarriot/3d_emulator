"""The emulator on hand-made layer clusters, and the partition comparison."""
import numpy as np

from ticltune import clue, closure


def blob(mk, centres, n=6, e=2.0, layers=range(10, 16)):
    pos, lay, tr = [], [], []
    for b, (x, y) in enumerate(centres):
        for k, L in enumerate(list(layers)[:n]):
            pos.append([x, y, 350.0 + L]); lay.append(L); tr.append((len(pos) - 1, b, e))
    return mk(len(pos), tr, positions=pos, layers=lay)


def lc(ev, p=clue.DEFAULT, **kw):
    return clue.cluster(clue.points(ev, "lc"), p, **kw)


def test_two_far_blobs_give_two_tracksters(mk):
    # at |z| ~ 350 cm a transverse radius of 150 cm gives r ~ 0.43 > rhocPivotRadius, so rhoc ~ 0.8
    ev = blob(mk, [(150.0, 0.0), (0.0, 150.0)])
    lab = lc(ev)
    assert len(np.unique(lab[lab >= 0])) == 2
    assert len(np.unique(lab[:6])) == 1 and len(np.unique(lab[6:])) == 1


def test_isolated_soft_point_is_an_outlier(mk):
    ev = mk(1, [(0, 0, 0.01)], positions=[[30.0, 30.0, 350.0]], layers=[10])
    assert lc(ev, drop_small=False)[0] == -1


def test_small_tracksters_are_dropped(mk):
    ev = mk(1, [(0, 0, 50.0)], positions=[[150.0, 0.0, 350.0]], layers=[10])
    assert lc(ev, drop_small=False)[0] == 0      # a seed of its own
    assert lc(ev, drop_small=True)[0] == -1      # but fewer than 2 layer clusters


def test_masked_layer_clusters_are_never_assigned(mk):
    ev = blob(mk, [(150.0, 0.0)])
    ev.lc["mask"][2] = 0
    assert lc(ev)[2] == -1


def test_endcaps_never_mix(mk):
    ev = mk(4, [(i, 0, 3.0) for i in range(4)],
            positions=[[150, 0, 350], [150, 0, 351], [150, 0, -350], [150, 0, -351]], layers=[10, 11, 10, 11])
    lab = lc(ev)
    assert lab[0] == lab[1] and lab[2] == lab[3] and lab[0] != lab[2]


def test_neighbour_exactly_dc_away_is_rounded_as_cmssw(mk):
    # dc * layerScale = 6 layers, so the 6-layer neighbour sits on the dc boundary and float rounding
    # of CMSSW's layer coordinate decides; expected values from that arithmetic (float32, GPU fma)
    p = clue.DEFAULT.with_(fma=True)
    for z, l, inside in [(-350, 1, True), (-350, 2, False), (350, 1, True), (350, 13, False)]:
        ev = mk(2, [(0, 0, 1.0), (1, 0, 1.0)], positions=[[150, 0, z], [150, 0, z]], layers=[l, l + 6])
        pr = clue._pairs(clue.points(ev, "lc"), np.arange(2), p, 0)
        assert bool(np.any(pr["D"] <= p.dc)) == inside, (z, l)


def test_layer_coordinate_without_fma_rounds_product_then_sum(mk):
    # CPU serial backend: float32(float32(layer / layerScale) + gap), so on the + side a 6-layer neighbour
    # can fall either side of dc where the fused GPU arithmetic says otherwise
    inv = np.float32(1.0) / np.float32(clue.DEFAULT.layerScale)
    gap = np.float32(2.0) * np.float32(clue.DEFAULT.outlierDistance)
    for l in range(1, 42):
        ev = mk(2, [(0, 0, 1.0), (1, 0, 1.0)], positions=[[150, 0, 350], [150, 0, 350]], layers=[l, l + 6])
        pr = clue._pairs(clue.points(ev, "lc"), np.arange(2), clue.DEFAULT, 0)
        za, zb = (np.float32(np.float32(np.float32(k - 1 + clue.LAST_LAYER) * inv) + gap) for k in (l, l + 6))
        assert bool(np.any(pr["D"] <= clue.DEFAULT.dc)) == bool(abs(zb - za) <= np.float32(1.0)), l


def test_compare_ignores_label_numbering():
    a = np.array([0, 0, 1, 1, -1])
    b = np.array([7, 7, 3, 3, -1])
    r = closure.compare(a, b)
    assert r["lc_agree"] == 5 and r["identical_clusters"] == 2


def test_compare_detects_a_moved_layer_cluster():
    a = np.array([0, 0, 0, 1, 1])
    b = np.array([0, 0, 1, 1, 1])
    r = closure.compare(a, b)
    assert r["identical_clusters"] == 0 and r["lc_agree"] == 0


def test_candidate_search_holds_every_pair_within_the_metric():
    # per-detector-pair search vs brute force, three relative sigmas as in CMSSW (EE, HSi, HSci)
    rng = np.random.default_rng(7)
    n = 600
    X = rng.uniform(0, 30, n).astype(np.float32)
    Y = rng.uniform(0, 30, n).astype(np.float32)
    Z = (rng.integers(0, 47, n) / np.float32(6)).astype(np.float32)
    rel = rng.choice(np.array([1, 2, 4], np.float32), n)
    for rmax in (1.0, 2.8, 4.5):
        i, j = clue._candidates(X, Y, Z, rel, rmax)
        got = {(min(a, b), max(a, b)) for a, b in zip(i.tolist(), j.tolist())}
        I, J = np.triu_indices(n, 1)
        dT = np.sqrt((X[I] - X[J]) ** 2 + (Y[I] - Y[J]) ** 2) * np.float32(2) / (rel[I] + rel[J])
        d = np.maximum(dT, np.abs(Z[I] - Z[J]))
        want = {(a, b) for a, b in zip(I[d <= rmax].tolist(), J[d <= rmax].tolist())}
        assert want <= got and len(got) == len(set(zip(i.tolist(), j.tolist())))


def test_rechit_points_cluster_like_layer_clusters_of_one_rechit(mk):
    # every layer cluster is one rechit: the two levels see the same points (tags in the same order)
    ev = blob(mk, [(150.0, 0.0), (0.0, 150.0), (150.0, 2.0)])
    a, b = lc(ev), clue.cluster(clue.points(ev, "rh"))
    assert closure.compare(a, b)["lc_agree"] == ev.n_lc


def test_rechit_level_uses_every_hgcal_rechit(mk):
    # rechits outside any layer cluster and in masked layer clusters are points at rechit level
    ev = blob(mk, [(150.0, 0.0)])
    ev.lc["mask"][:] = 0
    assert (lc(ev) == -1).all()
    assert (clue.cluster(clue.points(ev, "rh")) >= 0).all()


def test_all_layer_clusters_level_adds_what_only_the_size_cut_removed(mk):
    # lc_all = the CLUE3DHigh mask without the size cut: masked HGCAL layer clusters of algo 6/7/8 become points,
    # other masked ones (another algo) stay out
    ev = blob(mk, [(150.0, 0.0)])
    ev.lc["mask"][[1, 2]] = 0
    ev.lc["algo"][2] = 9
    assert list(clue.points(ev, "lc").active) == [True, False, False, True, True, True]
    assert list(clue.points(ev, "lc_all").active) == [True, True, False, True, True, True]
    lab = clue.cluster(clue.points(ev, "lc_all"))
    assert lab[1] == lab[0] >= 0 and lab[2] == -1
