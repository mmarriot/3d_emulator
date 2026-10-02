"""The emulator on hand-made layer clusters, and the partition comparison."""
import numpy as np

from ticltune import clue, closure


def blob(mk, centres, n=6, e=2.0, layers=range(10, 16)):
    pos, lay, tr = [], [], []
    for b, (x, y) in enumerate(centres):
        for k, L in enumerate(list(layers)[:n]):
            pos.append([x, y, 350.0 + L]); lay.append(L); tr.append((len(pos) - 1, b, e))
    return mk(len(pos), tr, [{} for _ in centres], positions=pos, layers=lay)


def test_two_far_blobs_give_two_tracksters(mk):
    # at |z| ~ 350 cm a transverse radius of 150 cm gives r ~ 0.43 > rhocPivotRadius, so rhoc ~ 0.8
    ev = blob(mk, [(150.0, 0.0), (0.0, 150.0)])
    lab = clue.cluster(ev)
    assert len(np.unique(lab[lab >= 0])) == 2
    assert len(np.unique(lab[:6])) == 1 and len(np.unique(lab[6:])) == 1


def test_isolated_soft_point_is_an_outlier(mk):
    ev = mk(1, [(0, 0, 0.01)], [{}], positions=[[30.0, 30.0, 350.0]], layers=[10])
    assert clue.cluster(ev, drop_small=False)[0] == -1


def test_small_tracksters_are_dropped(mk):
    ev = mk(1, [(0, 0, 50.0)], [{}], positions=[[150.0, 0.0, 350.0]], layers=[10])
    assert clue.cluster(ev, drop_small=False)[0] == 0      # a seed of its own
    assert clue.cluster(ev, drop_small=True)[0] == -1      # but fewer than 2 layer clusters


def test_masked_layer_clusters_are_never_assigned(mk):
    ev = blob(mk, [(150.0, 0.0)])
    ev.lc["mask"][2] = 0
    assert clue.cluster(ev)[2] == -1


def test_endcaps_never_mix(mk):
    ev = mk(4, [(i, 0, 3.0) for i in range(4)], [{}],
            positions=[[150, 0, 350], [150, 0, 351], [150, 0, -350], [150, 0, -351]], layers=[10, 11, 10, 11])
    lab = clue.cluster(ev)
    assert lab[0] == lab[1] and lab[2] == lab[3] and lab[0] != lab[2]


def test_neighbour_exactly_dc_away_is_rounded_as_cmssw(mk):
    # dc * layerScale = 6 layers, so the 6-layer neighbour sits on the dc boundary and float rounding
    # of CMSSW's layer coordinate decides; expected values from that arithmetic (float32, GPU fma)
    for z, l, inside in [(-350, 1, True), (-350, 2, False), (350, 1, True), (350, 13, False)]:
        ev = mk(2, [(0, 0, 1.0), (1, 0, 1.0)], [{}], positions=[[150, 0, z], [150, 0, z]], layers=[l, l + 6])
        pr = clue._pairs(ev, np.arange(2), clue.DEFAULT, cache_key=0)
        assert bool(np.any(pr["D"] <= clue.DEFAULT.dc)) == inside, (z, l)


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
