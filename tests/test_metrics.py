"""Each test is one property the metrics must have."""
import numpy as np
import pytest

from ticltune import metrics, truth
from ticltune.data import Tracksters


def score(ev, labels, **cfg):
    # the property tests use toy energies of a few GeV: select every target unless a test says
    # otherwise. The 5 GeV selection itself is tested in its own section below.
    cfg.setdefault("select_energy", 0.0)
    t = truth.build(ev)
    d = metrics.evaluate(ev, Tracksters.from_labels(labels, ev.lc_energy()), t, metrics.MetricConfig(**cfg),
                         keep_details=True)
    return metrics.summary(d["sums"]), d


@pytest.fixture
def two_separable(mk):
    # two signal particles, never in the same layer cluster
    return mk(4, [(0, 0, 5.0), (1, 0, 5.0), (2, 1, 3.0), (3, 1, 3.0)], [{}, {}])


def test_perfect_reconstruction(two_separable):
    s, d = score(two_separable, [0, 0, 1, 1])
    assert s["sel_C"] == pytest.approx(1) and s["sel_P"] == pytest.approx(1) and s["sel_F"] == pytest.approx(1)
    assert s["sel_eff_individual"] == 1 and s["selts_merge_rate"] == 0


def test_merging_separable_targets_costs_purity_not_completeness(two_separable):
    s, d = score(two_separable, [0, 0, 0, 0])
    assert s["sel_C"] == pytest.approx(1)
    assert s["sel_P"] == pytest.approx(10 / 16)            # main target 10 of 16 GeV
    assert s["selts_merge_rate"] == 1
    assert s["selts_other_sel_frac"] == pytest.approx(6 / 16)


def test_splitting_costs_completeness_and_shows_in_fragmentation(mk):
    ev = mk(2, [(0, 0, 4.0), (1, 0, 4.0)], [{}])
    s, d = score(ev, [0, 1])
    assert s["sel_C"] == pytest.approx(0.5) and s["sel_P"] == pytest.approx(1) and s["sel_F"] == pytest.approx(2)
    assert s["sel_split_rate"] == 1 and s["sel_eff_individual"] == 0


def test_crumbs_are_cheap_in_energy(mk):
    ev = mk(3, [(0, 0, 9.8), (1, 0, 0.1), (2, 0, 0.1)], [{}])
    s, _ = score(ev, [0, 1, 2])
    assert s["sel_C"] == pytest.approx(0.98) and 1 < s["sel_F"] < 1.05


def test_inseparable_merge_is_not_penalised(mk):
    # particle 1 embedded in particle 0: one target, so one trackster is perfect
    ev = mk(2, [(0, 0, 8.0), (0, 1, 0.5), (1, 0, 2.0)], [{}, {"origin": 7}])
    s, _ = score(ev, [0, 0])
    assert s["sel_P"] == pytest.approx(1) and s["selts_merge_rate"] == 0 and s["sel_multi_origin_frac"] == 1


def test_inseparable_pileup_is_reported_not_penalised(mk):
    ev = mk(2, [(0, 0, 8.0), (0, 1, 0.5), (1, 0, 2.0)], [{"signal": 1}, {"signal": 0, "evt": 12}])
    s, _ = score(ev, [0, 0])
    assert s["sel_P"] == pytest.approx(1) and s["sig_pileup_inside_frac"] == pytest.approx(0.5 / 10.5)


def test_separable_pileup_merged_into_signal_costs_signal_purity(mk):
    ev = mk(3, [(0, 0, 6.0), (1, 0, 4.0), (2, 1, 2.0)], [{"signal": 1}, {"signal": 0, "evt": 5}])
    s, _ = score(ev, [0, 0, 0])
    assert s["sig_P"] == pytest.approx(10 / 12) and s["sigts_pileup_frac"] == pytest.approx(2 / 12)
    assert s["sig_C"] == pytest.approx(1)


def test_no_truth_energy_costs_purity_and_makes_fakes(mk):
    ev = mk(3, [(0, 0, 6.0), (1, 0, 4.0)], [{}], no_truth=[0, 0, 5.0])
    s, d = score(ev, [0, 0, 0])
    assert s["sel_P"] == pytest.approx(10 / 15) and s["selts_notruth_frac"] == pytest.approx(5 / 15)
    s2, d2 = score(ev, [0, 0, 1])     # the no-truth cluster on its own is a fake trackster
    assert s2["sel_P"] == pytest.approx(1) and d2["sums"]["ts_fake"] == 1


def test_unclustered_energy_costs_completeness(mk):
    ev = mk(3, [(0, 0, 6.0), (1, 0, 3.0), (2, 0, 1.0)], [{}])
    s, _ = score(ev, [0, 0, -1])
    assert s["sel_C"] == pytest.approx(0.9) and s["sel_unclustered_frac"] == pytest.approx(0.1)


def test_trackster_weights_share_a_layer_cluster(mk):
    ev = mk(2, [(0, 0, 4.0), (1, 0, 4.0)], [{}])
    ts = Tracksters(np.zeros(2), np.array([0, 2, 3]), np.array([0, 1, 1]), np.array([1.0, 0.5, 0.5]))
    d = metrics.evaluate(ev, ts, truth.build(ev), keep_details=True)
    assert d["tracksters"]["E"] == pytest.approx([6.0, 2.0])


def test_sums_are_additive(two_separable, mk):
    # default config: every target here (10, 6, 8 GeV) is above 5 GeV
    ev2 = mk(2, [(0, 0, 4.0), (1, 0, 4.0)], [{}])
    a = metrics.evaluate(two_separable, Tracksters.from_labels([0, 0, 0, 0]), truth.build(two_separable))
    b = metrics.evaluate(ev2, Tracksters.from_labels([0, 1]), truth.build(ev2))
    c = metrics.combine([a, b])
    assert c["sel_E"] == pytest.approx(a["sel_E"] + b["sel_E"]) and c["sel_n"] == 3
    assert metrics.summary(c)["sel_C"] == pytest.approx((a["sel_best"] + b["sel_best"]) / (a["sel_E"] + b["sel_E"]))


def test_no_tracksters_at_all(two_separable):
    s, _ = score(two_separable, [-1, -1, -1, -1])
    assert s["sel_C"] == 0 and np.isnan(s["sel_P"]) and s["sel_lost_rate"] == 1


# ---- selection: targets with E_t >= 5 GeV are scored, the rest is noise ----

SEL = dict(select_energy=5.0)


def test_default_selection_is_5_GeV_and_objectives_2_GeV():
    assert metrics.MetricConfig().select_energy == 5.0 and metrics.MetricConfig().objective_energy == 2.0


def test_selection_is_on_the_target_after_linking(mk):
    # particles 0 and 1 (3 GeV each) are inseparable: one 6 GeV target, selected although
    # neither particle passes alone. Particle 2 sits exactly at the cut, particle 3 below it.
    ev = mk(4, [(0, 0, 2.0), (0, 1, 2.0), (1, 0, 1.0), (1, 1, 1.0), (2, 2, 5.0), (3, 3, 4.0)], [{}, {}, {}, {}])
    s, d = score(ev, [0, 0, 1, 2], **SEL)
    E, sel = d["targets"]["E"], d["targets"]["selected"]
    assert sorted(E[sel]) == pytest.approx([5.0, 6.0]) and list(E[~sel]) == pytest.approx([4.0])
    assert s["sel_energy_frac"] == pytest.approx(11 / 15)


def test_selection_uses_reachable_energy(mk):
    # particle 0 has 10 GeV, but 6 of it is in a masked layer cluster: 4 GeV reachable, not selected
    ev = mk(3, [(0, 0, 4.0), (1, 0, 6.0), (2, 1, 6.0)], [{}, {}], mask=[1, 0, 1])
    s, d = score(ev, [0, -1, 1], **SEL)
    assert d["sums"]["sel_n"] == 1 and d["sums"]["sel_E"] == pytest.approx(6.0)


def test_soft_inseparable_particle_is_part_of_the_hard_target(mk):
    # a 1 GeV pileup particle embedded in a 20 GeV one: no contamination
    ev = mk(2, [(0, 0, 16.0), (0, 1, 1.0), (1, 0, 4.0)], [{"signal": 1}, {"signal": 0, "evt": 9}])
    s, _ = score(ev, [0, 0], **SEL)
    assert s["sel_P"] == pytest.approx(1) and s["selts_soft_frac"] == 0
    assert s["sel_pileup_inside_frac"] == pytest.approx(1 / 21)


def test_soft_separable_target_merged_costs_purity_only(mk):
    ev = mk(3, [(0, 0, 12.0), (1, 0, 8.0), (2, 1, 2.0)], [{"signal": 1}, {"signal": 0, "evt": 4}])
    s, _ = score(ev, [0, 0, 0], **SEL)
    assert s["sel_C"] == pytest.approx(1) and s["sel_P"] == pytest.approx(20 / 22)
    assert s["selts_soft_frac"] == pytest.approx(2 / 22) and s["selts_other_sel_frac"] == 0


def test_soft_targets_do_not_enter_completeness_or_fragmentation(mk):
    # a perfect 10 GeV target next to a 4 GeV one split in two
    ev = mk(4, [(0, 0, 6.0), (1, 0, 4.0), (2, 1, 2.0), (3, 1, 2.0)], [{}, {}])
    s, d = score(ev, [0, 0, 1, 2], **SEL)
    assert s["sel_C"] == pytest.approx(1) and s["sel_F"] == pytest.approx(1) and s["sel_P"] == pytest.approx(1)
    assert s["all_C"] == pytest.approx(12 / 14)   # but they are monitored


def test_tracksters_led_by_a_soft_target_are_not_scored(mk):
    # 2 of the 10 GeV target end up in a trackster whose main target is the 4 GeV one:
    # that trackster is not in P; the hard target pays in C and F
    ev = mk(3, [(0, 0, 8.0), (1, 0, 2.0), (2, 1, 4.0)], [{}, {}])
    s, d = score(ev, [0, 1, 1], **SEL)
    assert d["sums"]["selts_n"] == 1 and d["sums"]["ts_n"] == 2
    assert s["sel_P"] == pytest.approx(1) and s["sel_C"] == pytest.approx(0.8)
    assert s["sel_F"] == pytest.approx(1 / (0.8 ** 2 + 0.2 ** 2))


def test_hard_pileup_is_scored_like_signal(mk):
    # a perfect 10 GeV signal target next to a 6 GeV pileup target split in two
    ev = mk(4, [(0, 0, 5.0), (1, 0, 5.0), (2, 1, 3.0), (3, 1, 3.0)], [{"signal": 1}, {"signal": 0, "evt": 3}])
    s, _ = score(ev, [0, 0, 1, 2], **SEL)
    assert s["sel_C"] == pytest.approx(13 / 16) and s["sel_F"] == pytest.approx((10 * 1 + 6 * 2) / 16)
    assert s["sel_P"] == pytest.approx(1) and s["sel_pileup_inside_frac"] == pytest.approx(6 / 16)
    # the signal-only diagnostics still see the signal alone
    assert s["sig_C"] == pytest.approx(1) and s["sig_F"] == pytest.approx(1) and s["sig_P"] == pytest.approx(1)


def test_selected_trackster_energy_decomposes(mk):
    # main 10 GeV + another selected 6 GeV + soft 2 GeV + 2 GeV no-truth, all in one trackster
    ev = mk(4, [(0, 0, 10.0), (1, 1, 6.0), (2, 2, 2.0)], [{}, {}, {}], no_truth=[0, 0, 0, 2.0])
    s, _ = score(ev, [0, 0, 0, 0], **SEL)
    parts = (s["sel_P"], s["selts_other_sel_frac"], s["selts_soft_frac"], s["selts_notruth_frac"])
    assert parts == pytest.approx((10 / 20, 6 / 20, 2 / 20, 2 / 20)) and sum(parts) == pytest.approx(1)


# ---- objectives: signal truth groups against the ideal clustering of whole layer clusters ----

SIG, PU = {"signal": 1}, {"signal": 0, "evt": 3}


def objectives(ev, labels, **cfg):
    cfg.setdefault("select_energy", 0.0)
    cfg.setdefault("objective_energy", 0.0)
    return score(ev, labels, **cfg)


def test_ideal_clustering_scores_one(mk):
    # signal 10 GeV over clusters 0-1, pileup 3 GeV in cluster 1 (signal leads it: unavoidable) and alone in cluster 2
    ev = mk(3, [(0, 0, 6.0), (1, 0, 4.0), (1, 1, 1.0), (2, 1, 2.0)], [SIG, PU])
    s, d = objectives(ev, [0, 0, 1])
    assert s["C"] == pytest.approx(1) and s["P"] == pytest.approx(1)
    assert s["obj_unavoidable_frac"] == pytest.approx(1 / 11)   # the pileup under the signal costs nothing
    assert s["obj_ideal_purity"] == pytest.approx(10 / 11)
    assert s["sel_P"] == pytest.approx(12 / 13)   # the previous purity charged the unavoidable 1 GeV


def test_only_signal_targets_are_scored(mk):
    # a pileup shower split in two does not change the objectives
    ev = mk(4, [(0, 0, 5.0), (1, 0, 5.0), (2, 1, 3.0), (3, 1, 3.0)], [SIG, PU])
    s, d = objectives(ev, [0, 0, 1, 2])
    assert d["sums"]["obj_n"] == 1 and s["C"] == pytest.approx(1) and s["P"] == pytest.approx(1)
    assert s["sel_C"] < 1   # the previous objective sees the split pileup


def test_target_must_be_mostly_signal(mk):
    # an inseparable 0.5 GeV signal particle inside a 9 GeV pileup shower: a pileup target, not scored
    ev = mk(2, [(0, 1, 6.0), (0, 0, 0.5), (1, 1, 3.0)], [SIG, PU])
    s, d = objectives(ev, [0, 0])
    assert d["sums"]["obj_n"] == 0 and np.isnan(s["C"])
    assert s["signal_in_pileup_targets_frac"] == pytest.approx(1)


def test_separable_pileup_taken_in_costs_purity(mk):
    # pileup 2 GeV alone in cluster 2: putting it in the signal trackster is avoidable
    ev = mk(3, [(0, 0, 6.0), (1, 0, 4.0), (2, 1, 2.0)], [SIG, PU])
    s, d = objectives(ev, [0, 0, 0])
    assert s["C"] == pytest.approx(1) and s["P"] == pytest.approx(1 - 2 / 12)


def test_no_truth_cluster_taken_in_costs_purity_but_no_truth_under_the_signal_does_not(mk):
    ev = mk(3, [(0, 0, 6.0), (1, 0, 4.0)], [SIG], no_truth=[1.0, 0, 3.0])
    s, d = objectives(ev, [0, 0, 1])          # cluster 0 holds 1 GeV noise, under the signal: free
    assert s["P"] == pytest.approx(1)
    s2, _ = objectives(ev, [0, 0, 0])         # the noise-only cluster 2 taken in: avoidable
    assert s2["P"] == pytest.approx(1 - 3 / 14)


def test_splitting_into_clean_pieces_keeps_completeness(mk):
    # two pieces of one signal target: both are its pieces, completeness counts both; F reports the split
    ev = mk(2, [(0, 0, 4.0), (1, 0, 4.0)], [SIG])
    s, d = objectives(ev, [0, 1])
    assert s["C"] == pytest.approx(1) and s["P"] == pytest.approx(1)
    assert s["F"] == pytest.approx(2) and s["obj_tracksters_per_group"] == 2
    assert s["sel_C"] == pytest.approx(0.5)   # the previous objective took the best trackster only


def test_inaccurate_merge_costs_purity_and_completeness(mk):
    # signal B (6 GeV over clusters 2-4): one trackster takes all of A and only cluster 2 of B (2 of 6 GeV);
    # B's other 4 GeV are a trackster of their own. The first does not claim B: B's 2 GeV in it are contamination.
    A, B = 0, 1
    ev = mk(5, [(0, A, 5.0), (1, A, 5.0), (2, B, 2.0), (3, B, 2.0), (4, B, 2.0)], [SIG, SIG])
    s, d = objectives(ev, [0, 0, 0, 1, 1])
    assert d["sums"]["obj_groups"] == 2 and s["obj_merged_group_frac"] == 0
    assert s["P"] == pytest.approx(1 - 2 / 16)
    assert s["C"] == pytest.approx(14 / 16)   # B's 2 GeV in A's trackster are not collected for B


def test_merged_chain_is_one_group(mk):
    # an electron and its brem photon on one axis: the photon leads the late clusters. A trackster holding both
    # is one group; splitting them at the hand-over is also fine, as two groups.
    e, g = 0, 1
    ev = mk(4, [(0, e, 6.0), (1, e, 4.0), (1, g, 1.0), (2, e, 1.0), (2, g, 4.0), (3, g, 3.0)], [SIG, SIG])
    s, d = objectives(ev, [0, 0, 0, 0])
    assert s["C"] == pytest.approx(1) and s["P"] == pytest.approx(1) and d["sums"]["obj_groups"] == 1
    s2, d2 = objectives(ev, [0, 0, 1, 1])
    assert d2["sums"]["obj_groups"] == 2 and s2["C"] == pytest.approx(1) and s2["P"] == pytest.approx(1)


def test_pileup_never_joins_a_group(mk):
    # separable pileup taken in whole is still contamination, however accurately it is taken
    ev = mk(3, [(0, 0, 6.0), (1, 0, 4.0), (2, 1, 5.0)], [SIG, PU])
    s, d = objectives(ev, [0, 0, 0])
    assert s["P"] == pytest.approx(1 - 5 / 15) and d["sums"]["obj_groups"] == 1


def test_energy_outside_the_ideal_cluster_is_not_credited(mk):
    # the signal leads cluster 0; cluster 1 is led by pileup (2 vs 1 GeV of signal)
    ev = mk(2, [(0, 0, 8.0), (1, 0, 1.0), (1, 1, 2.0)], [SIG, PU])
    s, d = objectives(ev, [0, 0])
    assert s["obj_ideal_completeness"] == pytest.approx(8 / 9)
    assert s["C"] == pytest.approx(1)                       # all of what an ideal clustering could collect
    assert s["P"] == pytest.approx(1 - 2 / 11)               # the pileup-led cluster brought in 2 GeV of pileup
    assert s["obj_own_outside_frac"] == pytest.approx(1 / 11)


def test_unclustered_ideal_energy_costs_completeness(mk):
    ev = mk(3, [(0, 0, 6.0), (1, 0, 3.0), (2, 0, 1.0)], [SIG])
    s, _ = objectives(ev, [0, 0, -1])
    assert s["C"] == pytest.approx(0.9) and s["P"] == pytest.approx(1)


def test_shared_layer_cluster_weights(mk):
    # a layer cluster shared by two tracksters (multiplicity 2) counts half in each
    ev = mk(2, [(0, 0, 4.0), (1, 0, 4.0)], [SIG])
    ts = Tracksters(np.zeros(2), np.array([0, 2, 3]), np.array([0, 1, 1]), np.array([1.0, 0.5, 0.5]))
    d = metrics.evaluate(ev, ts, truth.build(ev), metrics.MetricConfig(select_energy=0.0), keep_details=True)
    s = metrics.summary(d["sums"])
    assert s["C"] == pytest.approx(1) and s["P"] == pytest.approx(1)


def test_objective_selection_energy(mk):
    # a 4 GeV signal target in a trackster of its own is below 5 GeV: not scored, reported as unscored signal
    ev = mk(2, [(0, 0, 10.0), (1, 1, 4.0)], [SIG, SIG])
    s, d = score(ev, [0, 1], objective_energy=5.0)
    assert d["sums"]["obj_n"] == 1 and s["signal_unscored_frac"] == pytest.approx(4 / 14)
    s2, d2 = score(ev, [0, 0], objective_energy=5.0)   # taken whole into the 10 GeV one, which it does not touch:
    assert s2["P"] == pytest.approx(1 - 4 / 14) and s2["signal_unscored_frac"] == pytest.approx(4 / 14)  # contamination


def test_targets_apart_are_not_joined_even_if_held_whole(mk):
    # two signal showers sharing no layer cluster, both whole in one trackster: not one object; the smaller is
    # contamination of the larger (and has no trackster of its own)
    ev = mk(4, [(0, 0, 5.0), (1, 0, 5.0), (2, 1, 3.0), (3, 1, 3.0)], [SIG, SIG])
    s, d = objectives(ev, [0, 0, 0, 0])
    assert d["sums"]["obj_groups"] == 2 and s["obj_merged_group_frac"] == 0
    assert s["P"] == pytest.approx(1 - 6 / 16) and s["C"] == pytest.approx(10 / 16)


def test_grazing_targets_held_whole_are_not_joined(mk):
    # two showers sharing cluster 2, where the smaller one (B, 6 GeV) has only 2 GeV: they graze, they do not
    # overlap. A trackster holding both whole does not join them: B is contamination of A
    ev = mk(4, [(0, 0, 5.0), (1, 0, 4.0), (2, 0, 1.0), (2, 1, 2.0), (3, 1, 4.0)], [SIG, SIG])
    s, d = objectives(ev, [0, 0, 0, 0])
    assert d["sums"]["obj_groups"] == 2 and s["obj_merged_group_frac"] == 0
    assert s["P"] == pytest.approx(1 - 6 / 16)


def test_overlapping_targets_held_whole_are_one_group(mk):
    # B (4 GeV) has 3 GeV in cluster 1, which A shares: 75% of the smaller one in shared clusters
    ev = mk(3, [(0, 0, 6.0), (1, 0, 2.0), (1, 1, 3.0), (2, 1, 1.0)], [SIG, SIG])
    assert truth.build(ev).n == 2       # the separability test keeps them apart
    s, d = objectives(ev, [0, 0, 0])
    assert d["sums"]["obj_groups"] == 1 and s["C"] == pytest.approx(1) and s["P"] == pytest.approx(1)


def test_targets_overlapping_through_a_third_are_one_group(mk):
    # A and C share nothing, but B overlaps A (3 of its 4 GeV in A's cluster 1) and C overlaps B (2 of its 2.5 GeV
    # in cluster 2): a trackster holding all three joins them
    A, B, C = 0, 1, 2
    ev = mk(4, [(0, A, 6.0), (1, A, 2.0), (1, B, 3.0), (2, B, 1.0), (2, C, 2.0), (3, C, 0.5)], [SIG, SIG, SIG])
    assert truth.build(ev).n == 3
    s, d = objectives(ev, [0, 0, 0, 0])
    assert d["sums"]["obj_groups"] == 1 and s["C"] == pytest.approx(1) and s["P"] == pytest.approx(1)

