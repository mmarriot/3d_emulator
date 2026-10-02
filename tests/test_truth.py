"""Target definition: the ideal-clustering test. Each test is one situation, with the cells it is decided on."""
import pytest

from ticltune import truth


def test_disjoint_particles_are_separate_targets(mk):
    ev = mk(2, [(0, 0, 5.0), (1, 1, 3.0)], [{}, {}])
    t = truth.build(ev)
    assert t.n == 2 and t.completeness == pytest.approx([1, 1]) and t.purity == pytest.approx([1, 1])


def test_particle_hidden_under_another_is_merged(mk):
    # particle 1 (0.5 GeV) only in layer clusters where particle 0 has more: it leads none, completeness 0
    ev = mk(2, [(0, 0, 4.0), (1, 0, 4.0), (1, 1, 0.5)], [{}, {}])
    t = truth.build(ev)
    assert t.n == 1 and t.E[0] == pytest.approx(8.5)
    assert t.merged_at[1] == 0 and t.partner[1] == 0 and t.fail_completeness[1] == pytest.approx(0)
    assert t.merged_at[0] == -1


def test_overlapping_showers_that_each_lead_most_of_themselves_stay_separate(mk):
    # two equal showers on top of each other, each leading one cluster 60/40: the ideal clustering gives
    # each 60% of itself at 60% purity, so a perfect algorithm separates them
    ev = mk(2, [(0, 0, 1.2), (0, 1, 0.8), (1, 0, 0.8), (1, 1, 1.2)], [{}, {}])
    t = truth.build(ev)
    assert t.n == 2 and t.completeness == pytest.approx([0.6, 0.6]) and t.purity == pytest.approx([0.6, 0.6])


def test_indistinguishable_showers_are_merged(mk):
    # equal energy in every cell: whoever gets the cells, the other has completeness 0
    ev = mk(2, [(0, 0, 1.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.0)], [{}, {}])
    assert truth.build(ev).n == 1


def test_small_particle_leading_its_own_cell_stays_separate(mk):
    # the small particle shares a cluster with a big one, but most of it (1.0 of 1.1) is in a cluster it leads
    ev = mk(2, [(0, 0, 9.0), (0, 1, 0.1), (1, 1, 1.0)], [{}, {}])
    t = truth.build(ev)
    assert t.n == 2 and sorted(t.completeness) == pytest.approx([1.0 / 1.1, 1.0])


def test_purity_fails_when_a_cell_is_shared_three_ways(mk):
    # a leads the cluster but is only 1/2.8 of it: no clustering of whole cells gives it purity > 0.5
    ev = mk(1, [(0, 0, 1.0), (0, 1, 0.9), (0, 2, 0.9)], [{}, {}, {}])
    t = truth.build(ev)
    assert t.fail_purity[0] == pytest.approx(1 / 2.8)
    assert t.n == 1  # a merges with b or c, and the other one leads nothing any more


def test_soft_particle_between_two_showers_joins_one_and_does_not_chain_them(mk):
    # S is half in A's cluster, half in B's, leading neither: merged with one of them; A and B stay apart
    A, B, S = 0, 1, 2
    ev = mk(2, [(0, A, 5.0), (0, S, 0.5), (1, S, 0.5), (1, B, 5.0)], [{}, {}, {}])
    t = truth.build(ev)
    assert t.n == 2 and t.member_of[A] != t.member_of[B] and t.member_of[S] in (t.member_of[A], t.member_of[B])


def test_merging_is_repeated_until_every_target_passes(mk):
    # round 0: a (1 GeV) leads nothing, most of it is under c: a merges with c. b (1.95 GeV) passes, leading
    # clusters 1 and 2. Round 1: c+a now leads cluster 1 (0.7 + 0.4 > 1.0), b keeps 0.95 of 1.95 and fails.
    a, b, c = 0, 1, 2
    ev = mk(3, [(0, c, 3.0), (0, a, 0.6), (1, a, 0.4), (1, c, 0.7), (1, b, 1.0), (2, b, 0.95)], [{}, {}, {}])
    t = truth.build(ev)
    assert t.merged_at[a] == 0 and t.partner[a] == c
    assert t.merged_at[b] == 1 and t.fail_completeness[b] == pytest.approx(0.95 / 1.95)
    assert t.n == 1 and t.n_iterations == 2


def test_inseparable_pileup_joins_the_signal_target(mk):
    ev = mk(2, [(0, 0, 8.0), (0, 1, 0.5), (1, 0, 2.0)], [{"signal": 1}, {"signal": 0, "evt": 37}])
    t = truth.build(ev)
    assert t.n == 1 and t.is_signal[0]
    assert t.E_pileup[0] == pytest.approx(0.5) and t.n_interactions[0] == 2


def test_masked_energy_is_unreachable_not_target_energy(mk):
    ev = mk(2, [(0, 0, 4.0), (1, 0, 6.0)], [{}], mask=[1, 0])
    t = truth.build(ev)
    assert t.E[0] == pytest.approx(4.0) and t.E_unreachable[0] == pytest.approx(6.0)


def test_particle_with_only_masked_energy_is_not_a_target(mk):
    ev = mk(2, [(0, 0, 4.0), (1, 1, 6.0)], [{}, {}], mask=[1, 0])
    t = truth.build(ev)
    assert t.n == 1 and t.member_of[1] == -1 and t.E_untargeted == pytest.approx(6.0)


def test_targets_have_no_energy_threshold(mk):
    # selection happens in the metrics; an isolated 50 MeV particle is a target of its own
    ev = mk(2, [(0, 0, 20.0), (1, 1, 0.05)], [{}, {}])
    t = truth.build(ev)
    assert t.n == 2 and sorted(t.E) == pytest.approx([0.05, 20.0])


def test_rechits_can_separate_what_layer_clusters_cannot(mk):
    # one layer cluster holds both particles (3 and 2 GeV), but in different rechits
    ev = mk(1, [(0, 0, 3.0), (0, 1, 2.0)], [{}, {}], rechits=([0, 0], [(0, 0, 3.0), (1, 1, 2.0)]))
    assert truth.build(ev, level="lc").n == 1
    t = truth.build(ev, level="rh")
    assert t.n == 2 and t.Tc.shape == (2, 2)
    assert t.T.toarray().ravel() == pytest.approx([3.0, 2.0])  # the metrics still see layer clusters


def test_rechits_of_masked_layer_clusters_are_unreachable(mk):
    ev = mk(2, [(0, 0, 3.0), (1, 0, 1.0)], [{}], mask=[1, 0], rechits=([0, 1], [(0, 0, 3.0), (1, 0, 1.0)]))
    t = truth.build(ev, level="rh")
    assert t.E[0] == pytest.approx(3.0) and t.E_unreachable[0] == pytest.approx(1.0)
