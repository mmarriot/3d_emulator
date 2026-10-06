"""The truth (V3_TRUTH_AND_METRICS.md section 4): one situation per test, with the rechits it is decided on."""
import pytest

from ticltune import truth

PU1 = dict(signal=0, evt=1)   # a particle of pileup collision 1
PU2 = dict(signal=0, evt=2)


def test_disjoint_units_are_separate_targets(mk):
    ev = mk(2, [(0, 0, 5.0), (1, 1, 3.0)])
    t = truth.build(ev)
    assert t.n == 2 and t.completeness == pytest.approx([1, 1]) and t.purity == pytest.approx([1, 1])
    assert t.E == pytest.approx([5, 3]) and not t.unreachable.any()


def test_soft_unit_hidden_under_another_of_the_same_interaction_is_merged(mk):
    # unit 1 (0.5 GeV) only in rechits where unit 0 has more: it leads none, completeness 0 -> merged
    ev = mk(2, [(0, 0, 4.0), (1, 0, 4.0), (1, 1, 0.5)])
    t = truth.build(ev)
    assert t.n == 1 and t.E[0] == pytest.approx(8.5) and t.signal[0]
    assert t.merged_at[1] == 0 and t.partner[1] == 0 and t.merged_at[0] == -1


def test_signal_hidden_under_pileup_is_never_merged_with_it(mk):
    # the same situation with the hard unit from a pileup collision: no merge across interactions; the signal unit
    # stays a target of its own, flagged unreachable, and the pileup stays pileup
    ev = mk(2, [(0, 0, 4.0), (1, 0, 4.0), (1, 1, 0.5)], units=[PU1, {}])
    t = truth.build(ev)
    assert t.n == 2 and list(t.signal) == [False, True]
    assert list(t.unreachable) == [False, True]
    assert t.E == pytest.approx([8.0, 0.5])


def test_pileup_merges_within_a_collision_but_not_across_collisions(mk):
    same = truth.build(mk(2, [(0, 0, 4.0), (1, 0, 4.0), (1, 1, 0.5)], units=[PU1, PU1]))
    other = truth.build(mk(2, [(0, 0, 4.0), (1, 0, 4.0), (1, 1, 0.5)], units=[PU1, PU2]))
    assert same.n == 1 and other.n == 2 and other.unreachable[1]


def test_bunch_crossing_is_part_of_the_interaction(mk):
    t = truth.build(mk(2, [(0, 0, 4.0), (1, 0, 4.0), (1, 1, 0.5)], units=[dict(signal=0, bx=-1), dict(signal=0)]))
    assert t.n == 2


def test_overlapping_showers_that_each_lead_most_of_themselves_stay_separate(mk):
    # two equal showers, each leading one rechit 60/40: the ideal clustering gives each 60% at 60% purity
    t = truth.build(mk(2, [(0, 0, 1.2), (0, 1, 0.8), (1, 0, 0.8), (1, 1, 1.2)]))
    assert t.n == 2 and t.completeness == pytest.approx([0.6, 0.6]) and t.purity == pytest.approx([0.6, 0.6])


def test_indistinguishable_showers_are_merged(mk):
    assert truth.build(mk(2, [(0, 0, 1.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.0)])).n == 1


def test_purity_fails_when_a_rechit_is_shared_three_ways(mk):
    # a leads the rechit but is only 1/2.8 of it: no clustering of whole rechits gives it purity > 0.5
    t = truth.build(mk(1, [(0, 0, 1.0), (0, 1, 0.9), (0, 2, 0.9)]))
    assert t.n == 1 and t.E[0] == pytest.approx(2.8)


def test_merging_is_repeated_until_every_target_passes(mk):
    # round 0: b leads no rechit and merges with a (its larger confusion, 0.6 vs 0.5); a+b then leads rechit 1
    # (0.7 vs c's 0.6), so in round 1 c keeps only 0.5 of 1.1 and merges too
    ev = mk(3, [(0, 0, 3.0), (0, 1, 0.6), (1, 0, 0.2), (1, 1, 0.5), (1, 2, 0.6), (2, 2, 0.5)])
    t = truth.build(ev)
    assert t.n == 1 and t.n_iterations == 2 and list(t.merged_at) == [-1, 0, 1]


def test_deposited_energy_counts_rechits_outside_layer_clusters(mk):
    # rechit 2 is in no layer cluster: it still counts in the target's deposited energy
    ev = mk(3, [(0, 0, 1.0), (1, 0, 2.0), (2, 0, 4.0)], lcs=[[(0, 1.0)], [(1, 1.0)]])
    assert truth.build(ev).E[0] == pytest.approx(7.0)


def test_class_follows_the_unit_with_the_most_energy(mk):
    units = [dict(pdg=22), dict(pdg=211), dict(pdg=22, kind=1), dict(pdg=13), dict(pdg=-11)]
    t = truth.build(mk(5, [(i, i, 1.0) for i in range(5)], units=units))
    assert list(t.em) == [True, False, True, False, True]
    # a merged target takes the class of its leading unit
    t = truth.build(mk(2, [(0, 0, 4.0), (1, 0, 4.0), (1, 1, 0.5)], units=[dict(pdg=211), dict(pdg=22)]))
    assert t.n == 1 and not t.em[0]


def test_units_group_their_atoms(mk):
    # two atoms of one unit (e.g. an electron and its brem photon) form one target
    ev = mk(2, [(0, 0, 3.0), (1, 1, 1.0)], atoms=[dict(unit=0), dict(unit=0, parent=0)])
    t = truth.build(ev)
    assert t.n == 1 and t.E[0] == pytest.approx(4.0)


def test_natural_pieces(mk):
    # a unit whose two atoms deposit in different rechits: two natural pieces
    far = mk(2, [(0, 0, 3.0), (1, 1, 1.0)], atoms=[dict(unit=0), dict(unit=0, parent=0)])
    assert list(truth.build(far).n_pieces) == [2]
    # the child hidden under its parent: one piece
    near = mk(2, [(0, 0, 3.0), (1, 0, 3.0), (1, 1, 0.5)], atoms=[dict(unit=0), dict(unit=0, parent=0)])
    assert list(truth.build(near).n_pieces) == [1]
    # pieces are decided on the target's own energy: another target in the same rechits does not matter
    other = mk(2, [(0, 0, 3.0), (1, 1, 1.0), (0, 2, 5.0), (1, 2, 5.0)],
               atoms=[dict(unit=0), dict(unit=0, parent=0), dict(unit=1)], units=[{}, PU1])
    assert list(truth.build(other).n_pieces) == [2, 1]


def test_ideal_labels_at_both_levels(mk):
    # rechit 1 is mostly b, but sits in a layer cluster with rechit 0 (all a): on layer clusters it goes to a
    ev = mk(2, [(0, 0, 3.0), (1, 1, 1.0), (1, 0, 0.2)], lcs=[[(0, 1.0), (1, 1.0)]])
    t = truth.build(ev)
    assert list(truth.ideal_labels(ev, t, "rh")) == [0, 1]
    assert list(truth.ideal_labels(ev, t, "lc")) == [0]
