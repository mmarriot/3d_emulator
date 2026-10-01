"""Target definition: inseparability M and grouping."""
import numpy as np
import pytest

from ticltune import truth


def test_disjoint_particles_are_separate_targets(mk):
    ev = mk(2, [(0, 0, 5.0), (1, 1, 3.0)], [{}, {}])
    t = truth.build(ev, tau=0.9)
    assert t.n == 2
    assert len(t.pairs["M"]) == 0  # never share a layer cluster


def test_embedded_particle_joins_the_larger(mk):
    # particle 1 (0.5 GeV) lies entirely in layer clusters where particle 0 has more energy
    ev = mk(3, [(0, 0, 4.0), (1, 0, 4.0), (1, 1, 0.3), (2, 0, 2.0), (2, 1, 0.2)], [{}, {}])
    t = truth.build(ev, tau=0.9)
    assert t.pairs["M"][0] == pytest.approx(1.0)
    assert t.n == 1 and t.E[0] == pytest.approx(10.5)


def test_partial_overlap_depends_on_tau(mk):
    # equal particles, half of each in a shared cluster (equal amounts there)
    ev = mk(3, [(0, 0, 2.0), (1, 0, 2.0), (1, 1, 2.0), (2, 1, 2.0)], [{}, {}])
    assert truth.build(ev, tau=0.9).n == 2
    ev._cache.clear()
    t = truth.build(ev, tau=0.4)
    assert t.pairs["M"][0] == pytest.approx(0.5) and t.n == 1


def test_overlap_measured_by_min_not_by_co_occupancy(mk):
    # the small particle shares a cluster but holds more there than the big one: min() counts the
    # big one's 0.1 only, so M = 0.1 / 1.0 and they stay separate
    ev = mk(2, [(0, 0, 9.0), (1, 0, 0.1), (1, 1, 1.0)], [{}, {}])
    t = truth.build(ev, tau=0.9)
    assert t.pairs["M"][0] == pytest.approx(0.1) and t.n == 2


def test_chaining_links_through_a_shared_member(mk):
    # particle 1 (1 GeV) sits half in particle 0's cluster and half in particle 2's:
    # M(0,1) = M(1,2) = 0.5, while 0 and 2 never touch
    ev = mk(2, [(0, 0, 5.0), (0, 1, 0.5), (1, 1, 0.5), (1, 2, 5.0)], [{}, {}, {}])
    t = truth.build(ev, tau=0.5)
    assert t.n == 1 and t.n_members[0] == 3


def test_inseparable_pileup_joins_the_signal_target(mk):
    ev = mk(2, [(0, 0, 8.0), (0, 1, 0.5), (1, 0, 2.0)], [{"signal": 1}, {"signal": 0, "evt": 37}])
    t = truth.build(ev, tau=0.9)
    assert t.n == 1 and t.is_signal[0]
    assert t.E_pileup[0] == pytest.approx(0.5) and t.n_interactions[0] == 2


def test_masked_energy_is_unreachable_not_target_energy(mk):
    ev = mk(2, [(0, 0, 4.0), (1, 0, 6.0)], [{}], mask=[1, 0])
    t = truth.build(ev, tau=0.9)
    assert t.E[0] == pytest.approx(4.0) and t.E_unreachable[0] == pytest.approx(6.0)


def test_particle_with_only_masked_energy_is_not_a_target(mk):
    ev = mk(2, [(0, 0, 4.0), (1, 1, 6.0)], [{}, {}], mask=[1, 0])
    t = truth.build(ev, tau=0.9)
    assert t.n == 1 and t.member_of[1] == -1


def test_targets_have_no_energy_threshold(mk):
    # selection happens in the metrics; a 50 MeV particle is still a target of its own
    ev = mk(2, [(0, 0, 20.0), (1, 1, 0.05)], [{}, {}])
    t = truth.build(ev, tau=0.9)
    assert t.n == 2 and sorted(t.E) == pytest.approx([0.05, 20.0])
