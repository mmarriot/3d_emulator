"""Toy events built by hand, so each test states exactly the situation it checks."""
import numpy as np
import pytest

from ticltune.data import Event, LC_FIELDS, BP_FIELDS, RH_FIELDS


def make_event(n_lc, truth, particles, no_truth=None, mask=None, positions=None, layers=None, energies=None,
               rechits=None):
    """truth: list of (lc, particle, energy); particles: list of dicts with optional keys
    signal (1), bx (0), evt (0), origin (row), originPdg (211). rechits: (layer cluster of each rechit,
    list of (rechit, particle, energy))."""
    no_truth = np.zeros(n_lc) if no_truth is None else np.asarray(no_truth, float)
    tr = np.array(truth, dtype=float).reshape(-1, 3)
    tr_lc, tr_bp, tr_E = tr[:, 0].astype(int), tr[:, 1].astype(int), tr[:, 2]
    rec = np.bincount(tr_lc, weights=tr_E, minlength=n_lc) + no_truth
    pos = np.zeros((n_lc, 3)) + [10.0, 10.0, 350.0] if positions is None else np.asarray(positions, float)
    lc = dict(E=rec if energies is None else np.asarray(energies, float), x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
              layer=np.ones(n_lc, int) if layers is None else np.asarray(layers, int),
              side=(pos[:, 2] > 0).astype(int), det=np.full(n_lc, 8), nhits=np.full(n_lc, 3),
              algo=np.full(n_lc, 6), seed=np.arange(n_lc) + 1000,
              mask=np.ones(n_lc) if mask is None else np.asarray(mask, float),
              noTruthE=no_truth, recE=rec)
    nb = len(particles)
    bp = {f: np.zeros(nb) for f in BP_FIELDS}
    for i, p in enumerate(particles):
        bp["id"][i] = i
        bp["signal"][i] = p.get("signal", 1)
        bp["bx"][i] = p.get("bx", 0)
        bp["evt"][i] = p.get("evt", 0)
        bp["origin"][i] = p.get("origin", i)
        bp["originPdg"][i] = p.get("originPdg", 211)
    bp = {k: (v.astype(int) if k in ("id", "signal", "bx", "evt", "origin", "originPdg", "pdg", "kind") else v)
          for k, v in bp.items()}
    assert set(lc) == set(LC_FIELDS)
    ev = Event(1, 1, 1, lc, tr_lc, tr_bp, tr_E, bp)
    if rechits is not None:
        rh_lc, rtr = np.asarray(rechits[0], int), np.array(rechits[1], dtype=float).reshape(-1, 3)
        ev.rh = {f: np.zeros(len(rh_lc)) for f in RH_FIELDS}
        ev.rh["lc"] = rh_lc
        ev.trh_rh, ev.trh_bp, ev.trh_E = rtr[:, 0].astype(int), rtr[:, 1].astype(int), rtr[:, 2]
        ev.rh["E"] = np.bincount(ev.trh_rh, weights=ev.trh_E, minlength=len(rh_lc))
    return ev


@pytest.fixture
def mk():
    return make_event
