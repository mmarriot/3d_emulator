"""Toy events built by hand, so each test states exactly the situation it checks."""
import numpy as np
import pytest

from ticltune.data import Event


def make_event(n_rh, truth=(), units=None, atoms=None, no_truth=None, lcs=None, positions=None, layers=None,
               energies=None, det=8, mask=None):
    """A toy event.

    truth: (rechit, atom, energy) triples. atoms: list of dicts {unit, parent (-1), pdg}; default one atom per unit,
    atom i in unit i. units: list of dicts {signal (1), bx (0), evt (0), pdg (211), kind (0)}; default one signal
    unit per atom. Rechit energy = its truth + no_truth (unless `energies`). lcs: list of [(rechit, fraction), ...];
    default every rechit its own layer cluster. Layer clusters take the position and layer of their first rechit."""
    tr = np.array(truth, dtype=float).reshape(-1, 3)
    tra = dict(rh=tr[:, 0].astype(np.int64), at=tr[:, 1].astype(np.int64), E=tr[:, 2])
    n_at = int(tra["at"].max()) + 1 if len(tr) else 0
    atoms = [dict(unit=i) for i in range(n_at)] if atoms is None else atoms
    n_un = max([a["unit"] for a in atoms], default=-1) + 1
    units = [{} for _ in range(n_un)] if units is None else units
    at = dict(id=np.arange(len(atoms)), pdg=np.array([a.get("pdg", 211) for a in atoms], np.int64),
              parent=np.array([a.get("parent", -1) for a in atoms], np.int64),
              unit=np.array([a["unit"] for a in atoms], np.int64))
    un = dict(id=np.arange(len(units)) + 100, pdg=np.array([u.get("pdg", 211) for u in units], np.int64),
              kind=np.array([u.get("kind", 0) for u in units], np.int64),
              signal=np.array([u.get("signal", 1) for u in units], np.int64),
              bx=np.array([u.get("bx", 0) for u in units], np.int64),
              evt=np.array([u.get("evt", 0) for u in units], np.int64))
    no_truth = np.zeros(n_rh) if no_truth is None else np.asarray(no_truth, float)
    E = (np.bincount(tra["rh"], weights=tra["E"], minlength=n_rh) + no_truth) if energies is None \
        else np.asarray(energies, float)
    un["depE"] = np.bincount(at["unit"][tra["at"]], weights=tra["E"], minlength=len(units)) if len(tr) \
        else np.zeros(len(units))
    pos = np.zeros((n_rh, 3)) + [10.0, 10.0, 350.0] if positions is None else np.asarray(positions, float)
    layer = np.ones(n_rh, np.int64) if layers is None else np.asarray(layers, np.int64)
    rh = dict(id=np.arange(n_rh, dtype=np.int64) + 5000, E=E, x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], layer=layer,
              det=np.full(n_rh, det, np.int64), lc=np.full(n_rh, -1, np.int64), noTruthE=no_truth)
    lcs = [[(c, 1.0)] for c in range(n_rh)] if lcs is None else lcs
    lch = dict(lc=np.array([l for l, m in enumerate(lcs) for _ in m], np.int64),
               rh=np.array([c for m in lcs for c, _ in m], np.int64),
               frac=np.array([f for m in lcs for _, f in m], float))
    best = np.zeros(n_rh)
    for l, m in enumerate(lcs):
        for c, f in m:
            if f > best[c]:
                best[c], rh["lc"][c] = f, l
    first = np.array([m[0][0] for m in lcs], np.int64)
    n_lc = len(lcs)
    lc = dict(E=np.bincount(lch["lc"], weights=lch["frac"] * E[lch["rh"]], minlength=n_lc),
              x=pos[first, 0], y=pos[first, 1], z=pos[first, 2], layer=layer[first],
              side=(pos[first, 2] > 0).astype(np.int64), det=np.full(n_lc, det, np.int64),
              algo=np.full(n_lc, 6, np.int64), seed=np.arange(n_lc, dtype=np.int64) + 1000,
              mask=np.ones(n_lc) if mask is None else np.asarray(mask, float))
    return Event(1, lc=lc, rh=rh, lch=lch, at=at, un=un, tra=tra)


@pytest.fixture
def mk():
    return make_event
