"""Events as read from the trackster-building truth ntuple (CMSSW TruthMetrics/Ntuple, TracksterTruthNtuplizer).

An `Event` holds plain numpy arrays: every HGCAL rechit, the layer clusters with their rechit fractions, the truth
(atoms, units and the sparse (rechit, atom, energy) table) and, for closure and baselines, what CMSSW produced.
Nothing here depends on clustering parameters: an event is loaded once and clustered and scored many times.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import scipy.sparse as sp

# DetId::Detector values of HGCAL, in the order of the CLUEstering sigmaT list (EE, HSi, HSci)
HGCAL_DETECTORS = (8, 9, 10)
SIGMA_INDEX = {8: 0, 9: 1, 10: 2}
EM_PDG = (11, 22, 111)

LC_FIELDS = ("E", "x", "y", "z", "layer", "side", "det", "algo", "seed", "mask")
RH_FIELDS = ("id", "E", "x", "y", "z", "layer", "det", "lc", "noTruthE")
AT_FIELDS = ("id", "pdg", "parent", "unit")
UN_FIELDS = ("id", "pdg", "kind", "signal", "bx", "evt", "depE")


@dataclass
class Tracksters:
    """A CMSSW trackster collection as layer-cluster lists; `weight` is the fraction of the layer cluster the
    trackster takes (1 / vertex multiplicity)."""
    offsets: np.ndarray  # trackster k owns lc[offsets[k]:offsets[k+1]]
    lc: np.ndarray
    weight: np.ndarray

    @property
    def n(self):
        return len(self.offsets) - 1

    def owner(self):
        """Trackster index of every entry of `lc`."""
        return np.repeat(np.arange(self.n), np.diff(self.offsets))

    def matrix(self, n_lc):
        """(n tracksters x n_lc) sparse weights."""
        return sp.csr_matrix((self.weight, (self.owner(), self.lc)), shape=(self.n, n_lc))


@dataclass
class Event:
    event: int
    lc: Dict[str, np.ndarray]   # LC_FIELDS
    rh: Dict[str, np.ndarray]   # RH_FIELDS: every HGCAL rechit; lc = layer cluster holding most of it, -1 if none
    lch: Dict[str, np.ndarray]  # layer-cluster membership: lc, rh, frac
    at: Dict[str, np.ndarray]   # AT_FIELDS: atoms (truth-graph particles with energy in rechits); parent = atom row
    un: Dict[str, np.ndarray]   # UN_FIELDS: units; kind 0 final state, 1 pi0 daughter, 2 root
    tra: Dict[str, np.ndarray]  # (rechit, atom, energy) table: rh, at, E
    clue_assignment: Optional[np.ndarray] = None  # CMSSW CLUEstering assignment per layer cluster, if stored
    cmssw: Dict[str, Tracksters] = field(default_factory=dict)
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def n_lc(self):
        return len(self.lc["E"])

    @property
    def n_rh(self):
        return len(self.rh["E"])

    def lc_eligible(self):
        """Layer clusters the CLUE3DHigh step may use: HGCAL and passing its mask (>= 2 hits)."""
        return (self.lc["mask"] > 0) & np.isin(self.lc["det"], HGCAL_DETECTORS)

    def lc_to_rh(self):
        """(n_lc x n_rh) sparse rechit fractions of the layer clusters."""
        if "F" not in self._cache:
            self._cache["F"] = sp.csr_matrix((self.lch["frac"], (self.lch["lc"], self.lch["rh"])),
                                             shape=(self.n_lc, self.n_rh))
        return self._cache["F"]


_BRANCHES = ([f"lc_{f}" for f in LC_FIELDS] + [f"rh_{f}" for f in RH_FIELDS] + ["lch_lc", "lch_rh", "lch_frac"]
             + [f"at_{f}" for f in AT_FIELDS] + [f"un_{f}" for f in UN_FIELDS]
             + ["tra_rh", "tra_at", "tra_E", "event"])
_INT = {"lc_layer", "lc_side", "lc_det", "lc_algo", "rh_layer", "rh_det", "rh_lc", "lch_lc", "lch_rh", "at_id",
        "at_pdg", "at_parent", "at_unit", "un_id", "un_pdg", "un_kind", "un_signal", "un_bx", "un_evt", "tra_rh",
        "tra_at"}


def load(path, tree="tracksterTruthNtuplizer/events", max_events=0, entry_start=0, collections=None):
    """Read events from one ntuple file. collections: CMSSW trackster collections to keep (default: all stored)."""
    import uproot

    t = uproot.open(path)[tree]
    keys = set(t.keys())
    missing = [b for b in _BRANCHES if b not in keys]
    if missing:
        raise ValueError(f"{path}: not a v3 ntuple (missing {missing[:4]}...): rerun TracksterTruthNtuplizer")
    names = sorted({k[3:-3] for k in keys if k.startswith("ts_") and k.endswith("_lc")})
    if collections is not None:
        names = [n for n in names if n in collections]
    want = _BRANCHES + [f"ts_{n}_{s}" for n in names for s in ("nLC", "lc", "mult")]
    if "clue_assignment" in keys:
        want.append("clue_assignment")
    stop = None if not max_events else entry_start + max_events
    a = t.arrays(want, library="np", entry_start=entry_start, entry_stop=stop)

    def col(name, i):
        return np.asarray(a[name][i], dtype=np.int64 if name in _INT else None)

    events = []
    for i in range(len(a["event"])):
        cmssw = {}
        for n in names:
            nlc = col(f"ts_{n}_nLC", i)
            mult = np.asarray(a[f"ts_{n}_mult"][i], np.float64)
            cmssw[n] = Tracksters(np.concatenate([[0], np.cumsum(nlc)]).astype(np.int64),
                                  np.asarray(a[f"ts_{n}_lc"][i], np.int64),
                                  np.where(mult > 0, 1.0 / np.maximum(mult, 1e-9), 1.0))
        events.append(Event(
            int(a["event"][i]),
            lc={f: col(f"lc_{f}", i) for f in LC_FIELDS},
            rh={f: col(f"rh_{f}", i) for f in RH_FIELDS},
            lch={"lc": col("lch_lc", i), "rh": col("lch_rh", i), "frac": np.asarray(a["lch_frac"][i], np.float64)},
            at={f: col(f"at_{f}", i) for f in AT_FIELDS},
            un={f: col(f"un_{f}", i) for f in UN_FIELDS},
            tra={"rh": col("tra_rh", i), "at": col("tra_at", i), "E": np.asarray(a["tra_E"][i], np.float64)},
            clue_assignment=np.asarray(a["clue_assignment"][i]) if "clue_assignment" in a else None,
            cmssw=cmssw))
    return events
