"""Events as read from the trackster-building truth ntuple (TruthMetrics/Ntuple in CMSSW).

One `Event` holds plain numpy arrays: the layer clusters, the sparse truth table
(layer cluster, base particle, energy), the base particles and, for closure tests, what CMSSW
produced. Nothing here depends on clustering parameters, so an event is loaded once and scored
many times.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

# DetId::Detector values of the HGCAL layer clusters, in the order of the CLUEstering sigmaT list.
HGCAL_DETECTORS = (8, 9, 10)  # HGCalEE, HGCalHSi, HGCalHSc
SIGMA_INDEX = {8: 0, 9: 1, 10: 2}

LC_FIELDS = ("E", "x", "y", "z", "layer", "side", "det", "nhits", "algo", "seed", "mask", "noTruthE", "recE")
BP_FIELDS = ("id", "pdg", "signal", "bx", "evt", "kind", "origin", "originPdg", "originE", "E", "eta", "phi", "simE")
RH_FIELDS = ("E", "x", "y", "z", "layer", "det", "lc")  # rechits in layer clusters (newer ntuples only)


@dataclass
class Tracksters:
    """A trackster collection as layer-cluster lists. `weight` is the energy fraction of the layer
    cluster each trackster takes (1 / vertex multiplicity in TICL, 1 for CLUEstering)."""
    energy: np.ndarray
    offsets: np.ndarray  # trackster k owns lc[offsets[k]:offsets[k+1]]
    lc: np.ndarray
    weight: np.ndarray

    @property
    def n(self):
        return len(self.offsets) - 1

    @classmethod
    def from_labels(cls, labels, lc_energy=None):
        """Tracksters from a per-layer-cluster label array (-1 = not in a trackster)."""
        labels = np.asarray(labels)
        sel = np.nonzero(labels >= 0)[0]
        if len(sel) == 0:
            return cls(np.zeros(0), np.zeros(1, np.int64), np.zeros(0, np.int64), np.zeros(0))
        uniq, inv = np.unique(labels[sel], return_inverse=True)
        order = np.argsort(inv, kind="stable")
        lc = sel[order]
        counts = np.bincount(inv, minlength=len(uniq))
        offsets = np.concatenate([[0], np.cumsum(counts)])
        energy = (np.bincount(inv, weights=lc_energy[sel], minlength=len(uniq))
                  if lc_energy is not None else np.zeros(len(uniq)))
        return cls(energy, offsets, lc.astype(np.int64), np.ones(len(lc)))

    def owner(self):
        """Trackster index of every entry of `lc`."""
        return np.repeat(np.arange(self.n), np.diff(self.offsets))


@dataclass
class Event:
    run: int
    lumi: int
    event: int
    lc: Dict[str, np.ndarray]           # LC_FIELDS
    tr_lc: np.ndarray                    # truth table: layer cluster index
    tr_bp: np.ndarray                    #              base-particle row
    tr_E: np.ndarray                     #              energy [GeV]
    bp: Dict[str, np.ndarray]            # BP_FIELDS
    clue_assignment: Optional[np.ndarray] = None
    cmssw: Dict[str, Tracksters] = field(default_factory=dict)
    rh: Optional[Dict[str, np.ndarray]] = None  # RH_FIELDS; lc = the layer cluster holding most of the rechit
    trh_rh: Optional[np.ndarray] = None         # rechit truth table: rechit index
    trh_bp: Optional[np.ndarray] = None         #                     base-particle row
    trh_E: Optional[np.ndarray] = None          #                     E_rechit * share of the in-time sim energy [GeV]
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def n_lc(self):
        return len(self.lc["E"])

    @property
    def n_bp(self):
        return len(self.bp["id"])

    def eligible(self):
        """Layer clusters the trackster-building step may use: HGCAL, and passing its iteration mask."""
        return (self.lc["mask"] > 0) & np.isin(self.lc["det"], HGCAL_DETECTORS)

    def rh_eligible(self):
        """Rechits of the layer clusters the step may use."""
        lc = self.rh["lc"]
        return (lc >= 0) & self.eligible()[np.maximum(lc, 0)]

    def lc_energy(self):
        """Layer-cluster energy decomposed into truth + no-truth (= sum of fraction * rechit energy),
        so that the shares of a trackster's energy add up to one."""
        return self.lc["recE"]


def load(path, tree="tracksterTruthNtuplizer/events", max_events=0, entry_start=0):
    """Read events from one ntuple file."""
    import uproot

    t = uproot.open(path)[tree]
    stop = None if not max_events else entry_start + max_events
    a = t.arrays(library="np", entry_start=entry_start, entry_stop=stop)
    ts_names = sorted({k[3:-3] for k in a if k.startswith("ts_") and k.endswith("_lc")})
    events = []
    for i in range(len(a["event"])):
        lc = {f: np.asarray(a[f"lc_{f}"][i]) for f in LC_FIELDS}
        bp = {f: np.asarray(a[f"bp_{f}"][i]) for f in BP_FIELDS}
        cmssw = {}
        for nm in ts_names:
            nlc = np.asarray(a[f"ts_{nm}_nLC"][i])
            mult = np.asarray(a[f"ts_{nm}_mult"][i], dtype=np.float64)
            cmssw[nm] = Tracksters(np.asarray(a[f"ts_{nm}_E"][i]), np.concatenate([[0], np.cumsum(nlc)]),
                                   np.asarray(a[f"ts_{nm}_lc"][i], dtype=np.int64),
                                   np.where(mult > 0, 1.0 / np.maximum(mult, 1e-9), 1.0))
        rh = {}
        if "rh_E" in a:
            rh = dict(rh={f: np.asarray(a[f"rh_{f}"][i]) for f in RH_FIELDS},
                      trh_rh=np.asarray(a["trh_rh"][i], dtype=np.int64), trh_bp=np.asarray(a["trh_bp"][i], dtype=np.int64),
                      trh_E=np.asarray(a["trh_E"][i], dtype=np.float64))
        events.append(Event(int(a["run"][i]), int(a["lumi"][i]), int(a["event"][i]), lc,
                            np.asarray(a["tr_lc"][i], dtype=np.int64), np.asarray(a["tr_bp"][i], dtype=np.int64),
                            np.asarray(a["tr_E"][i], dtype=np.float64), bp,
                            np.asarray(a["clue_assignment"][i]) if "clue_assignment" in a else None, cmssw, **rh))
    return events
