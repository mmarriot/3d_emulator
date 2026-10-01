"""ticltune: truth-based metrics and a CLUEstering emulator for tuning TICL trackster building.

    data     events from the CMSSW truth ntuple (TruthMetrics/Ntuple)
    clue     the CLUEstering trackster-building emulation -> one label per layer cluster
    truth    base particles -> inseparability M -> targets(tau)
    metrics  completeness C, purity P, fragmentation F and every diagnostic, as additive sums
    closure  emulator vs CMSSW, compared as partitions
"""
from . import clue, closure, data, metrics, truth  # noqa: F401
from .clue import DEFAULT, Params  # noqa: F401
from .data import Event, Tracksters, load  # noqa: F401

__version__ = "0.1.0"
