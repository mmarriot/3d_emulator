"""ticltune: truth, metrics and a CLUEstering emulator for tuning TICL trackster building.

Specification: tuning/V3_TRUTH_AND_METRICS.md.

    data     events from the CMSSW truth ntuple (TruthMetrics/Ntuple): rechits, layer clusters, atoms, units
    clue     the CLUEstering trackster-building emulation, on layer clusters or on rechits
    truth    atoms -> units -> targets (ideal-clustering test on rechits, never across interactions)
    metrics  K_sig (contamination), eps_sig (efficiency), Phi_sig (surplus fragments) and diagnostics
    closure  emulator vs CMSSW, compared as partitions
"""
from . import clue, closure, data, metrics, truth  # noqa: F401
from .clue import DEFAULT, Params  # noqa: F401
from .data import Event, Tracksters, load  # noqa: F401

__version__ = "1.0.0"
