# ticltune

Truth, metrics and a CLUEstering emulator for tuning TICL **trackster building** outside CMSSW, on layer clusters or
on rechits, scored identically at both levels. The definitions (and the reason for every choice) are in
`../tuning/V3_TRUTH_AND_METRICS.md`; this package implements exactly that. CMSSW is used once per sample, to write the
ntuple (`TruthMetrics/Ntuple`, `TracksterTruthNtuplizer`).

```
src/ticltune/
  data.py     events from the ntuple: every HGCAL rechit, layer clusters (+ rechit fractions), atoms, units, the
              (rechit, atom, energy) truth table, CMSSW tracksters and CLUEstering assignment
  clue.py     CLUEstering trackster building (the CMSSW algorithm), points = layer clusters or rechits
  truth.py    atoms -> units -> targets: ideal-clustering test on rechits, never across interactions; ideal
              clusterings at both levels; natural pieces
  metrics.py  any clustering -> rechit fractions -> best target per object -> K_sig, eps_sig, Phi_sig (+ diagnostics),
              as additive per-event sums
  closure.py  emulator vs CMSSW, compared as partitions
  cli.py      python -m ticltune score | truth | closure
tests/        toy events, one per property of the definitions; test_acceptance.py on a real ntuple
```

Setup: `cmsenv` in the CMSSW area (numpy, scipy, uproot, pytest are there), then `pip install -e .` or
`PYTHONPATH=src`. Tests: `python3 -m pytest -q`; on a real ntuple:
`TICLTUNE_NTUPLE=/path/job_000.root python3 -m pytest -q tests/test_acceptance.py`.

## Truth (spec section 4)

* **Atoms**: the truth-graph particles with energy in HGCAL rechits, inside the calorimeter included. Energy of atom a
  in rechit c: `s(c, a) = E_rechit(c) * E_sim(a, c) / E_sim(all in-time, c)`; rechit energy with no in-time sim energy
  is no-truth energy (out-of-time pileup, noise).
* **Units** (made by the ntuplizer): an atom's nearest `reconstructableFinalState` ancestor-or-self; below a pi0, the
  pi0's decay daughter; else the root. The coarsest object downstream wants as one object.
* **Targets** (`truth.build`): units merged by the ideal-clustering test on all HGCAL rechits (every rechit to the
  object with the most energy in it; pass = completeness > 0.5 and purity > 0.5). A failing object merges with the
  object of the same interaction it is most confused with; with none, it stays alone, flagged `unreachable`. Signal =
  hard-scatter interaction; class EM if the leading unit is e, gamma or a pi0 daughter, else HAD.
* **References** (`truth.ideal_labels`): the ideal clustering on rechits and on layer clusters.
* **Natural pieces** (diagnostic): the same test on a target's atoms, a failing piece joining its parent atom's piece.

## Clustering (`clue.cluster(clue.points(ev, level), params)`)

The CMSSW CLUEstering trackster building (`clue.Params`, defaults = `CLUE3DHighStep_cff`). Points: `level="lc"` the
layer clusters passing the CLUE3DHigh mask (>= 2 hits in silicon); `level="lc_all"` every layer cluster, single-hit ones
included (the same filter with `min_cluster_size = 1`); `level="rh"` every HGCAL rechit (tag = DetId). Emulator vs CMSSW on 30 PU200 events of the tuning
sample: 99.995% of the layer clusters assigned identically, 29/30 events identical (the rest: 1-ulp density ties).

## Metrics (spec section 6)

Any clustering becomes rechit fractions `w_rc` (`metrics.objects`: labels at either level, or
`objects_from_tracksters` for CMSSW collections, through the exact layer-cluster fractions). Each object goes to the
target with the largest share (ties: larger deposited energy, then lower index); per class (EM, HAD), then averaged:

| metric | definition | best |
|---|---|---|
| `K_sig` | energy in signal objects that is not their target / their energy | 0 |
| `eps_sig` | each signal target's energy in the objects assigned to it / its deposited energy (all rechits) | 1 |
| `Phi_sig` | `1 - sum c_t / sum c_t N_t` (N_t = objects assigned to t, c_t = their energy of t): surplus fragments | 0 |

`metrics.evaluate` returns additive sums (any split of the events gives the same `summary`); `metrics.summary` gives
the metrics and the diagnostics (`K_pu`, `eps_pu`, cell floor of K, lost energy split, objects per event, every metric
per class and energy bin); `metrics.mean_over_samples` averages samples with equal weight.

## Command line

```bash
python3 -m ticltune score  FILES... [--level lc|lc_all|rh] [--settings s.json] [--cmssw ticlTrackstersCLUE3DHigh] [-o out.json]
python3 -m ticltune truth  FILES...
python3 -m ticltune closure FILES... [--collection ticlTrackstersCLUE3DHigh]
```
