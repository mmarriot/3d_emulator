# ticltune

Truth-based metrics and a CLUEstering emulator for tuning TICL **trackster building**, entirely
outside CMSSW. CMSSW is used once per sample, to write the ntuple (`TruthMetrics/Ntuple`).

```
src/ticltune/
  data.py      events from the ntuple: layer clusters, truth table, base particles, CMSSW reference
  clue.py      CLUEstering trackster building, emulated -> one label per layer cluster
  truth.py     base particles -> inseparability M -> targets(tau)
  metrics.py   C, P, F and every diagnostic, as additive per-event sums
  closure.py   emulator vs CMSSW, compared as partitions
  cli.py       `python -m ticltune score|closure ...`
tests/         one toy event per property of the definitions (pytest)
```

Setup: `cmsenv` in the CMSSW area (numpy, scipy, awkward, uproot, pytest are there), then
`pip install -e .` or `PYTHONPATH=src`. Run the tests with `python3 -m pytest -q`.

## Truth

* **Base particles**: the `caloBoundary` particles (crossed into the calorimeter) of the signal and
  of every in-time pileup interaction. Each calorimeter sim hit belongs to its particle's nearest
  `caloBoundary` ancestor (fallbacks: `reconstructableFinalState`, then the root; `bp_kind`).
* **Energy of a base particle in a layer cluster**, rechit weighted:
  `s(l,b) = sum_c fraction_l(c) E_rechit(c) E_sim(b,c) / E_sim(in-time, c)`.
  Rechit energy in cells with no in-time sim energy (out-of-time pileup, noise) is the layer
  cluster's **no-truth** energy.
* **Reachable energy**: only layer clusters the step may use (HGCAL, iteration mask) count. Energy in
  masked layer clusters is reported as unreachable.
* **Inseparability** `M(a,b) = sum_l min(s(l,a), s(l,b)) / min(E_a, E_b)`; `M = 1` means the smaller
  particle sits entirely in layer clusters shared with the other. Pairs with `M >= tau` are linked,
  connected components are the **targets**. One tau for every pair (signal or pileup, any origin).
  Default `tau = 0.9` (tight); to be set from the measured M distribution.
* Targets have no energy threshold. A target is **selected** if its reachable energy
  `E_t >= select_energy` (default 5 GeV), signal or pileup alike. The cut is on the target, after
  linking, so a soft particle inseparable from a hard one is part of it (not penalised). Unselected
  (soft) targets are noise: they never enter C or F and only lower the purity of tracksters they join.
  `E_t` is reachable energy, so the selected set depends on the iteration mask.
* A target is a **signal target** if it holds a signal particle; the signal-only objectives
  (`sig_C`, `sig_P`, `sig_F`) are kept as diagnostics.

## Metrics

`e(t,k)` = energy of target t in trackster k; a trackster's energy is `E_k = sum_t e(t,k) + no-truth`.

| | definition |
|---|---|
| **C** completeness | `sum_{sel t} max_k e(t,k) / sum_{sel t} E_t` |
| **P** purity | `sum_{sel k} e(main(k),k) / sum_{sel k} E_k`; main(k) = largest target in k (any), selected trackster = main target is selected. Other selected targets, soft targets and no-truth energy lower it (`P + selts_other_sel_frac + selts_soft_frac + selts_notruth_frac = 1`); tracksters led by a soft target are not scored |
| **F** fragmentation | energy-weighted mean of `1 / sum_k f_tk^2` over selected targets, `f_tk` = share of the target's clustered energy in trackster k |

Diagnostics (all in `metrics.summary`): selected share of the target energy, unclustered /
unreachable fractions, pileup inside selected and signal targets, the signal-only C/P/F with the
no-truth / pileup / other-signal fractions of signal tracksters, individual / split / lost
outcomes (`> 0.5` of both the target and the trackster = individual), merge rate (>= 2 targets each
>= 10% of the trackster), fake rate (no-truth >= 50%), N_mix, and the same for all targets including
pileup. Counting metrics can skip targets below `min_target_energy`.

## Emulator

`clue.cluster(event, Params(...))` reproduces `TrackstersCLUEsteringProducer` +
`PatternRecognitionbyCLUEstering` (defaults = `CLUE3DHighStep_cff` under `ticl_dev`); float32 on by
default; tie-breaks by the layer cluster's seed DetId as CLUEstering does. The research variants of
the original `clue_emu.py` are kept as `Params` fields, all off by default.
Validate with `python -m ticltune closure ntuple.root` before trusting a setting.
