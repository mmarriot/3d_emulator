# ticltune

Truth-based metrics and a CLUEstering emulator for tuning TICL **trackster building**, entirely
outside CMSSW. CMSSW is used once per sample, to write the ntuple (`TruthMetrics/Ntuple`).

```
src/ticltune/
  data.py      events from the ntuple: layer clusters, truth table, base particles, CMSSW reference
  clue.py      CLUEstering trackster building, emulated -> one label per layer cluster
  truth.py     base particles -> targets: what an ideal clustering could reconstruct on its own
  metrics.py   C, P, F and every diagnostic, as additive per-event sums
  closure.py   emulator vs CMSSW, compared as partitions
  truth_display.py  per-event JSON for the truth viewer (TruthMetrics/Display/viewer/truth_targets.html), no reco
  cli.py       `python -m ticltune score|closure|truth-display ...`
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
* **Rechit level**: the same table per rechit (rechits of layer clusters), `E_rechit * E_sim(b, c) / E_sim(in-time, c)`.
* **Reachable energy**: only layer clusters the step may use (HGCAL, iteration mask) count, and at rechit level the
  rechits of those layer clusters. Energy elsewhere is reported as unreachable.
* **Targets: the ideal-clustering test** (`truth.build(ev, level="lc"|"rh", frac=0.5)`). The ideal clustering gives
  every cell (layer cluster or rechit), whole, to the object with the most energy in it. An object passes if its ideal
  cluster holds more than `frac` of its energy (completeness) and more than `frac` of the truth energy in that cluster
  is its own (purity): the metrics' "individual" outcome, for the best clustering any algorithm could make of these
  cells. An object that fails cannot be reconstructed on its own by any algorithm working on these cells, so it is
  merged with the object it is most mixed with (its energy in the other's ideal cluster plus the other's in its own),
  and the test is repeated on the merged objects until every object passes. No geometry, no overlap threshold.
  On layer clusters the targets are what trackster building can be asked for; on rechits, what the whole chain could
  separate. A pair separable on rechits but not on layer clusters is merged by the layer clustering.
* Targets have no energy threshold. A target is **selected** if its reachable energy
  `E_t >= select_energy` (default 5 GeV), signal or pileup alike. The cut is on the target, after
  linking, so a soft particle inseparable from a hard one is part of it (not penalised). Unselected
  (soft) targets are noise: they never enter C or F and only lower the purity of tracksters they join.
  `E_t` is reachable energy, so the selected set depends on the iteration mask.
* A target is a **signal target** if it holds a signal particle; the signal-only objectives
  (`sig_C`, `sig_P`, `sig_F`) are kept as diagnostics.

## Metrics

`e(t,k)` = energy of target t in trackster k; a trackster's energy is `E_k = sum_t e(t,k) + no-truth`.

**Tuning objectives: signal truth groups against the ideal clustering** (`metrics.ideal_scores`, layer clusters).

* Signal targets: more than `signal_frac` (50%) of `E_t` from signal particles. The targets come from the test on the
  whole event, pileup included: pileup that cannot be separated from the signal is in the signal target (no cost),
  separable pileup is an object of its own.
* Truth groups follow the clustering: a signal target and a trackster are linked if the trackster is a piece of the
  target (it is the trackster's main target) or claims it (holds more than `claim_frac` = 50% of it). Connected sets are
  the groups; a group's truth is the sum of its targets. A trackster only joins targets that overlap: of two
  targets, more than `overlap_frac` (50%) of the smaller one's energy lies in layer clusters shared with the other,
  directly or through other targets it joins (no distance scale: a brem photon showering along its electron overlaps
  it, two showers grazing each other do not). Merging whole overlapping targets costs nothing, taking part of a
  target or a target that does not overlap makes it contamination, splitting a group into several tracksters is allowed (the linking joins them), and
  pileup never joins a group. Scored: groups holding a signal target with `E_t >= objective_energy` (2 GeV; the
  `sel_*` diagnostics keep `select_energy` = 5 GeV).
* Ideal clustering with every group summed: every layer cluster, whole, to its largest object; `K_g` = the layer
  clusters of group g. Its other content (other objects, no-truth energy) is contamination no clustering of layer
  clusters can avoid.

| | definition |
|---|---|
| **C** | `sum_g sum_{k in g} sum_{l in K_g} w_kl T_g(l) / sum_g sum_{l in K_g} T_g(l)`: what the group's tracksters collect of what an ideal clustering could |
| **P** | `1 - sum_k A_k / sum_k E_k` over the tracksters of scored groups, `A_k = sum_{l not in K_g} w_kl (E_l - T_g(l))`: what is not the group's, in layer clusters the ideal clustering gives to something else |
| **F** | reported, not tuned: energy-weighted mean over scored groups of `1 / sum_k f_gk^2` over its tracksters |

The ideal clustering scores C = P = 1, and so does any merge of whole targets. Diagnostics: groups per event, share
merged by the clustering, tracksters per group, groups with none, the ideal clustering's own completeness and purity,
the free contamination in the tracksters, the signal energy in pileup targets and in unscored groups.

The previous objectives (every selected target, signal or pileup, no ideal reference) are kept as `sel_C`, `sel_P`,
`sel_F`, and the same for targets holding a signal particle as `sig_C`, `sig_P`, `sig_F`; all other diagnostics
(`metrics.summary`) are as before: unclustered / unreachable fractions, pileup inside, individual / split / lost
outcomes, merge and fake rates, N_mix.

## Emulator

`clue.cluster(event, Params(...))` reproduces `TrackstersCLUEsteringProducer` +
`PatternRecognitionbyCLUEstering` (defaults = `CLUE3DHighStep_cff` under `ticl_dev`); float32 on by
default; tie-breaks by the layer cluster's seed DetId as CLUEstering does. The research variants of
the original `clue_emu.py` are kept as `Params` fields, all off by default.
Validate with `python -m ticltune closure ntuple.root` before trusting a setting.
