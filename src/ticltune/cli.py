"""Command line:
    ticltune score   NTUPLE... [--level lc|lc_all|rh] [--settings s.json] [--cmssw COLLECTION] [--nev N] [-o out.json]
    ticltune truth   NTUPLE... [--nev N]                  targets and the ideal references at both levels
    ticltune closure NTUPLE... [--collection ticlTrackstersCLUE3DHigh] [--nev N]
All files of one call are one sample; `metrics.mean_over_samples` combines samples.
"""
import argparse
import json
import time

import numpy as np

from . import clue, closure, data, metrics, truth


def _load(files, nev):
    evs = []
    for f in files:
        evs += data.load(f, max_events=max(0, nev - len(evs)) if nev else 0)
        if nev and len(evs) >= nev:
            break
    return evs


def params_of(setting):
    kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in setting.items() if k != "name"}
    return clue.DEFAULT.with_(**kw)


def score(events, settings, level="lc", cmssw=None):
    """Summary per setting (name -> summary): emulated settings at `level`, the ideal clustering at both levels and,
    optionally, a stored CMSSW collection."""
    out = {}
    for st in settings:
        p = params_of(st)
        out[st.get("name", "setting")] = metrics.summary(metrics.combine(
            metrics.score(ev, clue.cluster(clue.points(ev, level), p), level) for ev in events))
    for lv in ("rh", "lc"):
        out[f"ideal_{lv}"] = metrics.summary(metrics.combine(
            metrics.score(ev, truth.ideal_labels(ev, truth.build(ev), lv), lv) for ev in events))
    if cmssw:
        out[f"CMSSW:{cmssw}"] = metrics.summary(metrics.combine(
            metrics.evaluate(ev, metrics.objects_from_tracksters(ev, ev.cmssw[cmssw]), truth.build(ev))
            for ev in events if cmssw in ev.cmssw))
    return out


def truth_summary(events):
    """Targets per event and how they were made, and the natural fragmentation."""
    rows = []
    for ev in events:
        t = truth.build(ev)
        rows.append(dict(units=len(t.unit_target), targets=t.n, signal_targets=int(t.signal.sum()),
                         merged_units=int(np.sum(t.merged_at >= 0)), unreachable=int(t.unreachable.sum()),
                         signal_unreachable=int((t.unreachable & t.signal).sum()), iterations=t.n_iterations,
                         E_signal=float(t.E[t.signal].sum()), E_pileup=float(t.E[~t.signal].sum()),
                         E_no_truth=float(t.no_truth.sum()),
                         phi_nat_num=float(np.sum(t.E[t.signal])),
                         phi_nat_den=float(np.sum(t.E[t.signal] * t.n_pieces[t.signal]))))
    tot = {k: sum(r[k] for r in rows) for k in rows[0]}
    n = len(rows)
    out = {k: tot[k] / n for k in tot if not k.startswith("phi_nat")}
    out["Phi_nat_signal"] = 1 - tot["phi_nat_num"] / tot["phi_nat_den"] if tot["phi_nat_den"] else 0.0
    out["events"] = n
    return out


def _print(rows):
    keys = [k for k in next(iter(rows.values())) if "_E" not in k]  # energy-binned diagnostics: see -o
    w = max(max(len(n) for n in rows) + 2, 12)
    print(" " * 28 + "".join(f"{n:>{w}}" for n in rows))
    for k in keys:
        print(f"{k:28}" + "".join(f"{rows[n][k]:>{w}.4f}" for n in rows))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ticltune")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="metrics of emulator settings, the ideal references and a CMSSW collection")
    s.add_argument("files", nargs="+")
    s.add_argument("--level", choices=clue.LEVELS, default="lc", help="points of the emulated clustering")
    s.add_argument("--settings", help='json list of {"name":..., <Params field>: value}; default: CMSSW defaults')
    s.add_argument("--cmssw", default=None, help="also score this stored trackster collection")
    s.add_argument("--nev", type=int, default=0)
    s.add_argument("-o", "--out")
    t = sub.add_parser("truth", help="targets and natural fragmentation")
    t.add_argument("files", nargs="+")
    t.add_argument("--nev", type=int, default=0)
    c = sub.add_parser("closure", help="emulator vs CMSSW partitions (layer clusters)")
    c.add_argument("files", nargs="+")
    c.add_argument("--collection", default="ticlTrackstersCLUE3DHigh")
    c.add_argument("--nev", type=int, default=0)
    a = ap.parse_args(argv)
    t0 = time.time()
    events = _load(a.files, a.nev)
    print(f"{len(events)} events loaded in {time.time() - t0:.1f} s")
    t0 = time.time()
    if a.cmd == "score":
        settings = json.load(open(a.settings)) if a.settings else [{"name": "default"}]
        rows = score(events, settings, a.level, cmssw=a.cmssw)
        print(f"scored in {time.time() - t0:.1f} s (emulated clustering on {a.level})")
        _print(rows)
        if a.out:
            json.dump(rows, open(a.out, "w"), indent=1)
    elif a.cmd == "truth":
        for k, v in truth_summary(events).items():
            print(f"{k:24} {v:.4f}" if isinstance(v, float) else f"{k:24} {v}")
        print(f"({time.time() - t0:.1f} s)")
    else:
        res = closure.run(events, collection=a.collection)
        for k in ("raw", "final"):
            if res[k]:
                print(k, {kk: (round(v, 5) if isinstance(v, float) else v) for kk, v in res[k].items()})


if __name__ == "__main__":
    main()
