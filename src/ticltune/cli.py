"""Command line:
    ticltune score   NTUPLE... [--settings s.json] [--level lc|rh] [--frac 0.5] [--cmssw ticlTrackstersCLUE3DHigh]
    ticltune closure NTUPLE... [--collection ticlTrackstersCLUE3DHigh]
    ticltune truth-display NTUPLE... -o VIEWER/truth_data [--frac 0.5]   (truthgraph_<event>.json next to each NTUPLE)
"""
import argparse
import json
import time

from . import clue, closure, data, metrics, truth, truth_display
from .data import Tracksters


def _load(files, nev):
    evs = []
    for f in files:
        evs += data.load(f, max_events=max(0, nev - len(evs)) if nev else 0)
        if nev and len(evs) >= nev:
            break
    return evs


def score(events, settings, level="lc", frac=truth.DEFAULT_FRAC, cfg=metrics.MetricConfig(), cmssw=None):
    """Summary per setting (dict name -> summary). cmssw: also score that stored collection."""
    out = {}
    tg = [truth.build(ev, level, frac) for ev in events]
    for st in settings:
        name = st.get("name", "setting")
        p = clue.DEFAULT.with_(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in st.items() if k != "name"})
        sums = [metrics.evaluate(ev, Tracksters.from_labels(clue.cluster(ev, p), ev.lc_energy()), t, cfg)
                for ev, t in zip(events, tg)]
        out[name] = metrics.summary(metrics.combine(sums))
    if cmssw:
        sums = [metrics.evaluate(ev, ev.cmssw[cmssw], t, cfg) for ev, t in zip(events, tg) if cmssw in ev.cmssw]
        out[f"CMSSW:{cmssw}"] = metrics.summary(metrics.combine(sums))
    return out


def _print(rows):
    keys = list(next(iter(rows.values())))
    w = max(len(n) for n in rows) + 2
    print(" " * 30 + "".join(f"{n:>{w}}" for n in rows))
    for k in keys:
        print(f"{k:30}" + "".join(f"{rows[n][k]:>{w}.4f}" for n in rows))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ticltune")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="metrics of emulator settings (and optionally a CMSSW collection)")
    s.add_argument("files", nargs="+")
    s.add_argument("--settings", help='json list of {"name":..., <Params field>: value}; default: CMSSW defaults')
    s.add_argument("--level", choices=("lc", "rh"), default="lc", help="cells of the ideal-clustering test")
    s.add_argument("--frac", type=float, default=truth.DEFAULT_FRAC, help="completeness and purity it must exceed")
    s.add_argument("--cmssw", default=None, help="also score this stored trackster collection")
    s.add_argument("--nev", type=int, default=0)
    s.add_argument("-o", "--out")
    c = sub.add_parser("closure", help="emulator vs CMSSW partitions")
    c.add_argument("files", nargs="+")
    c.add_argument("--collection", default="ticlTrackstersCLUE3DHigh")
    c.add_argument("--nev", type=int, default=0)
    w = sub.add_parser("truth-display", help="export truth graph, particles and targets for the truth viewer")
    w.add_argument("files", nargs="+")
    w.add_argument("-o", "--out", required=True)
    w.add_argument("--frac", type=float, default=truth.DEFAULT_FRAC)
    w.add_argument("--roi", type=float, default=truth_display.DEFAULT_ROI,
                   help="with pileup, write only cells within this dR of a signal particle (0 = all)")
    a = ap.parse_args(argv)
    if a.cmd == "truth-display":
        t0 = time.time()
        index = truth_display.export(a.files, a.out, frac=a.frac, roi=a.roi)
        print(f"{len(index)} events exported to {a.out} in {time.time() - t0:.1f} s")
        return
    t0 = time.time()
    events = _load(a.files, a.nev)
    print(f"{len(events)} events loaded in {time.time() - t0:.1f} s")
    if a.cmd == "score":
        settings = json.load(open(a.settings)) if a.settings else [{"name": "default"}]
        t0 = time.time()
        rows = score(events, settings, a.level, a.frac, cmssw=a.cmssw)
        print(f"scored in {time.time() - t0:.1f} s (targets: ideal-clustering test on {a.level}, frac = {a.frac})")
        _print(rows)
        if a.out:
            json.dump(rows, open(a.out, "w"), indent=1)
    else:
        res = closure.run(events, collection=a.collection)
        for k in ("raw", "final"):
            if res[k]:
                print(k, {kk: (round(v, 5) if isinstance(v, float) else v) for kk, v in res[k].items()})


if __name__ == "__main__":
    main()
