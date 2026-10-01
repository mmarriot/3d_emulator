"""Emulator vs CMSSW. Cluster indices are not comparable (CMSSW numbers its seeds in the order the
device kernel found them), so partitions are compared: a cluster agrees if exactly the same layer
clusters form it on both sides."""
import numpy as np

from . import clue


def _canonical(labels):
    """Map every label to the smallest layer-cluster index of its cluster (-1 stays -1)."""
    labels = np.asarray(labels)
    out = np.full(len(labels), -1, np.int64)
    sel = labels >= 0
    if sel.any():
        idx = np.nonzero(sel)[0]
        uniq, inv = np.unique(labels[sel], return_inverse=True)
        first = np.full(len(uniq), np.iinfo(np.int64).max)
        np.minimum.at(first, inv, idx)
        out[idx] = first[inv]
    return out


def compare(a, b, consider=None):
    """Partition agreement of two label arrays over the layer clusters `consider` (default: all)."""
    a, b = np.asarray(a), np.asarray(b)
    if consider is not None:
        a, b = np.where(consider, a, -1), np.where(consider, b, -1)
    ca, cb = _canonical(a), _canonical(b)
    n = len(a) if consider is None else int(np.sum(consider))
    # a cluster of `a` is identical in `b` if all its members carry the same canonical id in b and
    # b's cluster has the same size
    size_a = np.bincount(ca[ca >= 0], minlength=len(a))
    size_b = np.bincount(cb[cb >= 0], minlength=len(b))
    in_a = ca >= 0
    member_ok = np.ones(len(a), bool)
    member_ok[in_a] = cb[in_a] == ca[in_a]
    bad = np.zeros(len(a), bool)
    np.logical_or.at(bad, ca[in_a], ~member_ok[in_a])
    ident = np.zeros(len(a), bool)
    roots = np.unique(ca[in_a])
    ident[roots] = ~bad[roots] & (size_a[roots] == size_b[roots])
    lc_agree = np.where(in_a, ident[np.maximum(ca, 0)], cb < 0)
    if consider is not None:
        lc_agree = lc_agree[consider]
    return dict(n_lc=n, lc_agree=int(lc_agree.sum()), lc_agree_frac=float(lc_agree.mean()) if n else float("nan"),
                clusters_a=int(len(roots)), clusters_b=int(len(np.unique(cb[cb >= 0]))),
                identical_clusters=int(ident[roots].sum()),
                outliers_a=int(((a < 0) & (b >= 0)).sum()), outliers_b=int(((b < 0) & (a >= 0)).sum()))


def run(events, params=clue.DEFAULT, collection=None):
    """Closure over events: the raw assignment (before the minimum-size cut) against
    clue_assignment, and, if `collection` is given, the final tracksters against it."""
    raw, final = [], []
    for ev in events:
        elig = ev.eligible()
        if ev.clue_assignment is not None:
            raw.append(compare(clue.cluster(ev, params, drop_small=False), ev.clue_assignment, elig))
        if collection and collection in ev.cmssw:
            ts = ev.cmssw[collection]
            ref = np.full(ev.n_lc, -1, np.int64)
            ref[ts.lc] = ts.owner()
            final.append(compare(clue.cluster(ev, params, drop_small=True), ref, elig))

    def agg(rows):
        if not rows:
            return None
        tot = {k: sum(r[k] for r in rows) for k in rows[0] if k != "lc_agree_frac"}
        tot["lc_agree_frac"] = tot["lc_agree"] / tot["n_lc"] if tot["n_lc"] else float("nan")
        tot["events_identical"] = sum(r["lc_agree"] == r["n_lc"] for r in rows)
        tot["events"] = len(rows)
        return tot

    return dict(raw=agg(raw), final=agg(final), per_event_raw=raw, per_event_final=final)
