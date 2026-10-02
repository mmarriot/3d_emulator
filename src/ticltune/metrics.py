"""Metrics of trackster building against the truth targets of `truth.build`.

Shared energy of target t and trackster k:   e(t,k) = sum_l w_kl T(l,t)
(w_kl: the fraction of layer cluster l given to trackster k; T: the targets' energies.)
A trackster's energy decomposes exactly as  E_k = sum_t e(t,k) + n_k,  n_k = its no-truth energy
(out-of-time pileup and noise: rechit energy in cells with no in-time sim energy).

Tuning objectives: the signal, measured against the ideal clustering, with truth groups that follow the
clustering (see `ideal_scores` and `summary`).
  Signal targets: targets of which more than signal_frac (0.5) of the energy is signal. They come from the
  ideal-clustering test on the whole event, pileup included, so pileup that cannot be separated from the
  signal is inside a signal target (no cost) and separable pileup is an object of its own.
  Truth groups: a signal target and a trackster are linked if the trackster is a piece of the target (the
  target is its main target, largest e(t,k)) or claims it (holds more than claim_frac = 0.5 of E_t), and a
  trackster only joins targets that overlap: of two targets, more than overlap_frac = 0.5 of the smaller one's
  energy lies in layer clusters it shares with the other, directly or through other targets it joins (no
  distance scale; e.g. a brem photon showering along its electron, not two showers that graze each other).
  The connected sets are the groups; a group's truth is the sum of its targets. So a trackster that merges
  overlapping targets accurately (holding most of each) makes them one group and is not charged for it; one
  that takes only part of a target, or a target that does not overlap the rest, does not claim it, and that is
  contamination. Several pieces of one group are
  allowed (splitting; the trackster linking joins them later). Pileup and other non-signal targets never
  join a group. A group is scored if it holds a signal target with E_t >= objective_energy (2 GeV).
  Ideal clustering of the layer clusters, with every group summed into one object: each layer cluster, whole,
  to its largest object; K_g = the layer clusters of group g. What g could ideally collect is its energy in
  K_g; everything else in K_g (other objects, no-truth energy) is contamination no clustering of layer
  clusters can avoid.
  C  completeness = sum_g sum_{k in g} sum_{l in K_g} w_kl T_g(l) / sum_g sum_{l in K_g} T_g(l)
  P  purity       = 1 - sum_k A_k / sum_k E_k over the tracksters of scored groups, with the avoidable
                    contamination A_k = sum_{l not in K_g} w_kl (E_l - T_g(l)): everything that is not the
                    group's, in layer clusters the ideal clustering gives to something else (another
                    object, or nothing: no-truth layer clusters). Contamination inside K_g is free.
  F  fragmentation (reported, not tuned): energy-weighted mean over scored groups of 1 / sum_k f_gk^2 over
                    its tracksters, f_gk = the share of the collected energy in trackster k.
  The ideal clustering of the targets themselves scores C = P = 1, and so does any merge of whole targets.

The previous objectives, on every selected target (signal or pileup alike, no ideal reference), are kept
as diagnostics:
  sel_C = sum_{t sel} max_k e(t,k) / sum_{t sel} E_t,   sel_P = sum_{k sel} e(main(k), k) / sum_{k sel} E_k,
  sel_F = energy-weighted mean of N_frag(t) = 1 / sum_k f_tk^2, f_tk = e(t,k) / sum_k e(t,k),
  where a selected target has E_t >= select_energy and a selected trackster's main target is selected.
The same restricted to targets holding any signal particle: sig_C, sig_P, sig_F.

All per-event results are additive sums, so events can be scored anywhere and `combine`d.
"""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from .data import Tracksters


@dataclass(frozen=True)
class MetricConfig:
    individual_frac: float = 0.5   # outcome "individual": e > this * E_t and e > this * E_k (strict, as TICL)
    split_frac: float = 0.5        # outcome "split": the target's own fragments hold >= this * E_t
    merge_frac: float = 0.1        # a trackster is a merge if >= 2 targets each give >= this * E_k
    fake_frac: float = 0.5         # a trackster is fake if its no-truth energy >= this * E_k
    min_target_energy: float = 0.0  # targets below this [GeV] are left out of the COUNTING metrics
    select_energy: float = 5.0     # sel_* diagnostics: targets with reachable energy >= this [GeV], signal or pileup
    objective_energy: float = 2.0  # objectives: a group is scored if it holds a signal target with E_t >= this [GeV]
    signal_frac: float = 0.5       # signal targets: more than this share of their energy is signal
    claim_frac: float = 0.5        # a trackster claims a signal target if it holds more than this share of it
    overlap_frac: float = 0.5      # two targets overlap if more than this share of the smaller lies in shared layer clusters


# per family of targets (prefix "sel" = selected, "sig" = signal) and of tracksters ("selts", "sigts")
TARGET_KEYS = ("n", "E", "best", "clustered", "unclustered", "unreachable", "frag_num", "frag_den",
               "pileup_inside", "individual", "split", "lost", "n_counted", "n_multi_interaction",
               "n_multi_origin")
TS_KEYS = ("n", "E", "main", "notruth", "merge", "mix_num")

SUM_KEYS = (
    ("n_events",)
    + tuple(f"{p}_{k}" for p in ("sel", "sig") for k in TARGET_KEYS)
    + tuple(f"{p}_{k}" for p in ("selts", "sigts") for k in TS_KEYS)
    # what else is in the tracksters of each family
    + ("selts_other_sel", "selts_soft", "sigts_pileup", "sigts_other_signal")
    # all targets / all tracksters
    + ("all_n", "all_E", "all_best", "all_individual", "all_split", "all_lost", "all_n_counted",
       "ts_n", "ts_E", "ts_main", "ts_notruth", "ts_fake", "ts_merge", "ts_n_lc")
    # energy the step can never see
    + ("untargeted_unreachable",)
    # the objectives (signal truth groups against the ideal clustering)
    + ("obj_n", "obj_groups", "obj_merged_groups", "obj_E", "obj_reach", "obj_Kall", "obj_credit", "obj_missed",
       "obj_frag_num", "obj_frag_den", "obj_ts_n", "obj_ts_E", "obj_avoid", "obj_unavoid", "obj_own_outside",
       "signal_E", "signal_in_pileup_targets", "signal_unscored")
)


def ideal_scores(ev, tracksters, targets, cfg, W=None):
    """The objectives' terms (module docstring). Returns the truth groups (group of every target, -1 for
    non-signal targets; group of every trackster, -1 if none), the ideal clustering with groups summed (dom:
    group whose ideal cluster holds each layer cluster, -1 if another object or no truth), per group its
    energy, ideal reach, collected energy, completeness, purity and pieces, and per trackster the energy it
    collects for its group inside K_g (credit), the contamination inside K_g (unavoidable) and outside it
    (avoidable)."""
    nT, nK = targets.n, tracksters.n
    W = _weight_matrix(tracksters, ev.n_lc) if W is None else W
    Elc = ev.lc_energy().astype(float)
    T = targets.T.tocsr()
    E_t = targets.E
    with np.errstate(divide="ignore", invalid="ignore"):
        fsig = np.where(E_t > 0, targets.E_signal / E_t, 0.0)
    signal = fsig > cfg.signal_frac
    objective = signal & (E_t >= cfg.objective_energy)
    e = (W @ T).tocsr() if nT else sp.csr_matrix((nK, 0))
    if nK and nT:
        e_main = e.max(axis=1).toarray().ravel()
        main = np.where(e_main > 0, np.asarray(e.argmax(axis=1)).ravel(), -1)
    else:
        main = np.full(nK, -1)

    # --- truth groups: signal targets linked to the tracksters that are their pieces or claim them ---
    ec = e.tocoo()
    claim = signal[ec.col] & (ec.data > cfg.claim_frac * E_t[ec.col]) if nT else np.zeros(0, bool)
    k_piece = np.nonzero((main >= 0) & (signal[np.maximum(main, 0)] if nT else False))[0]
    a_ = np.r_[ec.col[claim], main[k_piece]].astype(np.int64)          # target nodes 0..nT-1
    b_ = np.r_[ec.row[claim], k_piece].astype(np.int64)                # tracksters
    a_, b_ = _overlapping(T, E_t, e, main, signal, a_, b_, cfg.overlap_frac)
    b_ = b_ + nT                                                         # trackster nodes nT..nT+nK-1
    A = sp.csr_matrix((np.ones(len(a_)), (a_, b_)), shape=(nT + nK, nT + nK))
    _, comp = connected_components(A, directed=False)
    uc = np.unique(comp[:nT][signal])
    g_of = np.full(comp.max() + 1 if len(comp) else 0, -1, np.int64)
    g_of[uc] = np.arange(len(uc))
    nG = len(uc)
    t_group = np.where(signal, g_of[comp[:nT]], -1)
    k_group = g_of[comp[nT:]] if nK else np.zeros(0, np.int64)

    # --- ideal clustering with every group summed into one object (other targets stay themselves) ---
    others = np.nonzero(~signal)[0]
    obj_id = t_group.copy()
    obj_id[others] = nG + np.arange(len(others))
    G = sp.csr_matrix((np.ones(nT), (np.arange(nT), obj_id)), shape=(nT, nG + len(others)))
    Tm = (T @ G).tocsr()
    top = Tm.max(axis=1).toarray().ravel() if nT else np.zeros(ev.n_lc)
    dom = np.where(top > 0, np.asarray(Tm.argmax(axis=1)).ravel(), -1) if nT else np.full(ev.n_lc, -1)
    dom = np.where(dom < nG, dom, -1)                                   # layer cluster -> group, or -1
    has = dom >= 0
    reach = np.bincount(dom[has], weights=top[has], minlength=nG)      # sum_{l in K_g} T_g(l)
    Kall = np.bincount(dom[has], weights=Elc[has], minlength=nG)       # sum_{l in K_g} E_l

    # --- tracksters of each group: collected, free and avoidable contamination ---
    Wc = W.tocoo()
    g = k_group[Wc.row] if nK else np.zeros(0, np.int64)
    ok = g >= 0
    T_lg = np.zeros(len(g))
    if ok.any():
        T_lg[ok] = np.asarray(Tm[Wc.col[ok], g[ok]]).ravel()          # T_g(l) per (k, l)
    inK = ok & (dom[Wc.col] == g)
    other = Elc[Wc.col] - T_lg                                          # what in l is not the group's
    credit = np.bincount(Wc.row, weights=Wc.data * T_lg * inK, minlength=nK)
    unavoid = np.bincount(Wc.row, weights=Wc.data * other * inK, minlength=nK)
    avoid = np.bincount(Wc.row, weights=Wc.data * other * (ok & ~inK), minlength=nK)
    own_outside = np.bincount(Wc.row, weights=Wc.data * T_lg * (ok & ~inK), minlength=nK)
    E_k = W @ Elc
    ing = k_group >= 0
    g_credit = np.bincount(k_group[ing], weights=credit[ing], minlength=nG)
    g_sq = np.bincount(k_group[ing], weights=credit[ing] ** 2, minlength=nG)
    g_avoid = np.bincount(k_group[ing], weights=avoid[ing], minlength=nG)
    g_tsE = np.bincount(k_group[ing], weights=E_k[ing], minlength=nG)
    g_nts = np.bincount(k_group[ing], minlength=nG)
    g_E = np.bincount(t_group[signal], weights=E_t[signal], minlength=nG)
    g_ntargets = np.bincount(t_group[signal], minlength=nG)
    g_scored = np.bincount(t_group[objective], minlength=nG) > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        return dict(signal_frac=fsig, signal=signal, objective=objective, ts_main=main,
                    target_group=t_group, ts_group=k_group, dom=dom,
                    group_scored=g_scored, group_E=g_E, group_n_targets=g_ntargets, reach=reach, Kall=Kall,
                    credit=g_credit, avoid=g_avoid, pieces_E=g_tsE, n_pieces=g_nts,
                    completeness=np.where(reach > 0, g_credit / reach, np.nan),
                    purity=np.where(g_tsE > 0, 1 - g_avoid / g_tsE, np.nan),
                    nfrag=np.where(g_sq > 0, g_credit ** 2 / g_sq, np.nan),
                    ts_E=E_k, ts_credit=credit, ts_avoid=avoid, ts_unavoid=unavoid, ts_own_outside=own_outside)


def _overlapping(T, E_t, e, main, signal, t_, k_, frac):
    """Keep, for every trackster, only the targets it links (t_, k_ pairs) that overlap its anchor, directly or
    through other targets it links. Two targets overlap if more than frac of the smaller one's energy lies in
    layer clusters where the other one also has energy. The anchor is the trackster's main target if that one is
    a signal target, else the signal target it holds most of."""
    if len(t_) == 0:
        return t_, k_
    pairs = np.unique(np.stack([k_, t_]), axis=1)
    k_, t_ = pairs[0], pairs[1]
    first = np.r_[0, np.nonzero(np.diff(k_))[0] + 1]
    multi = first[np.diff(np.r_[first, len(k_)]) > 1]
    if not len(multi):
        return t_, k_
    Tb = (T > 0).astype(np.int8).tocsc()
    keep = np.ones(len(k_), bool)
    for f in multi:
        k = k_[f]
        sel = np.nonzero(k_ == k)[0]
        ts = t_[sel]
        Ts = T[:, ts].tocsc()
        sh = (Ts.T @ Tb[:, ts]).toarray()                                   # sh[i, j]: energy of i where j has energy
        Es = E_t[ts]
        small = Es[:, None] <= Es[None, :]                                  # i is the smaller of the pair (i, j)
        ov = np.where(small, sh > frac * Es[:, None], sh.T > frac * Es[None, :])
        A = ov | ov.T
        np.fill_diagonal(A, False)
        _, cc = connected_components(sp.csr_matrix(A), directed=False)
        m = main[k]
        anchor = np.nonzero(ts == m)[0][0] if (m >= 0 and signal[m] and m in ts) else int(np.argmax(e[k, ts].toarray()))
        keep[sel] = cc == cc[anchor]
    return t_[keep], k_[keep]


def _weight_matrix(tracksters: Tracksters, n_lc):
    owner = tracksters.owner()
    return sp.csr_matrix((tracksters.weight, (owner, tracksters.lc)), shape=(tracksters.n, n_lc))


def evaluate(ev, tracksters: Tracksters, targets, cfg: MetricConfig = MetricConfig(), keep_details=False):
    """Score one event. Returns a dict of additive sums (SUM_KEYS); with keep_details also the
    per-target and per-trackster arrays."""
    s = dict.fromkeys(SUM_KEYS, 0.0)
    s["n_events"] = 1.0
    nT = targets.n
    W = _weight_matrix(tracksters, ev.n_lc)
    Elc = ev.lc_energy().astype(float)
    E_k = W @ Elc
    nt_k = W @ ev.lc["noTruthE"].astype(float)
    e = (W @ targets.T).tocsr() if nT else sp.csr_matrix((tracksters.n, 0))  # (n_ts, n_targets)
    E_t = targets.E
    sig_t = targets.is_signal
    sel_t = E_t >= cfg.select_energy
    counted = E_t >= cfg.min_target_energy

    # ---- targets ----
    ec = e.tocsc()
    clustered = np.asarray(ec.sum(axis=0)).ravel() if nT else np.zeros(0)
    sq = np.asarray(ec.multiply(ec).sum(axis=0)).ravel() if nT else np.zeros(0)
    best = ec.max(axis=0).toarray().ravel() if (nT and tracksters.n) else np.zeros(nT)
    best_k = np.asarray(ec.argmax(axis=0)).ravel() if (nT and tracksters.n) else np.full(nT, -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        nfrag = np.where(sq > 0, clustered ** 2 / sq, np.nan)
    # ---- tracksters ----
    if tracksters.n and nT:
        main = np.asarray(e.argmax(axis=1)).ravel()
        e_main = e.max(axis=1).toarray().ravel()
        has_truth = e_main > 0
        main = np.where(has_truth, main, -1)
        e_sig = np.asarray(e[:, sig_t].sum(axis=1)).ravel() if sig_t.any() else np.zeros(tracksters.n)
        e_pu = np.asarray(e[:, ~sig_t].sum(axis=1)).ravel() if (~sig_t).any() else np.zeros(tracksters.n)
        e_sel = np.asarray(e[:, sel_t].sum(axis=1)).ravel() if sel_t.any() else np.zeros(tracksters.n)
        e_soft = np.asarray(e[:, ~sel_t].sum(axis=1)).ravel() if (~sel_t).any() else np.zeros(tracksters.n)
        sq_k = np.asarray(e.multiply(e).sum(axis=1)).ravel() + nt_k ** 2
        # number of targets giving >= merge_frac of the trackster energy
        er = e.tocoo()
        signif = er.data >= cfg.merge_frac * E_k[er.row]
        n_signif = np.bincount(er.row[signif], minlength=tracksters.n)
    else:
        main = np.full(tracksters.n, -1)
        e_main = np.zeros(tracksters.n)
        has_truth = np.zeros(tracksters.n, bool)
        e_sig = e_pu = e_sel = e_soft = np.zeros(tracksters.n)
        sq_k = nt_k ** 2
        n_signif = np.zeros(tracksters.n, int)
    with np.errstate(divide="ignore", invalid="ignore"):
        nmix = np.where(sq_k > 0, E_k ** 2 / sq_k, np.nan)
    is_sig_ts = has_truth & (sig_t[np.maximum(main, 0)] if nT else False)
    is_sel_ts = has_truth & (sel_t[np.maximum(main, 0)] if nT else False)
    merge = n_signif >= 2
    fake = nt_k >= cfg.fake_frac * E_k

    # ---- outcomes (counting) ----
    individual = np.zeros(nT, bool)
    split = np.zeros(nT, bool)
    if nT and tracksters.n:
        er = e.tocoo()
        ok = (er.data > cfg.individual_frac * E_t[er.col]) & (er.data > cfg.individual_frac * E_k[er.row])
        individual[er.col[ok]] = True
        own = main[er.row] == er.col
        own_E = np.bincount(er.col[own], weights=er.data[own], minlength=nT)
        split = ~individual & (own_E >= cfg.split_frac * E_t)
    lost = ~individual & ~split

    # ---- sums ----
    def add_targets(p, m):
        cm = m & counted
        fr = m & np.isfinite(nfrag)
        s[p + "_n"] = float(m.sum())
        s[p + "_E"] = float(E_t[m].sum())
        s[p + "_best"] = float(best[m].sum())
        s[p + "_clustered"] = float(clustered[m].sum())
        s[p + "_unclustered"] = float((E_t - clustered)[m].sum())
        s[p + "_unreachable"] = float(targets.E_unreachable[m].sum())
        s[p + "_frag_num"] = float((E_t * np.nan_to_num(nfrag))[fr].sum())
        s[p + "_frag_den"] = float(E_t[fr].sum())
        s[p + "_pileup_inside"] = float(targets.E_pileup[m].sum())
        s[p + "_individual"] = float((individual & cm).sum())
        s[p + "_split"] = float((split & cm).sum())
        s[p + "_lost"] = float((lost & cm).sum())
        s[p + "_n_counted"] = float(cm.sum())
        s[p + "_n_multi_interaction"] = float((m & (targets.n_interactions > 1)).sum())
        s[p + "_n_multi_origin"] = float((m & (targets.n_origins > 1)).sum())

    def add_tracksters(p, k):
        s[p + "_n"] = float(k.sum())
        s[p + "_E"] = float(E_k[k].sum())
        s[p + "_main"] = float(e_main[k].sum())
        s[p + "_notruth"] = float(nt_k[k].sum())
        s[p + "_merge"] = float((merge & k).sum())
        s[p + "_mix_num"] = float((E_k * np.nan_to_num(nmix))[k].sum())

    add_targets("sel", sel_t)
    add_targets("sig", sig_t)
    add_tracksters("selts", is_sel_ts)
    add_tracksters("sigts", is_sig_ts)
    s["selts_other_sel"] = float((e_sel - e_main)[is_sel_ts].sum())
    s["selts_soft"] = float(e_soft[is_sel_ts].sum())
    s["sigts_pileup"] = float(e_pu[is_sig_ts].sum())
    s["sigts_other_signal"] = float((e_sig - e_main)[is_sig_ts].sum())
    s["all_n"] = float(nT)
    s["all_E"] = float(E_t.sum())
    s["all_best"] = float(best.sum())
    s["all_individual"] = float((individual & counted).sum())
    s["all_split"] = float((split & counted).sum())
    s["all_lost"] = float((lost & counted).sum())
    s["all_n_counted"] = float(counted.sum())
    s["ts_n"] = float(tracksters.n)
    s["ts_E"] = float(E_k.sum())
    s["ts_main"] = float(e_main.sum())
    s["ts_notruth"] = float(nt_k.sum())
    s["ts_fake"] = float(fake.sum())
    s["ts_merge"] = float(merge.sum())
    s["ts_n_lc"] = float(len(tracksters.lc))
    # energy of base particles with no reachable part at all (entirely in masked layer clusters)
    s["untargeted_unreachable"] = targets.E_untargeted

    # ---- objectives: signal truth groups against the ideal clustering ----
    o = ideal_scores(ev, tracksters, targets, cfg, W)
    sc = o["group_scored"]
    kin = (o["ts_group"] >= 0) & (sc[np.maximum(o["ts_group"], 0)] if len(sc) else False)
    s["obj_n"] = float(o["objective"].sum())
    s["obj_groups"] = float(sc.sum())
    s["obj_merged_groups"] = float((sc & (o["group_n_targets"] > 1)).sum())
    s["obj_E"] = float(o["group_E"][sc].sum())
    s["obj_reach"] = float(o["reach"][sc].sum())
    s["obj_Kall"] = float(o["Kall"][sc].sum())
    s["obj_credit"] = float(o["credit"][sc].sum())
    s["obj_missed"] = float((sc & (o["n_pieces"] == 0)).sum())
    fr = sc & np.isfinite(o["nfrag"])
    s["obj_frag_num"] = float((o["group_E"] * np.nan_to_num(o["nfrag"]))[fr].sum())
    s["obj_frag_den"] = float(o["group_E"][fr].sum())
    s["obj_ts_n"] = float(kin.sum())
    s["obj_ts_E"] = float(o["ts_E"][kin].sum())
    s["obj_avoid"] = float(o["ts_avoid"][kin].sum())
    s["obj_unavoid"] = float(o["ts_unavoid"][kin].sum())
    s["obj_own_outside"] = float(o["ts_own_outside"][kin].sum())
    s["signal_E"] = float(targets.E_signal.sum())
    s["signal_in_pileup_targets"] = float(targets.E_signal[~o["signal"]].sum())
    tg_sc = (o["target_group"] >= 0) & (sc[np.maximum(o["target_group"], 0)] if len(sc) else False)
    s["signal_unscored"] = float(targets.E_signal[o["signal"] & ~tg_sc].sum())
    if not keep_details:
        return s
    return dict(sums=s, ideal=o,
                targets=dict(E=E_t, selected=sel_t, signal=sig_t, best=best, best_k=best_k, clustered=clustered,
                             completeness=np.divide(best, E_t, out=np.zeros_like(best), where=E_t > 0),
                             nfrag=nfrag, individual=individual, split=split, lost=lost,
                             pileup_inside=targets.E_pileup, unreachable=targets.E_unreachable,
                             n_members=targets.n_members, n_interactions=targets.n_interactions),
                tracksters=dict(E=E_k, notruth=nt_k, main=main, e_main=e_main,
                                purity=np.divide(e_main, E_k, out=np.zeros_like(e_main), where=E_k > 0),
                                selected=is_sel_ts, soft=e_soft, other_selected=e_sel - e_main,
                                signal=is_sig_ts, pileup=e_pu, other_signal=e_sig - e_main, nmix=nmix,
                                merge=merge, fake=fake, n_lc=np.diff(tracksters.offsets)))


def combine(sums_list):
    out = dict.fromkeys(SUM_KEYS, 0.0)
    for s in sums_list:
        for k in SUM_KEYS:
            out[k] += s[k]
    return out


def _r(a, b):
    return a / b if b > 0 else float("nan")


def _target_ratios(s, p):
    return {
        f"{p}_clustered_frac": _r(s[p + "_clustered"], s[p + "_E"]),
        f"{p}_unclustered_frac": _r(s[p + "_unclustered"], s[p + "_E"]),
        f"{p}_unreachable_frac": _r(s[p + "_unreachable"], s[p + "_E"] + s[p + "_unreachable"]),
        f"{p}_pileup_inside_frac": _r(s[p + "_pileup_inside"], s[p + "_E"]),
        f"{p}_eff_individual": _r(s[p + "_individual"], s[p + "_n_counted"]),
        f"{p}_split_rate": _r(s[p + "_split"], s[p + "_n_counted"]),
        f"{p}_lost_rate": _r(s[p + "_lost"], s[p + "_n_counted"]),
        f"{p}_targets_per_event": _r(s[p + "_n"], s["n_events"]),
        f"{p}_multi_interaction_frac": _r(s[p + "_n_multi_interaction"], s[p + "_n"]),
        f"{p}_multi_origin_frac": _r(s[p + "_n_multi_origin"], s[p + "_n"]),
    }


def _trackster_ratios(s, p):
    return {
        f"{p}_notruth_frac": _r(s[p + "_notruth"], s[p + "_E"]),
        f"{p}_merge_rate": _r(s[p + "_merge"], s[p + "_n"]),
        f"{p}_mix": _r(s[p + "_mix_num"], s[p + "_E"]),
        f"{p}_per_event": _r(s[p + "_n"], s["n_events"]),
    }


def summary(s):
    """Ratios from (combined) sums. The first three are the tuning objectives (C and P; F is reported)."""
    return {
        # --- tuning objectives: signal truth groups against the ideal clustering ---
        "C": _r(s["obj_credit"], s["obj_reach"]),
        "P": 1 - _r(s["obj_avoid"], s["obj_ts_E"]),
        "F": _r(s["obj_frag_num"], s["obj_frag_den"]),
        "obj_targets_per_event": _r(s["obj_n"], s["n_events"]),           # signal targets >= objective_energy
        "obj_groups_per_event": _r(s["obj_groups"], s["n_events"]),
        "obj_merged_group_frac": _r(s["obj_merged_groups"], s["obj_groups"]),  # groups the clustering merged
        "obj_tracksters_per_group": _r(s["obj_ts_n"], s["obj_groups"]),
        "obj_missed_rate": _r(s["obj_missed"], s["obj_groups"]),
        "obj_ideal_completeness": _r(s["obj_reach"], s["obj_E"]),   # what an ideal clustering collects
        "obj_ideal_purity": _r(s["obj_reach"], s["obj_Kall"]),      # and how pure it is
        "obj_unavoidable_frac": _r(s["obj_unavoid"], s["obj_ts_E"]),  # contamination in the tracksters that is free
        "obj_own_outside_frac": _r(s["obj_own_outside"], s["obj_ts_E"]),  # group energy taken outside K_g (not credited)
        "signal_in_pileup_targets_frac": _r(s["signal_in_pileup_targets"], s["signal_E"]),
        "signal_unscored_frac": _r(s["signal_unscored"], s["signal_E"]),  # in groups with no target >= objective_energy
        # --- previous objectives: every selected target, signal or pileup (diagnostic) ---
        "sel_C": _r(s["sel_best"], s["sel_E"]),
        "sel_P": _r(s["selts_main"], s["selts_E"]),
        "sel_F": _r(s["sel_frag_num"], s["sel_frag_den"]),
        # --- selected targets ---
        "sel_energy_frac": _r(s["sel_E"], s["all_E"]),
        **_target_ratios(s, "sel"),
        # --- selected tracksters: P + other_sel + soft + notruth = 1 ---
        "selts_other_sel_frac": _r(s["selts_other_sel"], s["selts_E"]),
        "selts_soft_frac": _r(s["selts_soft"], s["selts_E"]),
        **_trackster_ratios(s, "selts"),
        # --- the same, signal only (diagnostic) ---
        "sig_C": _r(s["sig_best"], s["sig_E"]),
        "sig_P": _r(s["sigts_main"], s["sigts_E"]),
        "sig_F": _r(s["sig_frag_num"], s["sig_frag_den"]),
        **_target_ratios(s, "sig"),
        "sigts_pileup_frac": _r(s["sigts_pileup"], s["sigts_E"]),
        "sigts_other_signal_frac": _r(s["sigts_other_signal"], s["sigts_E"]),
        **_trackster_ratios(s, "sigts"),
        # --- everything (soft targets included) ---
        "all_C": _r(s["all_best"], s["all_E"]),
        "all_P": _r(s["ts_main"], s["ts_E"]),
        "all_eff_individual": _r(s["all_individual"], s["all_n_counted"]),
        "all_split_rate": _r(s["all_split"], s["all_n_counted"]),
        "all_lost_rate": _r(s["all_lost"], s["all_n_counted"]),
        "ts_fake_rate": _r(s["ts_fake"], s["ts_n"]),
        "ts_merge_rate": _r(s["ts_merge"], s["ts_n"]),
        "ts_notruth_frac": _r(s["ts_notruth"], s["ts_E"]),
        "ts_per_event": _r(s["ts_n"], s["n_events"]),
        "lc_per_ts": _r(s["ts_n_lc"], s["ts_n"]),
        "targets_per_event": _r(s["all_n"], s["n_events"]),
    }
