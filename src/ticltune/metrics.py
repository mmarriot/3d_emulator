"""Metrics of trackster building against the truth targets of `truth.build`.

Shared energy of target t and trackster k:   e(t,k) = sum_l w_kl T(l,t)
(w_kl: the fraction of layer cluster l given to trackster k; T: the targets' energies.)
A trackster's energy decomposes exactly as  E_k = sum_t e(t,k) + n_k,  n_k = its no-truth energy
(out-of-time pileup and noise: rechit energy in cells with no in-time sim energy).

Selected targets: targets with reachable energy E_t >= select_energy (default 5 GeV), signal or
pileup alike. The cut is applied to targets, i.e. after linking, so a soft particle inseparable from
a hard one is part of the hard target. Unselected (soft) targets are treated like noise: they never
enter C or F, and they only lower the purity of the tracksters they are merged into.

Tuning objectives (selected targets; see `summary`):
  C  completeness  = sum_{t sel} max_k e(t,k)               / sum_{t sel} E_t
  P  purity        = sum_{k sel} e(main(k), k)              / sum_{k sel} E_k
     where main(k) is the target with the largest e in k (any target), a selected trackster is one
     whose main target is selected, and E_k includes energy from other selected targets (separable
     merges), from soft targets and no-truth. Tracksters led by a soft target are not scored.
  F  fragmentation = energy-weighted mean over selected targets of N_frag(t) = 1 / sum_k f_tk^2,
     f_tk = e(t,k) / sum_k e(t,k): 1 = all clustered energy in one trackster, 2 = an even split.

Every other number is a diagnostic, including the same objectives restricted to signal targets
(sig_C, sig_P, sig_F: a signal target holds a signal particle, a signal trackster's main target is
a signal target). Inseparable pileup is inside the signal targets by construction and is reported
(sig_pileup_inside_frac), never penalised.

All per-event results are additive sums, so events can be scored anywhere and `combine`d.
"""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .data import Tracksters


@dataclass(frozen=True)
class MetricConfig:
    individual_frac: float = 0.5   # outcome "individual": e > this * E_t and e > this * E_k (strict, as TICL)
    split_frac: float = 0.5        # outcome "split": the target's own fragments hold >= this * E_t
    merge_frac: float = 0.1        # a trackster is a merge if >= 2 targets each give >= this * E_k
    fake_frac: float = 0.5         # a trackster is fake if its no-truth energy >= this * E_k
    min_target_energy: float = 0.0  # targets below this [GeV] are left out of the COUNTING metrics
    select_energy: float = 5.0     # targets with reachable energy >= this [GeV] are the objectives' targets


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
)


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
    # energy of base particles that has no reachable part at all (entirely in masked layer clusters)
    full_bp = np.bincount(ev.tr_bp, weights=ev.tr_E, minlength=ev.n_bp)
    s["untargeted_unreachable"] = float(full_bp[targets.member_of < 0].sum())
    if not keep_details:
        return s
    return dict(sums=s,
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
    """Ratios from (combined) sums. The first three are the tuning objectives."""
    return {
        # --- tuning objectives (selected targets) ---
        "C": _r(s["sel_best"], s["sel_E"]),
        "P": _r(s["selts_main"], s["selts_E"]),
        "F": _r(s["sel_frag_num"], s["sel_frag_den"]),
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
