"""The metrics of tuning/V3_TRUTH_AND_METRICS.md (section 6), evaluated on rechits for any clustering.

A clustering is a set of objects r with rechit fractions w_rc (`objects_from_*`). With s_t(r) = sum_c w_rc s_t(c)
and E_r = sum_c w_rc E_c, every object is assigned to its best target t*(r) = argmax_t s_t(r) (ties: the larger
deposited energy, then the lower index); an object with no truth energy is unassigned. A signal object is one whose
best target is a signal target. R(t) = the objects assigned to t, N_t = |R(t)|, c_t = sum_{r in R(t)} s_t(r).
Per class X (EM, HAD; the class of the best target):

  K_X   = sum over signal objects of (E_r - s_t*(r)) / sum of E_r            contamination      (minimise)
  eps_X = sum over signal targets of c_t / sum of E_t (deposited)            efficiency         (maximise)
  Phi_X = 1 - sum_t c_t / sum_t c_t N_t  (signal targets with N_t >= 1)       surplus fragments  (minimise)

and K_sig, eps_sig, Phi_sig = the mean over the classes present (K_sig = 1, eps_sig = 0, Phi_sig = 0 if none).
`evaluate` returns additive per-event sums (any split of the events gives the same `summary` of their total);
`summary` turns sums into the metrics and the diagnostics; samples are averaged by the caller (`mean_over_samples`).
"""
import numpy as np
import scipy.sparse as sp

from . import truth as truth_mod

CLASSES = ("EM", "HAD")
E_BINS = (0.0, 5.0, 30.0, 150.0, np.inf)   # of the target's deposited energy, for the diagnostics
NB = len(E_BINS) - 1

# per class (2) x energy bin: the additive sums. "obj" quantities are binned by the best target's class and energy.
KEYS = ("k_num", "k_den", "k_floor",           # signal objects: contamination, energy, cell floor of contamination
        "e_num", "e_den",                       # signal targets: own energy in own objects, deposited energy
        "f_c", "f_cn",                          # signal targets with objects: c_t, c_t * N_t
        "lost_other", "lost_none",              # signal targets: energy in other targets' objects, in no object
        "n_targets", "n_targets_obj", "n_obj",  # signal targets, of which with objects; signal objects
        "pu_sig", "pu_den", "pu_eps_num", "pu_eps_den",  # pileup objects: signal inside, energy; pileup efficiency
        "n_pu_obj", "n_noise_obj", "events")
SHAPE = (len(CLASSES), NB)


def objects_from_labels(labels, n):
    """(n_obj x n) one-hot weights from a label per point (-1: in no object)."""
    labels = np.asarray(labels)
    sel = np.nonzero(labels >= 0)[0]
    uniq, inv = np.unique(labels[sel], return_inverse=True)
    return sp.csr_matrix((np.ones(len(sel)), (inv, sel)), shape=(len(uniq), n))


def objects_from_lc_labels(ev, labels):
    """Rechit fractions of objects given as a label per layer cluster."""
    return (objects_from_labels(labels, ev.n_lc) @ ev.lc_to_rh()).tocsr()


def objects_from_tracksters(ev, ts):
    """Rechit fractions of a CMSSW trackster collection (data.Tracksters)."""
    return (ts.matrix(ev.n_lc) @ ev.lc_to_rh()).tocsr()


def objects(ev, labels, level):
    """Rechit fractions of a clustering given as labels at `level` ("lc", "lc_all": per layer cluster; "rh")."""
    return objects_from_labels(labels, ev.n_rh) if level == "rh" else objects_from_lc_labels(ev, labels)


def assign(W, t):
    """Best target of every object (-1 if it holds no truth energy) and its energy in it."""
    S = (W @ t.T).tocoo()
    best = np.full(W.shape[0], -1, np.int64)
    own = np.zeros(W.shape[0])
    if S.nnz:
        order = np.lexsort((S.col, -t.E[S.col], -S.data, S.row))
        first = np.unique(S.row[order], return_index=True)[1]
        r = S.row[order][first]
        best[r] = S.col[order][first]
        own[r] = S.data[order][first]
    return best, own, S


def _bins(E):
    return np.clip(np.searchsorted(E_BINS, E, side="right") - 1, 0, NB - 1)


def evaluate(ev, W, t):
    """Additive sums of one event for the objects W (n_obj x n_rh rechit fractions) against targets t."""
    out = {k: np.zeros(SHAPE) for k in KEYS}
    E_rh = np.asarray(ev.rh["E"], float)
    E_r = W @ E_rh
    best, own, S = assign(W, t)
    sig_obj = (best >= 0) & t.signal[np.maximum(best, 0)]
    pu_obj = (best >= 0) & ~t.signal[np.maximum(best, 0)]
    cls_t = np.where(t.em, 0, 1)
    bin_t = _bins(t.E)
    # signal objects
    b, k = best[sig_obj], (cls_t[best[sig_obj]], bin_t[best[sig_obj]])
    floor_c = E_rh - (t.T.max(axis=1).toarray().ravel() if t.n else 0.0)
    np.add.at(out["k_num"], k, E_r[sig_obj] - own[sig_obj])
    np.add.at(out["k_den"], k, E_r[sig_obj])
    np.add.at(out["k_floor"], k, (W @ floor_c)[sig_obj])
    np.add.at(out["n_obj"], k, 1)
    # signal targets
    c = np.bincount(b, weights=own[sig_obj], minlength=t.n)
    N = np.bincount(b, minlength=t.n)
    in_any = np.asarray(S.tocsr().sum(axis=0)).ravel() if S.nnz else np.zeros(t.n)
    st = np.nonzero(t.signal)[0]
    kt = (cls_t[st], bin_t[st])
    np.add.at(out["e_num"], kt, c[st])
    np.add.at(out["e_den"], kt, t.E[st])
    np.add.at(out["f_c"], kt, c[st])
    np.add.at(out["f_cn"], kt, c[st] * N[st])
    np.add.at(out["lost_other"], kt, in_any[st] - c[st])
    np.add.at(out["lost_none"], kt, t.E[st] - in_any[st])
    np.add.at(out["n_targets"], kt, 1)
    np.add.at(out["n_targets_obj"], kt, N[st] > 0)
    # pileup objects and targets (diagnostics); binned by the best target, class index from its class
    S_sig = np.asarray((W @ t.T[:, t.signal]).sum(axis=1)).ravel() if t.signal.any() else np.zeros(W.shape[0])
    kp = (cls_t[best[pu_obj]], bin_t[best[pu_obj]])
    np.add.at(out["pu_sig"], kp, S_sig[pu_obj])
    np.add.at(out["pu_den"], kp, E_r[pu_obj])
    np.add.at(out["n_pu_obj"], kp, 1)
    pt = np.nonzero(~t.signal)[0]
    cp = np.bincount(best[pu_obj], weights=own[pu_obj], minlength=t.n)
    np.add.at(out["pu_eps_num"], (cls_t[pt], bin_t[pt]), cp[pt])
    np.add.at(out["pu_eps_den"], (cls_t[pt], bin_t[pt]), t.E[pt])
    out["n_noise_obj"][0, 0] = np.sum(best < 0)
    out["events"][0, 0] = 1
    return out


def combine(sums):
    """Sum of per-event (or per-job) sums."""
    tot = {k: np.zeros(SHAPE) for k in KEYS}
    for s in sums:
        for k in KEYS:
            tot[k] += s[k]
    return tot


def _ratio(a, b):
    return float(a / b) if b > 0 else float("nan")


def _class_mean(vals, empty):
    v = [x for x in vals if np.isfinite(x)]
    return float(np.mean(v)) if v else empty


def summary(s):
    """The three metrics and the diagnostics from (summed) sums of ONE sample."""
    cl = {k: s[k].sum(axis=1) for k in KEYS}   # per class, all energy bins
    out = {}
    K = [_ratio(cl["k_num"][i], cl["k_den"][i]) for i in range(2)]
    eps = [_ratio(cl["e_num"][i], cl["e_den"][i]) for i in range(2)]
    phi = [1 - _ratio(cl["f_c"][i], cl["f_cn"][i]) if cl["f_cn"][i] > 0 else float("nan") for i in range(2)]
    out["K_sig"] = _class_mean(K, 1.0)
    out["eps_sig"] = _class_mean(eps, 0.0)
    out["Phi_sig"] = _class_mean(phi, 0.0)
    for i, c in enumerate(CLASSES):
        out[f"K_{c}"], out[f"eps_{c}"], out[f"Phi_{c}"] = K[i], eps[i], phi[i]
        out[f"K_floor_{c}"] = _ratio(cl["k_floor"][i], cl["k_den"][i])
        out[f"lost_other_{c}"] = _ratio(cl["lost_other"][i], cl["e_den"][i])
        out[f"lost_none_{c}"] = _ratio(cl["lost_none"][i], cl["e_den"][i])
        for j in range(NB):
            tag = f"{c}_E{E_BINS[j]:g}-{E_BINS[j + 1]:g}"
            out[f"K_{tag}"] = _ratio(s["k_num"][i, j], s["k_den"][i, j])
            out[f"eps_{tag}"] = _ratio(s["e_num"][i, j], s["e_den"][i, j])
            out[f"Phi_{tag}"] = 1 - _ratio(s["f_c"][i, j], s["f_cn"][i, j]) if s["f_cn"][i, j] > 0 else float("nan")
    out["K_pu"] = _ratio(s["pu_sig"].sum(), s["pu_den"].sum())
    out["eps_pu"] = _ratio(s["pu_eps_num"].sum(), s["pu_eps_den"].sum())
    ev = max(s["events"].sum(), 1)
    out["signal_objects_per_event"] = s["n_obj"].sum() / ev
    out["pileup_objects_per_event"] = s["n_pu_obj"].sum() / ev
    out["noise_objects_per_event"] = s["n_noise_obj"].sum() / ev
    out["signal_targets_per_event"] = s["n_targets"].sum() / ev
    out["events"] = int(s["events"].sum())
    return out


def mean_over_samples(summaries):
    """The tuning objectives: every metric averaged over the samples with equal weight."""
    out = {}
    for k in summaries[0]:
        v = [s[k] for s in summaries]
        out[k] = int(sum(v)) if k == "events" else _class_mean(v, float("nan"))
    return out


def score(ev, labels, level, t=None):
    """Sums of one event for a clustering given as labels at `level` (targets built if not given)."""
    t = truth_mod.build(ev) if t is None else t
    return evaluate(ev, objects(ev, labels, level), t)
