"""Readout structure on Tahoe-100M (A100), and the zero-shot question: can a cell line's readout be predicted from its
baseline expression alone? Follows explore_AC (Z1 r): gamma ~ shared low-dimensional state x line-specific linear
readout.
Arrays: tahoe_program.py cache (same lines, responsive conditions, genes, halves; XA / XB lines x conditions x genes,
effects vs the line's plate-matched DMSO; base = DMSO mean log1p per line).
Per line c and condition p: ref_{-c}(p) = mean over the other lines (halves apart); target t_c = x_c - ref_{-c} (the
line's context-specific part). State z_c(p) = ref_{-c, A}(p) V, V = top k right singular vectors of the training
conditions' all-line mean response (half A). Readout M_c (k x G): ridge of t_{c, B} on z_c (lambda from inner CV).
Shares of the denoised target signal (||tB||^2 - ||tA - tB||^2 / 2) explained, pooled over lines:
  (1) within-line readout: conditions in 2 folds (seed 11); fit on one, score on the other.
  (2) zero-shot, leave one line out: V recomputed without the held-out line c*; readouts of the other 47 lines fitted
      on all conditions; c*'s readout predicted as
        mean      plain mean of the other readouts (no baseline information; ~0 by symmetry)
        knn       weighted mean of the 5 lines most similar in baseline (Pearson over genes of base - across-line mean),
                  weights = correlation
        ridge     ridge from the other lines' baseline (top 10 PCs of relative baseline) to their vectorised readouts
      scored on all of c*'s conditions (no data of c* except its baseline).
k in (5, 10, 20). Also the scalar version (t_c = a_c * ref, one gain per line) as reference.
Usage: tahoe_readout.py <tahoe_program out_dir with tahoe_arrays.npz> <out_dir>. Exploratory."""
import os, sys, json
import numpy as np

KS = (5, 10, 20)
LAMS = (1e-3, 1e-2, 1e-1, 1, 10, 100)


def ridge(Z, Y, lam):
    k = Z.shape[1]
    return np.linalg.solve(Z.T @ Z + lam * np.trace(Z.T @ Z) / k * np.eye(k), Z.T @ Y)


def pick_lam(Z, Y, rng):
    n = len(Z); perm = rng.permutation(n); a, b = perm[: n // 2], perm[n // 2:]
    best = None
    for lam in LAMS:
        e = sum(((Y[v] - Z[v] @ ridge(Z[t], Y[t], lam)) ** 2).sum() for t, v in ((a, b), (b, a)))
        if best is None or e < best[0]:
            best = (e, lam)
    return best[1]


def sig_expl(tA, tB, F):
    sig = (tB ** 2).sum() - ((tA - tB) ** 2).sum() / 2
    return (tB ** 2).sum() - ((tB - F) ** 2).sum(), sig


def main():
    src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
    z = np.load(src + "/tahoe_arrays.npz", allow_pickle=True)
    resp = z["resp"]; XA = z["XA"][:, resp].astype(np.float64); XB = z["XB"][:, resp].astype(np.float64)
    base = z["base"].astype(np.float64); lines = list(z["lines"])
    L, P, G = XA.shape
    sA, sB = XA.sum(0), XB.sum(0)
    refA = (sA[None] - XA) / (L - 1); refB = (sB[None] - XB) / (L - 1)
    tA, tB = XA - refA, XB - refB
    res = {"rules": __doc__, "n_lines": L, "n_conditions": P, "n_genes": G}
    # scalar reference: one gain per line on ref (cross-fitted: gain from half A target vs half A ref is noisy -> use
    # half A target on half B ref)
    num = den = 0.0
    for c in range(L):
        a = (tA[c] * refB[c]).sum() / max((refB[c] * refA[c]).sum(), 1e-12)
        e, s = sig_expl(tA[c], tB[c], a * refA[c]); num += e; den += s
    res["scalar_gain"] = float(num / den)
    perm = np.random.default_rng(11).permutation(P); folds = (perm[: P // 2], perm[P // 2:])
    rng = np.random.default_rng(0)
    for k in KS:
        # (1) within-line readout, condition folds
        num = den = 0.0
        for c in range(L):
            F = np.zeros((P, G))
            for f in (0, 1):
                tr, te = folds[f], folds[1 - f]
                V = np.linalg.svd(sA[tr] / L, full_matrices=False)[2][:k].T
                Ztr, Zte = refA[c, tr] @ V, refA[c, te] @ V
                lam = pick_lam(Ztr, tB[c, tr], rng)
                F[te] = Zte @ ridge(Ztr, tB[c, tr], lam)
            e, s = sig_expl(tA[c], tB[c], F); num += e; den += s
        res["k%d_within_line" % k] = float(num / den)
        # (2) zero-shot leave-one-line-out
        acc = {m: [0.0, 0.0] for m in ("mean", "knn", "ridge", "oracle_all_conditions")}
        relb = base - base.mean(0)
        per_line = {}
        for cs in range(L):
            others = [c for c in range(L) if c != cs]
            V = np.linalg.svd((sA - XA[cs]) / (L - 1), full_matrices=False)[2][:k].T
            Ms = []
            for c in others:
                Zc = refA[c] @ V
                Ms.append(ridge(Zc, tB[c], pick_lam(Zc, tB[c], rng)).ravel())
            Ms = np.array(Ms)
            rb = relb[others]; x = relb[cs]
            corr = np.array([np.corrcoef(x, r)[0, 1] for r in rb])
            nn = np.argsort(-corr)[:5]; w = np.clip(corr[nn], 1e-6, None)
            pred = {"mean": Ms.mean(0), "knn": (w[:, None] * Ms[nn]).sum(0) / w.sum()}
            U, S, Vt = np.linalg.svd(rb - rb.mean(0), full_matrices=False)
            Q = (rb - rb.mean(0)) @ Vt[:10].T; q = (x - rb.mean(0)) @ Vt[:10].T
            Wr = ridge(Q, Ms - Ms.mean(0), 1.0)
            pred["ridge"] = Ms.mean(0) + q @ Wr
            Zs = refA[cs] @ V
            pred["oracle_all_conditions"] = ridge(Zs, tB[cs], pick_lam(Zs, tB[cs], rng)).ravel()   # in-sample ceiling
            per_line[lines[cs]] = {}
            for m, Mv in pred.items():
                F = Zs @ Mv.reshape(k, G)
                e, s = sig_expl(tA[cs], tB[cs], F); acc[m][0] += e; acc[m][1] += s
                per_line[lines[cs]][m] = float(e / s)
            per_line[lines[cs]]["nn_corr"] = float(corr[nn].mean())
        res["k%d_zero_shot" % k] = {m: float(a / b) for m, (a, b) in acc.items()}
        res["k%d_zero_shot_per_line" % k] = per_line
        print("k", k, "within", round(res["k%d_within_line" % k], 4), "zero-shot",
              {m: round(v, 4) for m, v in res["k%d_zero_shot" % k].items()}, flush=True)
        json.dump(res, open(out + "/metrics_tahoe_readout.json", "w"), indent=1)
    print("scalar_gain", round(res["scalar_gain"], 4), "saved", flush=True)


if __name__ == "__main__":
    main()
