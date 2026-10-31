"""Pair-unit permutation with every line's coin, the target's included (samewell/notes.md §89; A100). For a pair of
wells (1, 2), each line's responses are swapped between the wells with probability 1/2, and the statistic is recomputed
on the swapped data: T_t = Pearson(A'_t, mean_{l!=t} A'_l) - Pearson(A'_t, mean_{l!=t} B'_l), with A' the (swapped)
well-1 responses and B' the well-2 responses, both orientations; pair statistic = mean over targets; group statistic =
mean over pairs; 1,000 permutations (seed 0). D' responses on the common genes, centred; Gram-matrix algebra.
Classes from labels84.json / labels85.json. Output $WORKSPACE/rt/samewell/perm89.json (observed, permuted values)."""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import json
import numpy as np
import sw_core as C
import verified as V

OUT = (_WS + "/rt/samewell/"); NB = 1000


def stat(sig, G11, G12, G22):
    G21 = G12.T; s = sig.astype(bool); vals = []
    for (H11, H12, H21, H22) in ((G11, G12, G21, G22), (G22, G21, G12, G11)):
        S0, S1, S2, S3 = ~s[:, None] & ~s[None, :], ~s[:, None] & s[None, :], s[:, None] & ~s[None, :], s[:, None] & s[None, :]
        MA = np.where(S0, H11, 0) + np.where(S1, H12, 0) + np.where(S2, H21, 0) + np.where(S3, H22, 0)
        MB = np.where(S0, H22, 0) + np.where(S1, H21, 0) + np.where(S2, H12, 0) + np.where(S3, H11, 0)
        MAB = np.where(S0, H12, 0) + np.where(S1, H11, 0) + np.where(S2, H22, 0) + np.where(S3, H21, 0)
        dg = np.diag(MA)
        numA = MA.sum(1) - dg; numB = MAB.sum(1) - np.diag(MAB)
        nA = MA.sum() - 2 * MA.sum(1) + dg; nB = MB.sum() - 2 * MB.sum(1) + np.diag(MB)
        vals.append(numA / np.sqrt(np.maximum(nA, 1e-12) * dg) - numB / np.sqrt(np.maximum(nB, 1e-12) * dg))
    return float(np.mean(np.concatenate(vals)))


def main():
    d = C.load(); base = d["base"]; plate = np.array(d["plate"]); L = d["D"].shape[0]
    G0 = np.where(np.median(base, 0) >= 0.2)[0]
    z = np.load(C.ARR, allow_pickle=True, mmap_mode="r")
    XA = np.stack([np.asarray(z["XA"][i], np.float32)[:, G0] for i in range(L)]).astype(np.float64)
    XB = np.stack([np.asarray(z["XB"][i], np.float32)[:, G0] for i in range(L)]).astype(np.float64)
    fin = np.isfinite(XA).all(2) & np.isfinite(XB).all(2)
    for H in (XA, XB):
        Hz = np.where(fin[:, :, None], H, 0)
        for p in sorted(set(plate)):
            js = np.where(plate == p)[0]; S = Hz[:, js].sum(1); n = fin[:, js].sum(1)
            for j in js:
                H[:, j] -= (S - Hz[:, j]) / np.maximum(n - fin[:, j], 1)[:, None]
    X = (XA + XB) / 2
    l84 = {(x["a"], x["b"]): x["cls"] for x in json.load(open(OUT + "labels84.json"))["pairs"]}
    l85 = {(x["a"], x["b"]) for x in json.load(open(OUT + "labels85.json"))["pairs"] if x["cls"] == "nodvia"}
    P = [p for p in V.pairs(d) if (fin[:, p["a"]] & fin[:, p["b"]]).sum() >= 10]
    sets = {"diffpos confirmed": lambda p: p["group"] == "diffpos" and l84.get((p["a"], p["b"])) == "confirmed",
            "diffpos nodvia": lambda p: p["group"] == "diffpos" and (p["a"], p["b"]) in l85,
            "sameplate confirmed": lambda p: p["group"] == "sameplate" and l84.get((p["a"], p["b"])) == "confirmed",
            "sameplate nodvia": lambda p: p["group"] == "sameplate" and (p["a"], p["b"]) in l85,
            "G614": lambda p: p["group"] == "G614"}
    rng = np.random.default_rng(0); res = {"rules": __doc__}
    for name, f in sets.items():
        ps = [p for p in P if f(p)]; obs, perm = [], []
        for p in ps:
            a, b = p["a"], p["b"]; ls = np.where(fin[:, a] & fin[:, b])[0]
            Ya = X[ls, a] - X[ls, a].mean(1, keepdims=True); Yb = X[ls, b] - X[ls, b].mean(1, keepdims=True)
            G11, G22, G12 = Ya @ Ya.T, Yb @ Yb.T, Ya @ Yb.T
            obs.append(stat(np.zeros(len(ls)), G11, G12, G22))
            perm.append([stat(rng.random(len(ls)) < 0.5, G11, G12, G22) for _ in range(NB)])
        o = float(np.mean(obs)); pm = np.mean(np.array(perm), 0)
        res[name] = dict(n_pairs=len(ps), observed=round(o, 3), perm_q=[round(float(x), 3) for x in np.percentile(pm, [2.5, 97.5])],
                         perm_max=round(float(pm.max()), 3), p=float((1 + (pm >= o).sum()) / (1 + NB)), perm=[round(float(x), 4) for x in pm])
        print(name, json.dumps({k: v for k, v in res[name].items() if k != "perm"}), flush=True)
    json.dump(res, open(OUT + "perm89.json", "w")); print("SUMMARY done", flush=True)


if __name__ == "__main__":
    main()
