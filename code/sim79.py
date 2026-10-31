"""Simulation with real well shifts (samewell/notes.md §79 F; A100). Well shifts are not independent Gaussian genes
but random combinations of the 14 plates' control-pair differences (per line H11 - H12, divided by sqrt 2), keeping
their real correlation across genes and across lines; half-sample noise is drawn from control wells' half differences
(x_A - x_B over sqrt 2, random sign), scaled per line to the treated wells' noise level; each plate adds a plate effect
of the same kind. Genes: 1,500 drawn from the common genes G0 (seed 0); lines with valid control halves on all plates.
Design per run: 8 plates x 48 wells; 24 replicated conditions (two wells on two different plates), the rest single.
Drug effect of condition c on line l: A s_c (v_c + 0.7 eta_lc), v and eta Gaussian with the genes' shift scale,
s_c = exp(N(-1.5, 1.2)), A set so that s = 1 gives the median shift energy; proxy strength = s_c exp(N(0, 0.5)).
Pipeline as in the analysis: D' per plate and half, centring over genes, W, Q, S_same, S_cross (§42), drug share
Q / W pooled, shared-well share (S_same - S_cross) / W, same-well advantage. Truth: the same D' operator applied to the
noise-free drug part and shift part separately; true drug share = sum of the drug part's energy / sum of the signal's
energy (mean of the two wells). Scenarios: no well shifts; real well shifts; real well shifts scaled up to the size
measured between replicate wells at different positions (factor f = sqrt(median over lines of V / median shift
energy), V as in §22 on the same genes; the control pair is adjacent and fixed, its difference is smaller). Strata: tertiles of the proxy and of the
pair's own r (median over lines of Pearson(D'_a, D'_b)), within each run; 20 runs (seeds 1-20).
Output $WORKSPACE/rt/samewell/sim79.json"""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import json, collections
import numpy as np
import pyarrow.parquet as pq
import sw_core as C

OUT = (_WS + "/rt/samewell/"); AGG = (_WS + "/tahoe_agg/")
NPL, NW, NR, K = 8, 48, 24, 14


def real_parts(d, G):
    idx = json.load(open(AGG + "index.json")); S = np.load(AGG + "sums64.npy", mmap_mode="r"); cnt = np.load(AGG + "counts.npy")
    Ps = len(idx["samples"]); gpos = {g: k for k, g in enumerate(idx["genes"])}; gk = np.array([gpos[d["genes"][g]] for g in G])
    sm = pq.read_table((_WS + "/data_ext/tahoe/Tahoe-100M/metadata/sample_metadata.parquet")).to_pydict()
    info = {s: (dr.strip(), p) for s, dr, p in zip(sm["sample"], sm["drug"], sm["plate"])}
    plates = sorted({p for _, p in info.values()}, key=lambda p: int(p[5:]))
    halves = {}
    for pl in plates:
        ds = sorted([j for j, s in enumerate(idx["samples"]) if info[s][1] == pl and info[s][0].startswith("DMSO")], key=lambda j: int(idx["samples"][j][4:]))
        for i, ln in enumerate(d["lines"]):
            c = idx["lines"].index(ln); w = []
            for j in ds:
                rows = [(c * Ps + j) * 2 + h for h in range(2)]; n = cnt[rows, 0]
                w.append([np.asarray(S[r], np.float64)[gk] / n[h] for h, r in enumerate(rows)] if n.min() >= 10 else None)
            halves[(i, pl)] = w
    lines = [i for i in range(len(d["lines"])) if all(len(halves[(i, p)]) == 2 and None not in halves[(i, p)] for p in plates)]
    Dl = np.stack([[0.5 * (halves[(i, p)][0][0] + halves[(i, p)][0][1]) - 0.5 * (halves[(i, p)][1][0] + halves[(i, p)][1][1]) for i in lines] for p in plates]) / np.sqrt(2)
    Npool = np.stack([[(halves[(i, p)][w][0] - halves[(i, p)][w][1]) / np.sqrt(2) for p in plates for w in range(2)] for i in lines])  # L x 28 x G
    return np.array(lines), Dl, Npool


def pear(a, b):
    a = a - a.mean(); b = b - b.mean(); dd = np.sqrt((a @ a) * (b @ b)); return float(a @ b / dd) if dd > 0 else np.nan


def dprime(H, plate_of):
    out = H.copy()
    for p in range(NPL):
        js = np.where(plate_of == p)[0]; S = H[:, js].sum(1)
        for j in js:
            out[:, j] = H[:, j] - (S - H[:, j]) / (len(js) - 1)
    return out - out.mean(2, keepdims=True)


def run(seed, Dl, Npool, kappa, shifts, f=1.0):
    rng = np.random.default_rng(seed); L, G = Dl.shape[1], Dl.shape[2]
    comb = lambda: f * np.tensordot(rng.normal(size=K), Dl, 1) / np.sqrt(K)
    gs = np.sqrt((Dl ** 2).mean((0, 1)))                                  # gene scale of the shifts
    Eu = f ** 2 * np.median((Dl ** 2).sum(2))                                    # median shift energy per line
    nwell = NPL * NW; plate_of = np.repeat(np.arange(NPL), NW)
    cond = -np.ones(nwell, int); rep = []
    for c in range(NR):                                                   # replicated conditions on two plates
        p1, p2 = rng.choice(NPL, 2, replace=False)
        w1 = rng.choice(np.where((plate_of == p1) & (cond < 0))[0]); w2 = rng.choice(np.where((plate_of == p2) & (cond < 0))[0])
        cond[w1] = cond[w2] = c; rep.append((w1, w2))
    free = np.where(cond < 0)[0]; cond[free] = NR + np.arange(len(free)); nc = cond.max() + 1
    s = np.exp(rng.normal(-1.5, 1.2, nc)); proxy = s * np.exp(rng.normal(0, 0.5, nc))
    A = np.sqrt(Eu / ((gs ** 2).sum() * 1.49))
    drug = np.zeros((L, nwell, G)); shift = np.zeros((L, nwell, G))
    eff = {c: A * s[c] * (rng.normal(size=G) * gs + 0.7 * rng.normal(size=(L, G)) * gs) for c in range(nc)}
    for w in range(nwell):
        drug[:, w] = eff[cond[w]]
        if shifts:
            shift[:, w] = comb()
    pe = [comb() for _ in range(NPL)]
    base = drug + shift + np.stack([pe[p] for p in plate_of], 1)
    H = []
    for h in range(2):
        pick = rng.integers(0, Npool.shape[1], (L, nwell)); sg = rng.choice([-1.0, 1.0], (L, nwell))
        noise = Npool[np.arange(L)[:, None], pick] * (sg * kappa[:, None])[:, :, None]
        H.append(dprime(base + noise, plate_of))
    XA, XB = H; X = (XA + XB) / 2
    Dd = dprime(drug, plate_of); Ds = dprime(shift + np.stack([pe[p] for p in plate_of], 1), plate_of); Sg = Dd + Ds
    recs = []
    for pi, (a, b) in enumerate(rep):
        Xa, Xb = X[:, a], X[:, b]; Sa = Xa @ Xa.T; Sb = Xb @ Xb.T; Sab = Xa @ Xb.T; Ma, Mb = Xa.sum(0), Xb.sum(0)
        own_r = float(np.median([pear(Xa[l], Xb[l]) for l in range(L)]))
        for l in range(L):
            W = 0.5 * (XA[l, a] @ XB[l, a] + XA[l, b] @ XB[l, b]); Q = 0.25 * sum(u @ v for u in (XA[l, a], XB[l, a]) for v in (XA[l, b], XB[l, b]))
            o = np.arange(L) != l
            Ss = 0.5 * (Sa[l, o].mean() + Sb[l, o].mean()); Sc = 0.5 * (Sab[l, o].mean() + Sab[o, l].mean())
            adv = []
            for tg, rp, Mt, Mr in ((a, b, Ma, Mb), (b, a, Mb, Ma)):
                y = X[l, tg]; adv.append(pear((Mt - y) / (L - 1), y) - pear((Mr - X[l, rp]) / (L - 1), y))
            recs.append(dict(pi=pi, W=W, Q=Q, sw=Ss - Sc, adv=float(np.mean(adv)),
                             tD=0.5 * (Dd[l, a] @ Dd[l, a] + Dd[l, b] @ Dd[l, b]), tS=0.5 * (Sg[l, a] @ Sg[l, a] + Sg[l, b] @ Sg[l, b])))
    pr = np.array([proxy[cond[a]] for a, b in rep]); rr = np.array([np.median([pear(X[l, a], X[l, b]) for l in range(L)]) for a, b in rep])
    tert = {"proxy": np.digitize(pr, np.quantile(pr, [1 / 3, 2 / 3])), "own_r": np.digitize(rr, np.quantile(rr, [1 / 3, 2 / 3]))}
    for r in recs:
        r["t_proxy"] = int(tert["proxy"][r["pi"]]); r["t_own_r"] = int(tert["own_r"][r["pi"]])
    return recs


def main():
    d = C.load(); base = d["base"]; L0 = d["D"].shape[0]
    G0 = np.where(np.median(base, 0) >= 0.2)[0]; G = np.sort(np.random.default_rng(0).choice(G0, 1500, replace=False))
    lines, Dl, Npool = real_parts(d, G)
    z = np.load(C.ARR, allow_pickle=True, mmap_mode="r")
    kappa = []
    for i in lines:
        A = np.asarray(z["XA"][i], np.float32)[:, G].astype(np.float64); B = np.asarray(z["XB"][i], np.float32)[:, G].astype(np.float64)
        ok = np.isfinite(A).all(1) & np.isfinite(B).all(1)
        kappa.append(np.sqrt(np.median(((A[ok] - B[ok]) ** 2).sum(1)) / np.median((2 * Npool[list(lines).index(i)] ** 2).sum(1))))
    kappa = np.array(kappa)
    pairs, _ = C.groups(d); other = [(a, b) for c, a, b, g in pairs if g in ("G13", "Gother")]; V = []
    for i in lines:
        A = np.asarray(z["XA"][i], np.float32)[:, G].astype(np.float64); B = np.asarray(z["XB"][i], np.float32)[:, G].astype(np.float64)
        es = [float((A[a] - A[b]) @ (B[a] - B[b])) for a, b in other if np.isfinite(A[a]).all() and np.isfinite(A[b]).all() and np.isfinite(B[a]).all() and np.isfinite(B[b]).all()]
        V.append(np.median(es) / 2)
    f = float(np.sqrt(np.median(V) / np.median((Dl ** 2).sum(2))))
    res = {"rules": __doc__, "n_lines": len(lines), "kappa_median": round(float(np.median(kappa)), 2),
           "shift_energy_median": round(float(np.median((Dl ** 2).sum(2))), 2),
           "V_median": round(float(np.median(V)), 2), "scale_f": round(f, 2)}
    print(json.dumps({k: res[k] for k in ("n_lines", "kappa_median", "shift_energy_median", "V_median", "scale_f")}), flush=True)
    for scen, shifts, fs in (("no well shifts", False, 1.0), ("real well shifts", True, 1.0), ("real well shifts, scaled", True, f)):
        allr = []
        for seed in range(1, 21):
            rs = run(seed, Dl, Npool, kappa, shifts, fs)
            for r in rs:
                r["seed"] = seed
            allr += rs
            print(scen, "seed", seed, flush=True)
        for meas in ("proxy", "own_r"):
            for s in ("all", 0, 1, 2):
                rs = [r for r in allr if s == "all" or r["t_" + meas] == s]
                by = collections.defaultdict(list)
                for r in rs:
                    by[r["seed"]].append(r)
                est = [sum(r["Q"] for r in v) / sum(r["W"] for r in v) for v in by.values()]
                tru = [sum(r["tD"] for r in v) / sum(r["tS"] for r in v) for v in by.values()]
                sw = [sum(r["sw"] for r in v) / sum(r["W"] for r in v) for v in by.values()]
                adv = [np.mean([r["adv"] for r in v]) for v in by.values()]
                e = dict(drug_share_est=round(float(np.mean(est)), 3), drug_share_true=round(float(np.mean(tru)), 3),
                         bias=round(float(np.mean(np.array(est) - np.array(tru))), 3), bias_sd=round(float(np.std(np.array(est) - np.array(tru))), 3),
                         shared_well_share=round(float(np.mean(sw)), 3), advantage=round(float(np.mean(adv)), 3), advantage_sd=round(float(np.std(adv)), 3))
                res["%s | %s | %s" % (scen, meas, s)] = e; print(scen, meas, s, json.dumps(e), flush=True)
    json.dump(res, open(OUT + "sim79.json", "w"), indent=1); print("SUMMARY done", flush=True)


if __name__ == "__main__":
    main()
