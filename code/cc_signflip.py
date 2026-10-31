"""Calibration of the across-line sign-flip test of Tahoe's cell-cycle analysis (samewell/notes.md §41; A100).
Counts: $WORKSPACE/tahoe_agg counts.npy (n, G1, S, G2M, lib_sum per (line, sample, half)), halves summed.
Per line l: log2 OR(G1) and log2 OR(G2M) of a treated well against a reference (Haldane-Anscombe +0.5);
Delta_l = log2OR_G1 - log2OR_G2M; T = mean over lines (>= 30 cells in both); two-sided p from 10,000 sign flips (seed 0).
  pseudo   per plate, DMSO well H11 as 'treated' against H12 (zero drug effect)
  drugs    every treated well against the pooled DMSO wells of its plate
  replicate wells at different positions (sw_core groups G13, Gother): correlation of T_a, T_b; concordance of p < 0.05
Output $WORKSPACE/rt/samewell/cc_signflip.json. Usage: cc_signflip.py"""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import json
import numpy as np
import pyarrow.parquet as pq
import sw_core as C

AGG, OUT = (_WS + "/tahoe_agg"), (_WS + "/rt/samewell/")
META = (_WS + "/data_ext/tahoe/Tahoe-100M/metadata/sample_metadata.parquet")
rng = np.random.default_rng(0)
SIGNS = rng.choice([-1.0, 1.0], size=(10000, 60))


def lor(k, n):
    return np.log2((k + 0.5) / (n - k + 0.5))


def test(cnt_t, cnt_r):
    """cnt_*: lines x 5 (n, G1, S, G2M, lib)."""
    ok = (cnt_t[:, 0] >= 30) & (cnt_r[:, 0] >= 30)
    d = (lor(cnt_t[ok, 1], cnt_t[ok, 0]) - lor(cnt_r[ok, 1], cnt_r[ok, 0])) - (lor(cnt_t[ok, 3], cnt_t[ok, 0]) - lor(cnt_r[ok, 3], cnt_r[ok, 0]))
    T = float(d.mean()); null = np.abs((SIGNS[:, :len(d)] * d).mean(1))
    return T, float((np.sum(null >= abs(T)) + 1) / (len(null) + 1)), int(ok.sum())


def main():
    idx = json.load(open(AGG + "/index.json")); cnt = np.load(AGG + "/counts.npy")
    L, P = len(idx["lines"]), len(idx["samples"]); cnt = cnt.reshape(L, P, 2, -1).sum(2)
    sm = pq.read_table(META).to_pydict(); info = {s: (d.strip(), p) for s, d, p in zip(sm["sample"], sm["drug"], sm["plate"])}
    sid = {s: k for k, s in enumerate(idx["samples"])}
    plates = sorted({p for _, p in info.values()}, key=lambda p: int(p[5:])); res = {"rules": __doc__, "pseudo": {}}
    ref = {}
    for p in plates:
        ds = sorted([s for s in idx["samples"] if info[s][1] == p and info[s][0].startswith("DMSO")], key=lambda s: int(s[4:]))
        if len(ds) != 2:
            continue
        T, pv, n = test(cnt[:, sid[ds[0]]], cnt[:, sid[ds[1]]]); res["pseudo"][p] = dict(T=T, p=pv, n_lines=n)
        ref[p] = cnt[:, sid[ds[0]]] + cnt[:, sid[ds[1]]]
    pp = np.array([v["p"] for v in res["pseudo"].values()]); pt = np.array([abs(v["T"]) for v in res["pseudo"].values()])
    res["pseudo_summary"] = dict(n_plates=len(pp), frac_p_lt_005=float(np.mean(pp < 0.05)), frac_p_lt_001=float(np.mean(pp < 0.01)),
                                 median_absT=float(np.median(pt)), max_absT=float(pt.max()))
    print("pseudo", json.dumps(res["pseudo_summary"]), flush=True)
    drug = {}
    for s in idx["samples"]:
        d, p = info[s]
        if not d.startswith("DMSO") and p in ref:
            drug[s] = test(cnt[:, sid[s]], ref[p])
    dp = np.array([v[1] for v in drug.values()]); dT = np.array([abs(v[0]) for v in drug.values()])
    res["drugs_summary"] = dict(n_wells=len(dp), frac_p_lt_005=float(np.mean(dp < 0.05)), median_absT=float(np.median(dT)),
                                frac_absT_below_pseudo_max=float(np.mean(dT < pt.max())))
    print("drugs", json.dumps(res["drugs_summary"]), flush=True)
    d = C.load(); pairs, _ = C.groups(d); Ta, Tb, sa, sb = [], [], [], []
    for c, a, b, g in pairs:
        if g not in ("G13", "Gother"):
            continue
        A, B = str(d["sample"][a]), str(d["sample"][b])
        if A in drug and B in drug:
            Ta.append(drug[A][0]); Tb.append(drug[B][0]); sa.append(drug[A][1] < 0.05); sb.append(drug[B][1] < 0.05)
    Ta, Tb, sa, sb = map(np.array, (Ta, Tb, sa, sb))
    res["replicates"] = dict(n_pairs=len(Ta), r_T=float(np.corrcoef(Ta, Tb)[0, 1]), frac_sig_a=float(sa.mean()), frac_sig_b=float(sb.mean()),
                             frac_sig_both_given_a=float((sa & sb).sum() / max(sa.sum(), 1)),
                             frac_same_sign_given_sig_a=float(np.mean(np.sign(Ta[sa]) == np.sign(Tb[sa]))) if sa.any() else None)
    print("replicates", json.dumps(res["replicates"]), flush=True)
    res["drug_records"] = {s: dict(T=v[0], p=v[1], n=v[2], drug=info[s][0], plate=info[s][1]) for s, v in drug.items()}
    json.dump(res, open(OUT + "cc_signflip.json", "w"), indent=1); print("SUMMARY done", flush=True)


if __name__ == "__main__":
    main()
