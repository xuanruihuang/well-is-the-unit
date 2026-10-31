"""Figure data for the same-well manuscript (samewell/notes.md §89; A100). Per verified pair (verified.py): same-well and
replicate-well predictions, Spearman-Brown split-half ceiling and cross-well agreement (D', line genes, medians over
the pair's lines, both wells as target), with the pair's group, label class (labels84) and no-viability flag
(labels85). Collects, unchanged, the summary numbers the figures plot from: labels84 (control pairs per plate),
core82 / core82_confirmed / core82_nodvia (shares with intervals), eval82 (model evaluation, both references, both
training directions), cc_signflip (control-pair cell-cycle tests), label81 and plate13_81 (plate 13), perm89
(permutations), post82 (Rhaister), ref88 (inert reference), dref92 and corr92 (§92). Output $WORKSPACE/rt/samewell/figdata89.json"""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import json, os
import numpy as np
import sw_core as C
import verified as V

OUT = (_WS + "/rt/samewell/")


def main():
    d = C.load(); base = d["base"]; plate = np.array(d["plate"]); L = d["D"].shape[0]
    z = np.load(C.ARR, allow_pickle=True, mmap_mode="r")
    XA = np.stack([np.asarray(z["XA"][i], np.float32) for i in range(L)]).astype(np.float64)
    XB = np.stack([np.asarray(z["XB"][i], np.float32) for i in range(L)]).astype(np.float64)
    fin = np.isfinite(XA).all(2) & np.isfinite(XB).all(2)
    for H in (XA, XB):
        Hz = np.where(fin[:, :, None], H, 0)
        for p in sorted(set(plate)):
            js = np.where(plate == p)[0]; S = Hz[:, js].sum(1); n = fin[:, js].sum(1)
            for j in js:
                H[:, j] -= (S - Hz[:, j]) / np.maximum(n - fin[:, j], 1)[:, None]
        H[~fin] = 0
    X = (XA + XB) / 2; Ss = X.sum(0); nf = fin.sum(0)
    l84 = {(x["a"], x["b"]): x["cls"] for x in json.load(open(OUT + "labels84.json"))["pairs"]}
    l85 = {(x["a"], x["b"]) for x in json.load(open(OUT + "labels85.json"))["pairs"] if x["cls"] == "nodvia"}
    dv = V.dvia_all(); sbf = lambda c: 2 * c / (1 + c); pairs = []
    for p in V.pairs(d, dv):
        a, b = p["a"], p["b"]; ls = np.where(fin[:, a] & fin[:, b])[0]
        if len(ls) < 10:
            continue
        same, other, ceil, cross = [], [], [], []
        for t in ls:
            G = np.where(base[t] >= 0.2)[0]
            for tg, rp in ((a, b), (b, a)):
                y = X[t, tg, G]; mo = lambda j: (Ss[j, G] - X[t, j, G]) / (nf[j] - 1)
                same.append(C.pear(mo(tg), y)); other.append(C.pear(mo(rp), y))
            ceil.append(np.mean([sbf(C.pear(XA[t, w, G], XB[t, w, G])) for w in (a, b)])); cross.append(C.pear(X[t, a, G], X[t, b, G]))
        pairs.append(dict(group=p["group"], comp=p["comp"], salt=p["salt"], plates=[p["pa"], p["pb"]], cls=l84.get((a, b)), nodvia=(a, b) in l85, dvia=p["dvia"],
                          same=round(float(np.nanmedian(same)), 4), other=round(float(np.nanmedian(other)), 4),
                          ceiling=round(float(np.nanmedian(ceil)), 4), cross=round(float(np.nanmedian(cross)), 4), n_lines=int(len(ls))))
    res = {"rules": __doc__, "pairs": pairs}
    J = lambda f: json.load(open(OUT + f)) if os.path.exists(OUT + f) else None
    lab = J("labels84.json"); res["control_per_plate"] = lab["control_per_plate"]
    for f in ("core82.json", "core82_confirmed.json", "core82_nodvia.json", "eval82_fewshot_dprime_confirmed.json", "eval82_fewshot_confirmed.json",
              "eval82_fewshot_dprime_confirmed_swap.json", "perm89.json", "post82.json", "ref88.json", "ref88b.json", "check83.json"):
        v = J(f)
        if v is not None:
            if isinstance(v, dict):
                v = {k: x for k, x in v.items() if k not in ("rules", "A_rows")}
            res[f] = v
    res["eval82_fewshot_dmso_confirmed.json"] = res.get("eval82_fewshot_confirmed.json")      # control-well reference run
    res["check83_rows"] = [{k: r[k] for k in ("group", "salt", "cls", "expr_pct", "via_pct")} for r in J("check83.json")["A_rows"]]
    # per-compound medians of the evaluation records (Fig. 3a paired points; §95)
    er = J("eval82_fewshot_dprime_confirmed_records.json")
    if er is not None:
        import collections
        byc = collections.defaultdict(list)
        for r in er:
            byc[(r["group"], r["comp"])].append(r)
        res["fig3a_compounds"] = [dict(group=g, comp=c, n=len(v), **{k: float(np.median([r[k] for r in v])) for k in ("pmean_train", "pmean_held", "additive_train", "additive_held")})
                                  for (g, c), v in byc.items() if g in ("G614", "diffpos")]
    if J("p13strong92.json") is not None:
        res["p13strong92_rows"] = J("p13strong92.json")["rows"]
    for f in ("dref92.json", "corr92.json"):
        if J(f) is not None:
            res[f] = {k: v for k, v in J(f).items() if k != "rules"}
    res["cc_control"] = J("cc_signflip.json")["pseudo"]
    res["label81_pairs"] = [dict(group=r["group"], pct=r["pct"], r=r["r"], amp=r["amp"], dvia=r["dvia"]) for r in J("label81.json")["pairs"]]
    res["plate13_rows"] = J("plate13_81.json")["rows"]
    json.dump(res, open(OUT + "figdata89.json", "w"), default=float); print("pairs", len(pairs), "SUMMARY done", flush=True)


if __name__ == "__main__":
    main()
