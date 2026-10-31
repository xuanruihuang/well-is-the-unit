"""Same-well checks in a design without pooling: sci-Plex3 (samewell/notes.md §76; 5090).
Input: sciplex_pb.py output (one cell line per plate; every treated condition in two or four wells on different plates and
positions; two vehicle wells per plate at one of four position pairs). Per line, genes with vehicle mean >= 0.2.
References: dmso = well minus the plate's vehicle mean (halves minus the same half of the vehicle wells); plate = well minus
the mean of the plate's other treated wells.
  1 halves     per replicate pair (a, b): within = mean Pearson of the two halves of a well; across = mean Pearson of a half
               of a with a half of b (same cell numbers); full = Pearson of the two whole wells. Strata: tertiles of full r
               on the plate reference (pooled over lines). Interval: 1,000 bootstrap resamples of compounds (seed 0) for
               the mean of within - across.
  2 components W = mean inner product of a well's two halves, Q = mean inner product of halves across the two wells (per
               gene); drug = Q, well = W - Q (a single line per well cannot split the shared and the own well shift);
               pooled ratios sum / sum W.
  3 constant   T = the line's mean treated profile leaving the well out; Pearson(T - c, y) with y on the dmso reference and
               c = the plate's own vehicle mean (shared with the truth), the vehicles of the line's other plates with the
               same vehicle positions, or with other positions; own_split: the constant minus vehicle half A scored against
               the well minus vehicle half B (same vehicle wells, different vehicle cells), which keeps the vehicle wells'
               own shift and removes their cell-sampling noise.
  4 calibration (Wang 2026) split-half reliability: each condition's cells pooled over its wells and split at random
               (the cached halves re-pooled exactly with the UMI totals), effect against the line's vehicle cells split
               the same way, rho = 2 r / (1 + r), reliable if rho >= 0.5; the same with the halves replaced by the two
               wells (well a against vehicle half A, well b against half B; same cell numbers, same correction).
               (§49) DRF and BDS of the Pearson delta metric with truth = half B of the target well, positive control =
               half A of the same well or of the replicate well, negative control = the line's mean response over the
               other conditions.
Output $WORKSPACE/mainline_rt_samewell/sciplex/sciplex_checks.json"""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import json, csv, collections, itertools
import numpy as np

D = (_WS + "/mainline_rt_samewell/sciplex/"); P = D + "pb/"


def pear(a, b):
    a = a - a.mean(); b = b - b.mean(); d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else np.nan


def main():
    X = np.load(P + "X.npy").astype(np.float64); XA = np.load(P + "XA.npy").astype(np.float64); XB = np.load(P + "XB.npy").astype(np.float64)
    tot = np.load(P + "tot.npy"); rows = list(csv.DictReader(open(P + "obs.csv")))
    ctx = np.array([r["context"] for r in rows]); plate = np.array([r["plate"] for r in rows]); pos = np.array([r["position"] for r in rows])
    cond = np.array([r["condition"] for r in rows]); ctl = np.array([r["is_control"] == "1" for r in rows])
    comp = np.array([c.split("|")[0] for c in cond]); ncell = np.array([int(r["n_cells"]) for r in rows])
    CA = np.expm1(XA) * tot[0][:, None] / 1e4; CB = np.expm1(XB) * tot[1][:, None] / 1e4  # counts of the kept genes
    lg = lambda c, t: np.log1p(1e4 * c / max(t, 1.0))
    recs, const, cal = [], [], []
    for line in sorted(set(ctx)):
        L = np.where(ctx == line)[0]; vw = L[ctl[L]]; g = np.where(X[vw].mean(0) >= 0.2)[0]
        y, yh, yp, yph, cpos, cmean, chalf = {}, {}, {}, {}, {}, {}, {}
        for p in sorted(set(plate[L])):
            ks = L[(plate[L] == p) & ctl[L]]; ts = L[(plate[L] == p) & ~ctl[L]]
            cpos[p] = tuple(sorted(pos[ks])); cmean[p] = X[ks].mean(0)
            cA, cB = XA[ks].mean(0), XB[ks].mean(0); chalf[p] = (cA, cB); sF, sA, sB = X[ts].sum(0), XA[ts].sum(0), XB[ts].sum(0); n = len(ts)
            for t in ts:
                y[t] = X[t] - cmean[p]; yh[t] = (XA[t] - cA, XB[t] - cB)
                yp[t] = X[t] - (sF - X[t]) / (n - 1); yph[t] = (XA[t] - (sA - XA[t]) / (n - 1), XB[t] - (sB - XB[t]) / (n - 1))
        by = collections.defaultdict(list)
        for t in y:
            by[cond[t]].append(t)
        # 1, 2 halves and components
        for c, ws in by.items():
            for a, b in itertools.combinations(ws, 2):
                rec = dict(line=line, condition=c, compound=comp[a], n_a=int(ncell[a]), n_b=int(ncell[b]))
                for ref, F, H in (("dmso", y, yh), ("plate", yp, yph)):
                    ha, hb = [v[g] for v in H[a]], [v[g] for v in H[b]]
                    rec[ref + "_within"] = 0.5 * (pear(*ha) + pear(*hb))
                    rec[ref + "_across"] = float(np.mean([pear(u, v) for u in ha for v in hb]))
                    rec[ref + "_full"] = pear(F[a][g], F[b][g])
                    rec[ref + "_W"] = 0.5 * (ha[0] @ ha[1] + hb[0] @ hb[1]) / len(g)
                    rec[ref + "_Q"] = 0.25 * sum(u @ v for u in ha for v in hb) / len(g)
                recs.append(rec)
        # 3 constant
        tr = list(y); S = X[tr].sum(0); n = len(tr)
        for t in tr:
            T = (S - X[t]) / (n - 1); p = plate[t]
            same = [cmean[q] for q in cmean if q != p and cpos[q] == cpos[p]]; diff = [cmean[q] for q in cmean if cpos[q] != cpos[p]]
            const.append(dict(line=line, plate=p, own=pear((T - cmean[p])[g], y[t][g]), own_split=pear((T - chalf[p][0])[g], (X[t] - chalf[p][1])[g]),
                              same_pos=pear((T - np.mean(same, 0))[g], y[t][g]) if same else np.nan,
                              other_pos=pear((T - np.mean(diff, 0))[g], y[t][g]) if diff else np.nan))
        # 4 calibration
        vA = lg(CA[vw].sum(0), tot[0][vw].sum()); vB = lg(CB[vw].sum(0), tot[1][vw].sum())
        allr = np.array([y[t] for t in tr]); idx = {t: k for k, t in enumerate(tr)}; tsum = allr.sum(0)
        for c, ws in by.items():
            mu_c = (tsum - allr[[idx[t] for t in ws]].sum(0)) / (len(tr) - len(ws))
            h1 = lg(CA[ws].sum(0), tot[0][ws].sum()) - vA; h2 = lg(CB[ws].sum(0), tot[1][ws].sum()) - vB
            r_half = pear(h1[g], h2[g])
            for a, b in itertools.combinations(ws, 2):
                wa = lg(CA[a] + CB[a], tot[0][a] + tot[1][a]) - vA; wb = lg(CA[b] + CB[b], tot[0][b] + tot[1][b]) - vB
                r_well = pear(wa[g], wb[g]); e = dict(line=line, condition=c, r_half=r_half, r_well=r_well,
                                                       rho_half=2 * r_half / (1 + r_half), rho_well=2 * r_well / (1 + r_well))
                for tg, rp in ((a, b), (b, a)):
                    truth = yh[tg][1][g]; neg = pear(mu_c[g], truth)
                    for k, pc in (("same", yh[tg][0][g]), ("other", yh[rp][0][g])):
                        pv = pear(pc, truth); e["pos_" + k + ("_a" if tg == a else "_b")] = pv
                        e["drf_" + k + ("_a" if tg == a else "_b")] = float(np.clip((pv - neg) / (1 - neg), -1, 1)) if neg < 1 else np.nan
                        e["bds_" + k + ("_a" if tg == a else "_b")] = float(pv > neg)
                    e["neg" + ("_a" if tg == a else "_b")] = neg
                cal.append(e)
        print("line", line, "pairs", sum(r["line"] == line for r in recs), flush=True)
    # summaries
    res = {"rules": __doc__, "n_pairs": len(recs)}
    full = np.array([r["plate_full"] for r in recs]); q = np.nanquantile(full, [1 / 3, 2 / 3]); tert = np.digitize(full, q)
    rng = np.random.default_rng(0); comps = sorted({r["compound"] for r in recs}); cix = np.array([comps.index(r["compound"]) for r in recs])
    K = np.stack([np.bincount(rng.integers(0, len(comps), len(comps)), minlength=len(comps)) for _ in range(1000)])
    for sname, m in [("all", np.ones(len(recs), bool))] + [("full r tertile %d" % k, tert == k) for k in range(3)]:
        e = {"n": int(m.sum())}
        for ref in ("dmso", "plate"):
            wi = np.array([r[ref + "_within"] for r in recs]); ac = np.array([r[ref + "_across"] for r in recs]); fu = np.array([r[ref + "_full"] for r in recs])
            d = wi - ac; ok = m & np.isfinite(d)
            bm = (K @ np.bincount(cix[ok], weights=d[ok], minlength=len(comps))) / np.maximum(K @ np.bincount(cix[ok], minlength=len(comps)), 1)
            Wt = np.array([r[ref + "_W"] for r in recs])[m]; Qt = np.array([r[ref + "_Q"] for r in recs])[m]
            e[ref] = dict(within=round(float(np.nanmedian(wi[m])), 3), across=round(float(np.nanmedian(ac[m])), 3), full=round(float(np.nanmedian(fu[m])), 3),
                          within_minus_across=[round(float(d[ok].mean()), 3)] + [round(float(v), 3) for v in np.percentile(bm, [2.5, 97.5])],
                          frac_within_higher=round(float(np.mean(d[ok] > 0)), 3),
                          components=dict(drug=round(float(Qt.sum() / Wt.sum()), 3), well=round(float((Wt - Qt).sum() / Wt.sum()), 3)))
        res["halves | " + sname] = e; print("halves", sname, json.dumps(e), flush=True)
    for line in ["all"] + sorted(set(ctx)):
        cs = [c for c in const if line == "all" or c["line"] == line]
        res["constant | " + line] = dict({k: round(float(np.nanmedian([c[k] for c in cs])), 3) for k in ("own", "own_split", "same_pos", "other_pos")}, n=len(cs))
        print("constant", line, res["constant | " + line], flush=True)
    for line in ["all"] + sorted(set(ctx)):
        cs = [c for c in cal if line == "all" or c["line"] == line]; e = {"n_pairs": len(cs)}
        for k in ("half", "well"):
            rho = np.array([c["rho_" + k] for c in cs]); e["rho_" + k] = round(float(np.nanmedian(rho)), 3); e["reliable_" + k] = round(float(np.nanmean(rho >= 0.5)), 3)
        for k in ("same", "other"):
            dv = [c["drf_%s_%s" % (k, s)] for c in cs for s in "ab"]; bv = [c["bds_%s_%s" % (k, s)] for c in cs for s in "ab"]
            e["drf_" + k] = round(float(np.nanmean(dv)), 3); e["bds_" + k] = round(float(np.nanmean(bv)), 3)
        res["calibration | " + line] = e; print("calibration", line, json.dumps(e), flush=True)
    json.dump(res, open(D + "sciplex_checks.json", "w"), indent=1)
    json.dump(dict(pairs=recs, constant=const, calibration=cal), open(D + "sciplex_records.json", "w"))
    print("SUMMARY done", flush=True)


if __name__ == "__main__":
    main()
