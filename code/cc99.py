"""Calibration of the cell-cycle sign-flip test with the original design (samewell/notes.md §108, independent review
10-29 item A11; A100). The paper calibrated the test with one control well against the other, while the original
analysis tests a treated well against the plate's two pooled control wells; the variances differ. Here the original
design itself is applied to treated wells whose drugs should not move the cell cycle: no detectable viability effect
(dvia < 0.289, as labels85) and a mechanism (drug_metadata moa-fine, also moa-broad and targets) without cell-cycle,
DNA-damage, mitotic, transcription or chromatin terms (keyword list below), plate 13 excluded. Reported: wells, share
with p < 0.05 and p < 0.01 (from cc_signflip.json drug_records, 10,000 sign flips), binomial probability of at least
that many under 5%. Drugs of these wells may still act a little on the cell cycle, so the share is an upper bound on
the test's false-positive rate in this design.
Output $WORKSPACE/rt/samewell/cc99.json"""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import json, math, re
import numpy as np
import pyarrow.parquet as pq
import verified as V

OUT = (_WS + "/rt/samewell/"); DM = (_WS + "/data_ext/tahoe/Tahoe-100M/metadata/drug_metadata.parquet")
KEY = re.compile(r"cdk|cyclin|aurora|plk|polo|kinesin|tubulin|microtub|mitot|spindle|topoisomer|dna|replicat|nucleo|"
                 r"antimetab|thymid|purine|pyrimid|ribonucl|parp|atr|atm|chk|wee1|hdac|bet|brd|bromodom|methyltransf|ezh|"
                 r"proteasom|hsp90|mtor|pi3k|akt|mek|erk|raf|egfr|her2|kras|ras|cell cycle|apoptos|bcl|mdm2|p53|"
                 r"transcript|rna pol|cdc|e2f|histone|chromatin|alkylat|platin|intercal|anthracycl", re.I)


def main():
    cc = json.load(open(OUT + "cc_signflip.json"))["drug_records"]; dv = V.dvia_all()
    dm = pq.read_table(DM).to_pandas(); cols = [c for c in dm.columns if any(k in c.lower() for k in ("moa", "target"))]
    name = "drug" if "drug" in dm.columns else dm.columns[0]
    text = {str(r[name]).strip().lower(): " ".join(str(r[c]) for c in cols) for _, r in dm.iterrows()}
    keep, excl_moa, excl_via = [], 0, 0
    for s, r in cc.items():
        if r["plate"] == "plate13":
            continue
        if dv.get(s, 1) >= 0.289:
            excl_via += 1; continue
        t = text.get(r["drug"].strip().lower(), "")
        if not t or KEY.search(t):
            excl_moa += 1; continue
        keep.append(r)
    p = np.array([r["p"] for r in keep]); n = len(p); k5 = int((p < 0.05).sum()); k1 = int((p < 0.01).sum())
    tail = lambda k, q: sum(math.comb(n, i) * q ** i * (1 - q) ** (n - i) for i in range(k, n + 1))
    res = {"rules": __doc__, "metadata_columns": cols, "excluded_viability": excl_via, "excluded_mechanism_or_unknown": excl_moa,
           "wells": n, "compounds": len({r["drug"] for r in keep}), "share_p_lt_005": round(k5 / n, 4), "share_p_lt_001": round(k1 / n, 4),
           "binomial_p_ge_k5_at_5pct": tail(k5, 0.05), "binomial_p_ge_k1_at_1pct": tail(k1, 0.01), "drugs": sorted({r["drug"] for r in keep})}
    # per plate (10-29, reader review: every well of a plate shares the plate's pooled control wells, so the plate,
    # not the well, is the unit): share significant, and among significant wells the share whose mean shift has the
    # plate's majority sign (near 1 if the shared control wells drive the calls)
    sk = "T" if "T" in keep[0] else next(k for k in keep[0] if k not in ("p", "plate", "drug") and isinstance(keep[0][k], (int, float)))
    pp = {}
    for pl in sorted({r["plate"] for r in keep}, key=lambda x: int(x[5:])):
        rs = [r for r in keep if r["plate"] == pl]; sig = [r for r in rs if r["p"] < 0.05]
        pos = sum(r[sk] > 0 for r in sig)
        pp[pl] = dict(wells=len(rs), share_p_lt_005=round(len(sig) / len(rs), 3), majority_sign_share=round(max(pos, len(sig) - pos) / max(len(sig), 1), 3))
    res["per_plate"] = pp; res["shift_key"] = sk
    sh = [v["share_p_lt_005"] for v in pp.values()]
    res["per_plate_share_range"] = [min(sh), float(np.median(sh)), max(sh)]
    print(json.dumps({k: v for k, v in res.items() if k not in ("rules", "drugs")}), flush=True)
    json.dump(res, open(OUT + "cc99.json", "w"), indent=1); print("SUMMARY done", flush=True)


if __name__ == "__main__":
    main()
