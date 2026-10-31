"""Label check without a position-sharing comparison group (samewell/notes.md §92, recheck; A100). Reads
p13strong92.json. Review points: plates 6/14 share positions, which the partner rank also detects, so they are not a fair
comparison for pairs at different positions; pairs are not independent (several doses of one compound); a response
length >= 3 can also come from a well shift. Reported per group (G614, same-name pairs at different positions, salt-name
pairs on different plates, one-plate pairs, plate 13):
  - strong pairs (length >= 3 in either well), their compounds, pairs and compounds recognised (top 10% both ways);
  - for pairs whose first well (for plate 13 the original, non-plate-13 well) has length >= 3: the partner's length,
    and how many partners have length >= 2.7 (the largest plate-13 partner length in this set);
  - the strong plate-13 pairs and recognised salt pairs, by compound.
Output $WORKSPACE/rt/samewell/p13len92.json. Usage: p13len92.py"""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import json, collections
import numpy as np

OUT = (_WS + "/rt/samewell/")


def main():
    rows = json.load(open(OUT + "p13strong92.json"))["rows"]; res = {"rules": __doc__}
    comp = lambda r: r["cond"].rsplit("_", 1)[0].lower()
    for g in ("G614", "diffpos same name", "diffpos salt", "sameplate", "plate13"):
        rs = [r for r in rows if r["group"] == g and max(r["amp"]) >= 3]; ok = [r for r in rs if min(r["pct"]) >= 0.9]
        e = dict(strong_pairs=len(rs), strong_compounds=len({comp(r) for r in rs}), recognised_pairs=len(ok), recognised_compounds=len({comp(r) for r in ok}),
                 compounds=sorted(collections.Counter(comp(r) for r in rs).items()), recognised=sorted(collections.Counter(comp(r) for r in ok).items()))
        # partner length when one well is strong: for plate 13 the original well is the first one (a); otherwise both orders
        part = []
        for r in [r for r in rows if r["group"] == g]:
            orders = [(0, 1)] if g == "plate13" else [(0, 1), (1, 0)]
            for x, y in orders:
                if r["amp"][x] >= 3:
                    part.append(r["amp"][y])
        e["partner_length_when_one_well_strong"] = dict(n=len(part), median=round(float(np.median(part)), 2) if part else None,
                                                        min=round(float(min(part)), 2) if part else None, n_ge_2_7=int(sum(p >= 2.7 for p in part)))
        # the same count in both directions for every group, plate 13 included (one unit for all groups)
        both = []
        for r in [r for r in rows if r["group"] == g]:
            for x, y in ((0, 1), (1, 0)):
                if r["amp"][x] >= 3:
                    both.append(r["amp"][y])
        e["partner_length_both_directions"] = dict(n=len(both), median=round(float(np.median(both)), 2) if both else None, n_ge_2_7=int(sum(v >= 2.7 for v in both)))
        # by pairs and with the stated threshold (review 10-28: the 2.7 cut above was chosen after seeing plate 13)
        e["strong_pairs_both_wells_ge_3"] = int(sum(min(r["amp"]) >= 3 for r in rs))
        res[g] = e; print(g, json.dumps(e), flush=True)
    json.dump(res, open(OUT + "p13len92.json", "w"), indent=1); print("SUMMARY done", flush=True)


if __name__ == "__main__":
    main()
