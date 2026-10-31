"""Plate-13 response-length ratio with one definition for every group (samewell/notes.md §107, independent review 10-29
item A9; A100). Reads p13strong92.json (response length of each well of a pair). The paper compared plate 13's
partner / original ratio, conditioned on the original well being strong, with min / max of strong plates-6/14 pairs;
the two statistics differ (conditioning on one well being strong regresses its partner toward the mean). Here, for
every group, the first well is the reference well (plate 13: the non-plate-13 well; plates 6/14: the plate-6 well;
other groups: the well on the lower-numbered plate, or the lower sample number on one plate), the pair enters when the
first well's length >= 3, and the statistic is partner length / first-well length. Reported: pairs, compounds,
median and quartiles of the ratio, number with ratio >= 0.5.
Output $WORKSPACE/rt/samewell/p13r99.json"""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import json, collections
import numpy as np
import sw_core as C

OUT = (_WS + "/rt/samewell/")


def main():
    d = C.load(); plate = np.array(d["plate"]); rows = json.load(open(OUT + "p13strong92.json"))["rows"]; res = {"rules": __doc__}
    pn = lambda j: int(str(plate[j])[5:])
    for g in ("plate13", "G614", "diffpos same name", "diffpos salt", "sameplate"):
        rs = [r for r in rows if r["group"] == g]; out = []
        for r in rs:
            a, b = r["a"], r["b"]; amp = dict(zip((a, b), r["amp"]))
            if g == "plate13":
                first = a if pn(a) != 13 else b
            elif g == "G614":
                first = a if pn(a) == 6 else b
            else:
                first = min((a, b), key=lambda j: (pn(j), j))
            other = b if first == a else a
            if amp[first] >= 3:
                out.append((r["cond"].rsplit("_", 1)[0].lower(), amp[other] / amp[first]))
        q = [x[1] for x in out]
        res[g] = dict(pairs=len(out), compounds=len({x[0] for x in out}), median=round(float(np.median(q)), 3) if q else None,
                      q25_q75=[round(float(v), 3) for v in np.quantile(q, [0.25, 0.75])] if q else None, ratio_ge_half=int(sum(v >= 0.5 for v in q)))
        print(g, json.dumps(res[g]), flush=True)
    json.dump(res, open(OUT + "p13r99.json", "w"), indent=1); print("SUMMARY done", flush=True)


if __name__ == "__main__":
    main()
