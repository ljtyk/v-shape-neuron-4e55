"""Apply the Phase B inclusion rules (phase-b-design.md) and write the final stimulus list. No activations.

Natural images use the mean of the two raters' pointedness; views only where both raters agree.
  T1_pointed     mean pointedness >= 3.5, not marked too small by either rater
  T1_roundface   mean pointedness <= 2.5, both raters' view frontal or three-quarter
  T3_round       mean pointedness <= 1.5, not too small
  T3_boxy        2.5 <= mean pointedness <= 3.5, not too small
  T5_car         both raters agree on the view (frontal = head-on)
Synthetic: every composite (Test 2) and every occlusion set (Test 4) built in round 2.
Writes data/phase_b/final_stimuli.csv.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAB = ROOT / "data" / "phase_b" / "labels"


def main():
    key = {r["image_id"]: r for r in csv.DictReader(open(ROOT / "data" / "phase_b" / "candidates.csv"))}
    A = {r["image_id"]: r for r in csv.DictReader(open(LAB / "phase_b_labels_raterA.csv"))}
    B = {r["image_id"]: r for r in csv.DictReader(open(LAB / "phase_b_labels_raterB.csv"))}
    rows = []
    for iid, k in key.items():
        g = k["group"]
        if g not in ("T1_pointed", "T1_roundface", "T3_round", "T3_boxy", "T5_car"):
            continue
        a, b = A[iid], B[iid]
        p = (float(a["pointedness"]) + float(b["pointedness"])) / 2
        small = a["too_small"] == "1" or b["too_small"] == "1"
        fb = {"frontal", "three-quarter"}
        keep = {
            "T1_pointed": p >= 3.5 and not small,
            "T1_roundface": p <= 2.5 and a["view"] in fb and b["view"] in fb,
            "T3_round": p <= 1.5 and not small,
            "T3_boxy": 2.5 <= p <= 3.5 and not small,
            "T5_car": a["view"] == b["view"],
        }[g]
        if keep:
            rows.append(dict(test=g[:2], group=g, kind="natural", stim_id=iid, path=k["file"],
                             imagenet_class=k["imagenet_class"], pointedness=p,
                             view=a["view"] if a["view"] == b["view"] else "", extra=""))
    for r in csv.DictReader(open(ROOT / "data" / "phase_b" / "round2_manifest_composites.csv")):
        rows.append(dict(test="T2", group="T2_composite", kind="composite", stim_id=r["file"][:-4],
                         path=str(ROOT / "stimuli" / "phase_b" / "composites" / r["file"]), imagenet_class="",
                         pointedness="", view="", extra=f"{r['shape']}|{r['size']}|{r['texture']}|{r['texture_source']}"))
    for r in csv.DictReader(open(ROOT / "data" / "phase_b" / "round2_manifest_occlusions.csv")):
        for v in ("original", "ear", "control"):
            rows.append(dict(test="T4", group=r["group"], kind=f"occlusion_{v}", stim_id=f"{r['image_id']}_{v}",
                             path=str(ROOT / "stimuli" / "phase_b" / "occlusions" / f"{r['image_id']}_{v}.png"),
                             imagenet_class=r["animal"], pointedness="", view="", extra=r["image_id"]))
    with open(ROOT / "data" / "phase_b" / "final_stimuli.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    c = Counter(r["group"] if r["test"] != "T5" else f"T5_{r['view']}" for r in rows)
    for k_, v in sorted(c.items()):
        print(f"  {k_:<22} {v}")
    print(f"total {len(rows)} stimuli")


if __name__ == "__main__":
    main()
