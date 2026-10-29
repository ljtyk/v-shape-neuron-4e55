"""Convert AI-rater JSON Lines (grid-cell boxes) into the labeling page's CSV format.

Grid: columns A-J, rows 0-9 on the model's square; "C2:E4" (inclusive) ->
x0 = 0.2, y0 = 0.2, x1 = 0.5, y1 = 0.5 in normalized coordinates.

Usage: python raters_to_csv.py <dir with verified raterX_batchNN.jsonl>  (the accepted/ folder)
Writes data/phase_b/labels/phase_b_labels_rater{A,B}.csv and reports missing or malformed rows.
"""
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEADER = ["rater", "image_id", "tasks", "main_object", "pointedness", "view", "no_pointed", "too_small",
          "pt_x0", "pt_y0", "pt_x1", "pt_y1", "ears_not_visible",
          "ear1_x0", "ear1_y0", "ear1_x1", "ear1_y1", "ear2_x0", "ear2_y0", "ear2_x1", "ear2_y1",
          "obj_x0", "obj_y0", "obj_x1", "obj_y1"]
CELL = re.compile(r"^\s*([A-Ja-j])\s*([0-9])\s*$")
VIEWS = {"frontal", "three-quarter", "side", "rear"}


def box(s):
    if not s:
        return ["", "", "", ""]
    a, b = s.split(":")
    (c0, r0), (c1, r1) = [CELL.match(x).groups() for x in (a, b)]
    c0, c1 = sorted((ord(c0.upper()) - 65, ord(c1.upper()) - 65))
    r0, r1 = sorted((int(r0), int(r1)))
    return [f"{c0 / 10:.4f}", f"{r0 / 10:.4f}", f"{(c1 + 1) / 10:.4f}", f"{(r1 + 1) / 10:.4f}"]


def main():
    src = Path(sys.argv[1])
    out = ROOT / "data" / "phase_b" / "labels"
    out.mkdir(parents=True, exist_ok=True)
    items = json.loads((ROOT / "labeling_b" / "images.js").read_text().split("=", 1)[1].rstrip().rstrip(";"))
    tasks = {it["id"]: it["tasks"] for it in items}
    for rater in ("A", "B"):
        recs, bad = {}, []
        for f in sorted(src.glob(f"rater{rater}_batch*.jsonl")):
            for ln, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    recs[str(r["id"])] = r
                except Exception as e:  # noqa: BLE001
                    bad.append(f"{f.name}:{ln} {e}")
        rows, problems = [], []
        for iid, t in tasks.items():
            r = recs.get(iid)
            if r is not None and r.get("could_not_open"):
                problems.append(f"could_not_open {iid}")
                continue
            if r is None:
                problems.append(f"missing {iid}")
                continue
            try:
                if "rate" in t:
                    assert int(r["pointedness"]) in range(1, 6), "pointedness"
                    assert r["view"] in VIEWS, "view"
                    assert r.get("no_pointed") or r.get("pointed_box"), "pointed box"
                ears = r.get("ear_boxes") or []
                if "ears" in t:
                    assert r.get("ears_not_visible") or ears, "ears"
                if "object" in t:
                    assert r.get("object_box"), "object box"
                ears = (ears + [None, None])[:2]
                rows.append(["AI-" + rater, iid, "+".join(t), r.get("main_object", ""), r.get("pointedness") or "", r.get("view") or "",
                             int(bool(r.get("no_pointed"))), int(bool(r.get("too_small"))),
                             *box(None if r.get("no_pointed") else r.get("pointed_box")),
                             int(bool(r.get("ears_not_visible"))), *box(ears[0]), *box(ears[1]),
                             *box(r.get("object_box"))])
            except Exception as e:  # noqa: BLE001
                problems.append(f"invalid {iid}: {e}")
        with open(out / f"phase_b_labels_rater{rater}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(HEADER)
            w.writerows(rows)
        print(f"rater {rater}: {len(rows)} valid of {len(tasks)}; {len(problems)} problems; {len(bad)} unparseable lines")
        for p in (problems + bad)[:15]:
            print("   ", p)


if __name__ == "__main__":
    main()
