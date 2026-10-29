"""Verify one AI-rater batch before accepting it.

Checks:
  1. Every image on the batch list has exactly one line; every line is valid.
  2. Every annotated (not could_not_open) image was actually viewed: the agent's real transcript
     (~/.claude/projects/<proj>/<session>/subagents/agent-<id>.jsonl) has a Read of that image
     whose result is an image, not an error or "media removed" placeholder.
  3. main_object names match the hidden class key (keyword match) for >= 75% of seen images.
     An agent that never saw the image cannot name it except by chance.
Accepted batches are copied to accepted/ ; rejected ones are moved to rejected/.

Usage: python verify_rater_batch.py <raters_dir> <rater A|B> <batch NN> <agent id>
"""
import csv
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUBAGENTS = Path.home() / ".claude" / "projects" / "C--Users-ljtyk-PycharmProjects-LLMclean" /     "4a1e20d5-a65a-4469-af91-17e2cf533f3b" / "subagents"
MATCH_MIN = 0.75
VIEWS = {"frontal", "three-quarter", "side", "rear"}
CAR = ["car", "van", "truck", "vehicle", "taxi", "cab", "ambulance", "bus", "jeep", "pickup", "limo",
       "wagon", "convertible", "racer", "race", "police", "suv", "sedan", "coupe", "roadster", "hatchback",
       "minivan", "automobile"]
DOG = ["dog", "puppy", "pup", "hound", "spaniel", "pomeranian", "pekingese", "pekinese", "shih", "basset",
       "bernard", "clumber"]
KEYWORDS = {
    "obelisk": ["obelisk", "monument", "tower", "pillar", "column", "spire", "needle", "memorial"],
    "bell cote": ["bell", "tower", "church", "steeple", "belfry", "spire", "chapel", "building", "roof"],
    "church": ["church", "cathedral", "chapel", "steeple", "spire", "tower", "building", "basilica", "abbey"],
    "candle": ["candle", "flame", "candles"],
    "volcano": ["volcano", "mountain", "crater", "peak", "eruption", "lava", "smoke"],
    "mountain tent": ["tent"],
    "Persian cat": ["cat", "kitten"], "tabby": ["cat", "kitten"],
    "Pekinese": DOG, "Japanese spaniel": DOG, "Shih-Tzu": DOG, "Pomeranian": DOG,
    "basset": DOG, "clumber": DOG, "Saint Bernard": DOG,
    "red fox": ["fox"], "Arctic fox": ["fox"],
    "golf ball": ["ball", "golf"], "tennis ball": ["ball", "tennis"], "ping-pong ball": ["ball", "ping", "pong", "table tennis"],
    "wall clock": ["clock"], "analog clock": ["clock", "watch"],
    "plate": ["plate", "dish", "food", "meal", "bowl"], "orange": ["orange", "fruit", "citrus", "tangerine", "mandarin"],
    "carton": ["box", "carton", "package", "container", "packaging"], "crate": ["crate", "box", "case"],
    "chest": ["chest", "trunk", "box", "cabinet", "dresser"], "safe": ["safe", "vault", "box", "lock"],
    "washer": ["wash", "washer", "laundry", "appliance", "machine"],
    "monitor": ["monitor", "screen", "computer", "display", "tv", "television", "desktop", "pc"],
    "television": ["tv", "television", "screen", "monitor", "display", "set"],
    "refrigerator": ["fridge", "refrigerator", "freezer", "appliance"],
    "microwave": ["microwave", "oven", "appliance"], "dishwasher": ["dishwasher", "appliance", "kitchen"],
    "wardrobe": ["wardrobe", "cabinet", "closet", "armoire", "cupboard", "furniture", "dresser"],
    "file": ["cabinet", "file", "drawer", "filing", "furniture"],
    "stone wall": ["wall", "stone", "rock", "brick"],
}
for c in ["sports car", "beach wagon", "convertible", "cab", "racer", "minivan", "limousine", "jeep",
          "Model T", "pickup", "police van", "ambulance", "minibus"]:
    KEYWORDS[c] = CAR


def main():
    rdir, rater, batch, agent = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
    rnd = sys.argv[5] if len(sys.argv) > 5 else ""          # "" for round 1, "_r2" for round 2
    transcript = SUBAGENTS / f"agent-{agent}.jsonl"
    out = rdir / f"out{rnd}" / f"rater{rater}_batch{batch}.jsonl"
    want = [l.split("\t") for l in (rdir / f"batches{rnd}" / f"batch_{batch}.txt").read_text().splitlines() if l.strip()]
    tasks = {i: t.split("+") for i, t in want}
    key = {r["image_id"]: r["imagenet_class"] for r in csv.DictReader(open(ROOT / "data" / "phase_b" / "candidates.csv"))}
    r2key = ROOT / "data" / "phase_b" / "round2_review_key.csv"
    if rnd and r2key.exists():
        key.update({r["review_id"]: r["imagenet_class"] for r in csv.DictReader(open(r2key))})

    problems, recs = [], {}
    if out.exists():
        for ln, line in enumerate(out.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    r = json.loads(line)
                    if str(r["id"]) in recs:
                        problems.append(f"duplicate {r['id']}")
                    recs[str(r["id"])] = r
                except Exception as e:  # noqa: BLE001
                    problems.append(f"line {ln}: {e}")
    else:
        problems.append("no output file")
    missing = [i for i in tasks if i not in recs]
    extra = [i for i in recs if i not in tasks]
    unopened = [i for i, r in recs.items() if r.get("could_not_open")]
    seen = {i: r for i, r in recs.items() if not r.get("could_not_open") and i in tasks}
    for i, r in seen.items():
        t = tasks[i]
        try:
            assert r.get("main_object"), "main_object"
            if "rate" in t:
                assert int(r["pointedness"]) in range(1, 6), "pointedness"
                assert r["view"] in VIEWS, "view"
                assert r.get("no_pointed") or r.get("pointed_box"), "pointed box"
            if "ears" in t:
                assert r.get("ears_not_visible") or r.get("ear_boxes"), "ears"
            if "object" in t:
                assert r.get("object_box"), "object box"
            if t[0] == "earzoom":
                assert r.get("ears_not_visible") or r.get("ear_boxes"), "earzoom"
            if t[0] == "body":
                assert r.get("no_body") or r.get("body_box"), "body"
            if t[0].startswith("texbox"):
                assert r.get("no_clean_area") or r.get("texture_box"), "texbox"
        except Exception as e:  # noqa: BLE001
            problems.append(f"invalid {i}: {e}")
    viewed, blocked, pending = set(), 0, {}
    if transcript.exists():
        for line in transcript.read_text(encoding="utf-8").splitlines():
            try:
                content = (json.loads(line).get("message") or {}).get("content")
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(content, list):
                continue
            for c in content:
                if c.get("type") == "tool_use" and c.get("name") == "Read":
                    m = re.search(r"(\d{6})\.png", c.get("input", {}).get("file_path", ""))
                    if m:
                        pending[c["id"]] = m.group(1)
                if c.get("type") == "tool_result" and c.get("tool_use_id") in pending:
                    body = json.dumps(c.get("content"))
                    if "media removed" in body:
                        blocked += 1
                    elif '"image"' in body:
                        viewed.add(pending[c["tool_use_id"]])
    unviewed = [i for i in seen if i not in viewed]
    hits = [any(k in str(r.get("main_object", "")).lower() for k in KEYWORDS[key[i]]) for i, r in seen.items()]
    match = sum(hits) / max(len(hits), 1)
    mism = [(i, key[i], r.get("main_object")) for (i, r), h in zip(seen.items(), hits) if not h]

    ok = (transcript.exists() and not missing and not extra and not problems and not unviewed
          and match >= MATCH_MIN)
    verdict = "ACCEPT" if ok else "REJECT"
    print(f"{verdict} rater {rater} batch {batch}: lines {len(recs)}/{len(tasks)}, seen {len(seen)}, "
          f"could_not_open {len(unopened)}, annotated-but-not-viewed {len(unviewed)}, blocked reads {blocked}, object match {match:.0%}, "
          f"missing {len(missing)}, extra {len(extra)}, problems {len(problems)}")
    for p in problems[:5]:
        print("   ", p)
    if mism:
        print("    mismatches:", "; ".join(f"{c} -> {o!r}" for _, c, o in mism[:8]))
    if out.exists():
        dest = rdir / (("accepted" if ok else "rejected") + rnd)
        dest.mkdir(exist_ok=True)
        shutil.move(str(out), dest / out.name)


if __name__ == "__main__":
    main()
