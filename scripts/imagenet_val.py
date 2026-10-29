"""ImageNet validation set access + the held-out stimulus pool.

Expects in data/imagenet/:
  val_images.tar.gz        Hugging Face ILSVRC/imagenet-1k, data/val_images.tar.gz
                           (files named ILSVRC2012_val_00000001_n01751748.JPEG)
  LOC_val_solution.csv     Kaggle imagenet-object-localization-challenge (object boxes)
  LOC_synset_mapping.txt   Kaggle, same competition (wnid -> class names, in class order)

The image archive is extracted to data/imagenet/val/ on first use.
"""
import csv
import random
import tarfile
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent / "data" / "imagenet"
VAL_DIR = ROOT / "val"
SEED = 20261019
EXPECTED = 50000

# Candidate pool per stimulus family: (ImageNet class names, images drawn per class).
# Pointed/round and view are NOT decided here: raters assign them blind, and groups
# are formed from their labels. Classes only guarantee enough of each kind exist.
POOL = {
    "cat":   (["tabby", "tiger cat", "Persian cat", "Siamese cat", "Egyptian cat"], 30),
    "car":   (["sports car", "beach wagon", "convertible", "cab", "racer", "minivan",
               "limousine", "jeep", "Model T", "pickup"], 15),
    "dog":   (["basenji", "Doberman", "German shepherd", "Pembroke", "Siberian husky",
               "malamute", "pug", "French bulldog", "bull mastiff", "boxer",
               "Pekinese", "Shih-Tzu"], 10),
    "other_pointed": (["bullet train", "candle", "church", "obelisk", "airliner",
                       "warplane", "mountain tent", "volcano"], 10),
    "other_round":   (["golf ball", "tennis ball", "wall clock", "analog clock", "plate",
                       "orange", "ping-pong ball", "soccer ball", "lampshade"], 9),
    "other_round_face": (["great grey owl"], 30),
}


def val_id(path):
    """ILSVRC2012_val_00000001[_n01751748] -> ILSVRC2012_val_00000001"""
    return "_".join(Path(path).stem.split("_")[:3])


class ImageNetVal(Dataset):
    """Minimal stand-in for torchvision's ImageNet: .samples, .classes, .transform."""

    def __init__(self, transform=None):
        self.transform = transform
        wnids, self.classes = [], []
        for line in (ROOT / "LOC_synset_mapping.txt").read_text().splitlines():
            wnid, names = line.split(" ", 1)
            wnids.append(wnid)
            self.classes.append(tuple(n.strip() for n in names.split(",")))
        to_idx = {w: i for i, w in enumerate(wnids)}
        if not VAL_DIR.exists():
            print("extracting val_images.tar.gz ...", flush=True)
            with tarfile.open(ROOT / "val_images.tar.gz") as tf:
                tf.extractall(VAL_DIR, filter="data")
        # Ground-truth class: the first wnid in the Kaggle solution file.
        label = {}
        with open(ROOT / "LOC_val_solution.csv") as f:
            for row in csv.DictReader(f):
                label[row["ImageId"]] = to_idx[row["PredictionString"].split()[0]]
        files = sorted(p for p in VAL_DIR.rglob("*") if p.suffix.lower() in (".jpeg", ".jpg"))
        self.samples = [(str(p), label[val_id(p)]) for p in files]
        if len(self.samples) != EXPECTED:
            raise RuntimeError(f"expected {EXPECTED} val images, found {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, t = self.samples[i]
        img = Image.open(path).convert("RGB")
        return (self.transform(img) if self.transform else img), t


def load():
    return ImageNetVal()


def class_index(ds):
    """Map every name in ImageNet's class-name tuples to its index."""
    idx = {}
    for i, names in enumerate(ds.classes):
        for n in names:
            idx.setdefault(n, i)
    return idx


def stimulus_pool(ds):
    """Deterministic list of (family, class_name, sample_index). Held out from verification."""
    idx = class_index(ds)
    by_class = {}
    for i, (_, t) in enumerate(ds.samples):
        by_class.setdefault(t, []).append(i)
    rng = random.Random(SEED)
    pool = []
    for family, (names, k) in POOL.items():
        for n in names:
            if n not in idx:
                raise KeyError(f"ImageNet class not found: {n!r}")
            picks = rng.sample(sorted(by_class[idx[n]]), k)
            pool += [(family, n, i) for i in picks]
    return pool


def object_boxes():
    """{val id: list of (x0, y0, x1, y1)} from the Kaggle solution file."""
    out = {}
    with open(ROOT / "LOC_val_solution.csv") as f:
        for row in csv.DictReader(f):
            parts = row["PredictionString"].split()
            out[row["ImageId"]] = [tuple(int(float(v)) for v in parts[k + 1:k + 5])
                                   for k in range(0, len(parts), 5)]
    return out
