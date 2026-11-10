"""Premiers tests de tuilage (Séance 1).

Usage :
    python scripts/test_tiling.py                       # image synthétique 1024x1024
    python scripts/test_tiling.py --image data/raw/xxx.tif --tile 512 --overlap 128

Vérifie que la reconstitution redonne exactement l'image d'origine et
enregistre une figure de la grille de tuiles dans results/figures/.
"""
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocessing.tiling import tile_image, untile  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, default=None)
    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--overlap", type=int, default=128)
    a = p.parse_args()

    if a.image:
        img = cv2.imread(a.image, cv2.IMREAD_COLOR)
        if img is None:
            sys.exit(f"Impossible de lire {a.image}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        name = Path(a.image).stem
    else:
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, (1024, 1024, 3), dtype=np.uint8)
        name = "synthetique"

    tiles, coords = tile_image(img, a.tile, a.overlap)
    rec = untile(tiles, coords, img.shape, mode="mean").round().astype(np.uint8)
    ok = np.array_equal(rec, img)
    print(f"Image {img.shape}, tuile {a.tile}, chevauchement {a.overlap} px")
    print(f"{len(tiles)} tuiles, coins (y, x) : {coords}")
    print(f"Reconstitution identique à l'original : {ok}")

    out = ROOT / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img)
    colors = plt.cm.tab10(np.linspace(0, 1, len(coords)))
    for (y, x), c in zip(coords, colors):
        ax.add_patch(patches.Rectangle((x, y), a.tile, a.tile, fill=False,
                                       edgecolor=c, linewidth=2))
    ax.set_title(f"{name} : {len(tiles)} tuiles {a.tile}px, chevauchement {a.overlap}px")
    ax.axis("off")
    f = out / f"tuilage_{name}_{a.tile}_{a.overlap}.png"
    fig.savefig(f, dpi=100, bbox_inches="tight")
    print(f"Figure : {f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
