"""Tuilage avec chevauchement et reconstitution.

Une image H x W est découpée en tuiles tile x tile, avec un chevauchement
`overlap` (pas = tile - overlap). La dernière tuile de chaque ligne/colonne
est recalée sur le bord pour couvrir toute l'image sans padding.
"""
from __future__ import annotations

import numpy as np


def tile_positions(size: int, tile: int, overlap: int) -> list[int]:
    """Positions de départ des tuiles le long d'un axe de longueur `size`."""
    if tile >= size:
        return [0]
    if not 0 <= overlap < tile:
        raise ValueError("overlap doit vérifier 0 <= overlap < tile")
    stride = tile - overlap
    pos = list(range(0, size - tile + 1, stride))
    if pos[-1] != size - tile:  # recaler la dernière tuile sur le bord
        pos.append(size - tile)
    return pos


def tile_image(img: np.ndarray, tile: int = 512, overlap: int = 128):
    """Découpe `img` (H, W[, C]) en tuiles.

    Retourne (tuiles, coords) avec coords[i] = (y0, x0) du coin haut-gauche.
    """
    h, w = img.shape[:2]
    tiles, coords = [], []
    for y in tile_positions(h, tile, overlap):
        for x in tile_positions(w, tile, overlap):
            tiles.append(img[y:y + tile, x:x + tile].copy())
            coords.append((y, x))
    return tiles, coords


def untile(tiles, coords, shape, mode: str = "mean") -> np.ndarray:
    """Reconstitue une image/carte de forme `shape` à partir des tuiles.

    mode="mean" : moyenne dans les zones de chevauchement (cartes de proba, images)
    mode="max"  : maximum (masques binaires : un toit vu dans une tuile est gardé)
    """
    acc = np.zeros(shape, dtype=np.float64)
    if mode == "mean":
        cnt = np.zeros(shape[:2], dtype=np.float64)
    for t, (y, x) in zip(tiles, coords):
        th, tw = t.shape[:2]
        if mode == "mean":
            acc[y:y + th, x:x + tw] += t
            cnt[y:y + th, x:x + tw] += 1
        elif mode == "max":
            acc[y:y + th, x:x + tw] = np.maximum(acc[y:y + th, x:x + tw], t)
        else:
            raise ValueError(f"mode inconnu : {mode}")
    if mode == "mean":
        cnt = cnt[..., None] if acc.ndim == 3 else cnt
        acc = acc / np.maximum(cnt, 1)
    return acc


def boxes_to_global(boxes: np.ndarray, coord) -> np.ndarray:
    """Passe des boîtes [x1, y1, x2, y2] d'une tuile au repère de l'image entière.

    Étape préalable à la fusion globale (NMS ou WBF) des détections en bordure.
    """
    y0, x0 = coord
    out = np.asarray(boxes, dtype=np.float64).copy()
    out[:, [0, 2]] += x0
    out[:, [1, 3]] += y0
    return out
