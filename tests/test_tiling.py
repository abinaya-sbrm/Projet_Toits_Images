import numpy as np

from src.preprocessing.tiling import boxes_to_global, tile_image, tile_positions, untile


def test_positions_1024():
    assert tile_positions(1024, 512, 128) == [0, 384, 512]
    assert tile_positions(1024, 512, 64) == [0, 448, 512]
    assert tile_positions(1024, 512, 0) == [0, 512]


def test_reconstruction_exacte():
    img = np.random.default_rng(1).integers(0, 255, (1024, 1024, 3), dtype=np.uint8)
    tiles, coords = tile_image(img, 512, 128)
    assert len(tiles) == 9
    rec = untile(tiles, coords, img.shape).round().astype(np.uint8)
    assert np.array_equal(rec, img)


def test_masque_max():
    m = np.zeros((1024, 1024), dtype=np.uint8)
    m[500:540, 500:540] = 1  # toit à cheval sur plusieurs tuiles
    tiles, coords = tile_image(m, 512, 128)
    assert np.array_equal(untile(tiles, coords, m.shape, mode="max"), m)


def test_boxes_to_global():
    b = boxes_to_global(np.array([[10, 20, 30, 40]]), (384, 512))
    assert b.tolist() == [[522, 404, 542, 424]]
