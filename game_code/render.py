"""Exact flat RGB renderer, separate from the game's textured artistic renderer.

Only render() / drawExact() frames are palette-decodable. Artistic screenshots
are not exact state encodings. Background layers must be recorded separately.
"""

from __future__ import annotations

import numpy as np

from core import (W, H, EMPTY, GROUND, STONE, WOOD, LEAF, BRICK, ROOF, DIRT, GRASS, DOOR,
                  WALL, GLASS, TORCH, LADDER, FLOWER, SAND, WATER, CLOUD, FormWorld, Action)

HEX_COLORS = ['#d9ebf1', '#454851', '#838a90', '#b78143', '#418b48', '#d9c18a', '#529bc0', '#f4fafb',
              '#896044', '#68a849', '#bf6750', '#8bd6df', '#934b38', '#a4723e', '#806244', '#f2b04d', '#c89c58',
              '#edbe65', '#69422b', '#a16a35', '#543724']
BASE = {i: tuple(int(color[j:j + 2], 16) for j in (1, 3, 5)) for i, color in enumerate(HEX_COLORS)}
EMPTY_RGB = BASE[EMPTY]


def make_lut() -> np.ndarray:
    return np.array([BASE[i] for i in range(len(BASE))], dtype=np.uint8)


LUT = make_lut()


def render(grid: np.ndarray, cell: int = 2) -> np.ndarray:
    """无噪声的精确 RGB 渲染，作为训练数据与解析器的 LUT。"""
    rgb = LUT[grid]
    return np.repeat(np.repeat(rgb, cell, axis=-3), cell, axis=-2)


def compose(foreground: np.ndarray, background: np.ndarray) -> np.ndarray:
    """What a viewer sees: the foreground block if present, otherwise the background item."""
    return np.where(foreground != 0, foreground, background).astype(np.uint8)


def render_layers(foreground: np.ndarray, background: np.ndarray, cell: int = 2) -> np.ndarray:
    """Exact RGB frame of both layers. Works on a single [H,W] state or a stack [...,H,W]."""
    return render(compose(foreground, background), cell)


def parse(image: np.ndarray, cell: int | None = None) -> np.ndarray:
    """Decode a flat frame. Pass cell explicitly for non-default grid sizes."""
    cell = image.shape[0] // H if cell is None else cell
    if cell < 1 or image.shape[0] % cell or image.shape[1] % cell:
        raise ValueError('Invalid image or cell dimensions')
    centers = image[cell // 2::cell, cell // 2::cell, :3].astype(np.int32)
    colors = LUT.astype(np.int32)
    dist = ((centers[:, :, None, :] - colors[None, None, :, :]) ** 2).sum(-1)
    return dist.argmin(-1).astype(np.uint8)


def verify_render_parser(seed: int = 0) -> float:
    """Round-trip random mixed worlds (all placeable materials, both layers) through render/parse."""
    world = FormWorld(seed)
    rng = np.random.default_rng(seed)
    front = [STONE, WOOD, LEAF, SAND, WATER, BRICK, ROOF, DIRT, GRASS, DOOR]
    back = [WALL, GLASS, TORCH, LADDER, FLOWER]
    correct = total = 0
    for _ in range(30):
        for _ in range(20):
            x = int(rng.integers(0, W))
            y = int(rng.integers(0, H - 1))
            world.step(Action("place", x, y, int(rng.choice(front))))
            world.step(Action("place", x, y, int(rng.choice(back)), "back"))
        seen = compose(world.grid, world.background)
        pred = parse(render(seen))
        correct += int((pred == seen).sum())
        total += pred.size
    return correct / total
