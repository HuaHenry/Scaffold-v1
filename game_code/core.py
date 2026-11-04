"""Deterministic grid physics. Default research grid: 28x22; arbitrary map sizes supported.

Foreground carries load and collision; background walls/windows/decor do not.
This is a discrete sandbox rule set, not a realistic structural simulator.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np

W, H = 28, 22
(EMPTY, GROUND, STONE, WOOD, LEAF, SAND, WATER, CLOUD, DIRT, GRASS,
 BRICK, GLASS, ROOF, DOOR, WALL, TORCH, LADDER, FLOWER, OPEN_DOOR,
 DOOR_TOP, OPEN_DOOR_TOP) = range(21)
LOAD = {GROUND, STONE, WOOD, DIRT, GRASS, BRICK, GLASS, ROOF}
STRUCT = LOAD - {GROUND}
LOOSE = {LEAF, SAND, WATER}
MOVABLE = set(range(21)) - {EMPTY, GROUND, CLOUD}
DOORS = {DOOR, DOOR_TOP, OPEN_DOOR, OPEN_DOOR_TOP}
BACK = {WALL, GLASS, TORCH, LADDER, FLOWER}
PANEL = {WALL, GLASS}
NEIGHBOURS = ((0, -1), (-1, 0), (1, 0), (0, 1))
LOAD |= DOORS
SPAN = {STONE: 2, WOOD: 4, DIRT: 0, GRASS: 0, BRICK: 6, GLASS: 2, ROOF: 6}
NAME = {STONE: 'stone', WOOD: 'wood', LEAF: 'leaf', SAND: 'sand', WATER: 'water',
        DIRT: 'dirt', GRASS: 'grass', BRICK: 'brick', GLASS: 'window', ROOF: 'roof',
        DOOR: 'door', WALL: 'wall', TORCH: 'torch', LADDER: 'ladder', FLOWER: 'flower'}


class SeededRNG:
    def __init__(self, seed=1):
        self.state = seed & 0xffffffff

    def random(self):
        self.state = (1664525 * self.state + 1013904223) & 0xffffffff
        return self.state / 4294967296

    def shuffle(self, values):
        for i in range(len(values) - 1, 0, -1):
            j = int(self.random() * (i + 1))
            values[i], values[j] = values[j], values[i]


def zeros(width=W, height=H):
    grid = np.zeros((height, width), dtype=np.uint8)
    grid[-1, :] = GROUND
    return grid


@dataclass
class Action:
    type: str = 'noop'
    x: int = 0
    y: int = 0
    material: int = STONE
    layer: str = 'front'

    def as_dict(self):
        return asdict(self)


class FormWorld:
    def __init__(self, seed=0, grid=None, background=None):
        self.grid = zeros() if grid is None else np.asarray(grid, dtype=np.uint8).copy()
        if self.grid.ndim != 2 or min(self.grid.shape) < 2 or self.grid.max() > 20:
            raise ValueError('Invalid foreground grid')
        self.h, self.w = self.grid.shape
        self.background = np.zeros_like(self.grid) if background is None else np.asarray(background, dtype=np.uint8).copy()
        if self.background.shape != self.grid.shape:
            raise ValueError('Layer dimensions must match')
        self.rng = SeededRNG(seed)
        self.steps = 0

    def inside(self, x, y):
        return 0 <= x < self.w and 0 <= y < self.h

    def supported(self):
        g = self.grid
        cost = np.full(g.shape, -1, dtype=np.int16)
        q = []
        for y in range(self.h):
            for x in range(self.w):
                if g[y, x] == GROUND:
                    cost[y, x] = 0
                    q.append((x, y))
        for x, y in q:
            for dx, dy, inc in ((0, -1, 0), (-1, 0, 1), (1, 0, 1)):
                nx, ny = x + dx, y + dy
                if not self.inside(nx, ny) or int(g[ny, nx]) not in LOAD:
                    continue
                nc = int(cost[y, x]) + inc
                if nc > SPAN.get(int(g[ny, nx]), 0) or 0 <= cost[ny, nx] <= nc:
                    continue
                cost[ny, nx] = nc
                q.append((nx, ny))
        return cost >= 0

    def attached_leaves(self):
        g = self.grid
        q = [(x, y, 0) for y in range(self.h) for x in range(self.w) if g[y, x] == WOOD]
        seen = set()
        for x, y, distance in q:
            if distance >= 4:
                continue
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if self.inside(nx, ny) and g[ny, nx] == LEAF and (nx, ny) not in seen:
                    seen.add((nx, ny))
                    q.append((nx, ny, distance + 1))
        return seen

    def _neighbour_has_wood(self, x, y):
        return (x, y) in self.attached_leaves()

    def water_target(self, x, y):
        g = self.grid
        for dx in (0, -1, 1):
            if self.inside(x + dx, y + 1) and g[y + 1, x + dx] == EMPTY:
                return x + dx, y + 1
        candidates = []
        for direction in (-1, 1):
            for distance in range(1, self.w):
                nx = x + direction * distance
                if not self.inside(nx, y) or g[y, nx] not in (EMPTY, WATER):
                    break
                if self.inside(nx, y + 1) and g[y + 1, nx] == EMPTY:
                    candidates.append((distance, direction))
                    break
        candidates.sort()
        if candidates and g[y, x + candidates[0][1]] == EMPTY:
            return x + candidates[0][1], y
        return None

    def tick(self):
        g, ok, leaves = self.grid, self.supported(), self.attached_leaves()
        xs = list(range(self.w))
        self.rng.shuffle(xs)
        touched, moves = set(), []

        def move(x, y, nx, ny):
            v, swapped = int(g[y, x]), int(g[ny, nx])
            g[ny, nx], g[y, x] = v, swapped
            touched.update(((x, y), (nx, ny)))
            moves.append({'from': (x, y), 'to': (nx, ny), 'v': v, 'swapped': swapped})

        for y in range(self.h - 2, -1, -1):
            for x in xs:
                if int(g[y, x]) in STRUCT and not ok[y, x] and (x, y) not in touched and int(g[y + 1, x]) in LOOSE | {EMPTY}:
                    move(x, y, x, y + 1)
        for y in range(self.h - 2, 0, -1):
            for x in xs:
                if g[y, x] not in (DOOR, OPEN_DOOR) or (x, y) in touched or int(g[y + 1, x]) not in LOOSE | {EMPTY}:
                    continue
                top, bottom, displaced = int(g[y - 1, x]), int(g[y, x]), int(g[y + 1, x])
                if top not in (DOOR_TOP, OPEN_DOOR_TOP):
                    continue
                g[y + 1, x], g[y, x], g[y - 1, x] = bottom, top, displaced
                touched.update((x, row) for row in (y - 1, y, y + 1))
                moves.extend([{'from': (x, y), 'to': (x, y + 1), 'v': bottom, 'swapped': 0},
                              {'from': (x, y - 1), 'to': (x, y), 'v': top, 'swapped': 0}])
                if displaced:
                    moves.append({'from': (x, y + 1), 'to': (x, y - 1), 'v': displaced, 'swapped': 0})
        for y in range(self.h - 2, -1, -1):
            for x in xs:
                if g[y, x] == LEAF and (x, y) not in leaves and (x, y) not in touched and g[y + 1, x] in (EMPTY, WATER):
                    move(x, y, x, y + 1)
        for y in range(self.h - 2, -1, -1):
            for x in xs:
                if g[y, x] != SAND or (x, y) in touched:
                    continue
                directions = (0, -1, 1) if self.rng.random() < .5 else (0, 1, -1)
                for dx in directions:
                    if self.inside(x + dx, y + 1) and g[y + 1, x + dx] in (EMPTY, WATER):
                        move(x, y, x + dx, y + 1)
                        break
        for y in range(self.h - 2, -1, -1):
            for x in xs:
                if g[y, x] == WATER and (x, y) not in touched:
                    target = self.water_target(x, y)
                    if target:
                        move(x, y, *target)
        return bool(moves), moves

    def door_base(self, x, y):
        if not self.inside(x, y) or int(self.grid[y, x]) not in DOORS:
            return None
        return x, y + 1 if self.grid[y, x] in (DOOR_TOP, OPEN_DOOR_TOP) else y

    def _anchored_by_block(self, x, y, m):
        g = self.grid
        if m == FLOWER:
            return self.inside(x, y + 1) and int(g[y + 1, x]) in (GRASS, DIRT)
        return any(self.inside(x + dx, y + dy) and int(g[y + dy, x + dx]) in LOAD
                   for dx, dy in ((0, 0),) + NEIGHBOURS)

    @staticmethod
    def _attaches(src, dst, vertical):
        return (src in PANEL and (dst in PANEL or dst in (TORCH, LADDER))) or (src == LADDER and dst == LADDER and vertical)

    def prune_background(self):
        """Remove background items that are not anchored to a solid block. Returns count removed."""
        b = self.background
        live, q = set(), []
        for y in range(self.h):
            for x in range(self.w):
                if int(b[y, x]) in BACK and self._anchored_by_block(x, y, int(b[y, x])):
                    live.add((x, y))
                    q.append((x, y))
        for x, y in q:
            for dx, dy in NEIGHBOURS:
                nx, ny = x + dx, y + dy
                if self.inside(nx, ny) and (nx, ny) not in live and self._attaches(int(b[y, x]), int(b[ny, nx]), dx == 0):
                    live.add((nx, ny))
                    q.append((nx, ny))
        removed = 0
        for y in range(self.h):
            for x in range(self.w):
                if b[y, x] and (x, y) not in live:
                    b[y, x] = EMPTY
                    removed += 1
        return removed

    def can_attach(self, x, y, m):
        if self._anchored_by_block(x, y, m):
            return True
        if m == FLOWER:
            return False
        b = self.background
        return any(self.inside(x + dx, y + dy) and self._attaches(int(b[y + dy, x + dx]), m, dx == 0)
                   for dx, dy in NEIGHBOURS)

    def apply_action(self, a):
        g, x, y, m = self.grid, a.x, a.y, a.material
        if a.type == 'noop':
            return True
        if not self.inside(x, y) or y == self.h - 1:
            return False
        if a.layer == 'back':
            b = self.background
            if a.type == 'remove':
                if not b[y, x]:
                    return False
                b[y, x] = EMPTY
                self.prune_background()
                return True
            if a.type == 'place' and m in BACK and b[y, x] != m:
                old = b[y, x]
                b[y, x] = EMPTY
                if not self.can_attach(x, y, m):
                    b[y, x] = old
                    return False
                b[y, x] = m
                self.prune_background()
                return True
            return False
        if a.type == 'interact':
            base = self.door_base(x, y)
            if not base:
                return False
            dx, dy = base
            opened = g[dy, dx] == DOOR
            g[dy, dx], g[dy - 1, dx] = (OPEN_DOOR, OPEN_DOOR_TOP) if opened else (DOOR, DOOR_TOP)
            return True
        if a.type == 'remove':
            if int(g[y, x]) not in MOVABLE:
                return False
            base = self.door_base(x, y)
            if base:
                g[base[1], x] = g[base[1] - 1, x] = EMPTY
            else:
                g[y, x] = EMPTY
            self.prune_background()
            return True
        if a.type == 'place':
            if m not in MOVABLE or g[y, x] != EMPTY:
                return False
            if m == DOOR:
                if y < 1 or g[y - 1, x] != EMPTY or int(g[y + 1, x]) not in LOAD:
                    return False
                g[y, x], g[y - 1, x] = DOOR, DOOR_TOP
            elif m not in (OPEN_DOOR, DOOR_TOP, OPEN_DOOR_TOP, WALL, TORCH, LADDER, FLOWER):
                g[y, x] = m
            else:
                return False
            return True
        if a.type == 'drop' and m in STRUCT | LOOSE:
            for yy in range(self.h - 1):
                if g[yy, x] == EMPTY:
                    g[yy, x] = m
                    return True
        return False

    def step(self, action=None):
        if action is not None:
            self.apply_action(action)
        for _ in range(self.h * 8):
            changed, _ = self.tick()
            if not changed:
                break
        self.prune_background()
        self.steps += 1
        return self.grid.copy()

    def copy_grid(self):
        return self.grid.copy()


def check_violations(grid, background=None):
    """Count rule violations in a state that is supposed to be at rest.

    Pass `background` to also count floating walls/windows/torches/ladders/flowers.
    """
    world = FormWorld(grid=grid, background=background)
    ok, g, leaves = world.supported(), world.grid, world.attached_leaves()
    result = {'floating_struct': 0, 'struct_on_loose': 0, 'floating_leaf': 0, 'floating_sand': 0}
    for y in range(world.h - 1):
        for x in range(world.w):
            if int(g[y, x]) in STRUCT and not ok[y, x]:
                result['floating_struct'] += 1
                result['struct_on_loose'] += int(int(g[y + 1, x]) in LOOSE)
            if g[y, x] == LEAF and g[y + 1, x] == EMPTY and (x, y) not in leaves:
                result['floating_leaf'] += 1
            if g[y, x] == SAND and g[y + 1, x] == EMPTY:
                result['floating_sand'] += 1
    if background is not None:
        result['floating_back'] = world.prune_background()
    return result
