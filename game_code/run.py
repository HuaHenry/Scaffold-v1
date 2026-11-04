"""FormWorld-2D 命令行入口。

  python run.py --selftest                 物理与渲染自检
  python run.py --record --png            导出一段形成/坍塌故事板
  python run.py --data --quick            生成四种训练条件 + FormBench（小规模）
  python run.py --data                    生成完整规模数据（256 episodes / 128 per split）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from core import (W, H, STONE, WOOD, LEAF, SAND, WATER, EMPTY,
                  Action, FormWorld, check_violations)
from render import render, parse, verify_render_parser


def physics_selftest() -> dict:
    results = {}

    # 1. 空中放一块石头：应当落到地面上方。
    w = FormWorld(0)
    w.step(Action("place", 10, 5, STONE))
    results["floating_stone_falls"] = bool(w.grid[H - 2, 10] == STONE)

    # 2. 从地面往上搭柱子：应当保持站立。
    w = FormWorld(0)
    for y in range(H - 2, H - 7, -1):
        w.step(Action("place", 10, y, STONE))
    results["supported_column_stands"] = bool(
        all(w.grid[y, 10] == STONE for y in range(H - 2, H - 7, -1)))

    # 3. 木的悬挑上限应大于石。
    def cantilever(mat: int, length: int) -> int:
        w = FormWorld(0)
        for y in range(H - 2, H - 5, -1):
            w.step(Action("place", 5, y, STONE))
        for x in range(6, 6 + length):
            w.step(Action("place", x, H - 4, mat))
        return int(np.count_nonzero(w.grid[H - 4, 6:6 + length] == mat))

    stone_long = cantilever(STONE, 6)
    wood_long = cantilever(WOOD, 6)
    results["stone_cantilever"] = stone_long
    results["wood_cantilever"] = wood_long
    results["wood_spans_farther"] = bool(wood_long > stone_long)

    # 4. 砂下落并形成堆积：所有砂粒都应落在地面附近，且下方不为空。
    w = FormWorld(0)
    for _ in range(12):
        w.step(Action("drop", 12, 0, SAND))
    sand_rows = np.where(w.grid == SAND)[0]
    stable = all(w.grid[y + 1, x] != EMPTY
                 for y, x in zip(*np.where(w.grid == SAND)) if y < H - 1)
    results["sand_count"] = int(np.count_nonzero(w.grid == SAND))
    results["sand_lowest_row"] = int(sand_rows.max()) if sand_rows.size else -1
    results["sand_settles"] = bool(
        np.count_nonzero(w.grid == SAND) >= 10
        and (sand_rows.size == 0 or sand_rows.min() >= H - 8)
        and stable)

    # 5. 叶无木依附时会下落。
    w = FormWorld(0)
    w.step(Action("place", 8, 6, LEAF))
    results["unsupported_leaf_falls"] = bool(np.count_nonzero(w.grid == LEAF) == 1
                                              and w.grid[6, 8] == EMPTY)

    # 6. Every loose-material combination: a structure must never rest on it.
    combos = {}
    for label, stack in (
        ("wood_on_leaf",  [(WOOD, H-2), (LEAF, H-3),  (WOOD, H-4)]),
        ("stone_on_sand", [(WOOD, H-2), (SAND, H-3),  (STONE, H-4)]),
        ("wood_on_water", [(WOOD, H-2), (WATER, H-3), (WOOD, H-4)]),
        ("stone_on_leaf", [(STONE, H-2), (LEAF, H-3), (STONE, H-4)]),
    ):
        w = FormWorld(0)
        for m, y in stack:
            w.step(Action("place", 10, y, m))
        combos[label] = int(sum(check_violations(w.grid).values()))
    results["loose_combinations"] = combos
    results["no_struct_on_loose"] = all(v == 0 for v in combos.values())

    # 7. A long mixed rollout must converge with zero violations.
    w = FormWorld(3)
    r = np.random.default_rng(3)
    for _ in range(60):
        w.step(Action("place", int(r.integers(0, W)), int(r.integers(0, H-1)),
                      int(r.choice([STONE, WOOD, LEAF, SAND, WATER]))))
    results["mixed_rollout_violations"] = int(
        sum(check_violations(w.grid).values()))

    # 8. tick() advances one row per call, so falling is animatable.
    w = FormWorld(0)
    w.apply_action(Action("place", 10, 5, STONE))
    ticks = 0
    while ticks < 4 * H:
        changed, _ = w.tick()
        if not changed:
            break
        ticks += 1
    results["animated_fall_ticks"] = ticks
    results["tick_is_animated"] = ticks > 5

    # 9. Violation counter is zero on a valid structure.
    w = FormWorld(0)
    for y in range(H - 2, H - 6, -1):
        w.step(Action("place", 10, y, STONE))
    results["no_violation_on_valid_structure"] = (
        sum(check_violations(w.grid).values()) == 0)

    return results


def demo_bridge() -> list[Action]:
    """一段合法的形成过程：立柱 → 顶梁。"""
    acts: list[Action] = []
    top = H - 6
    for x in (7, 12):
        for y in range(H - 2, top, -1):
            acts.append(Action("place", x, y, STONE))
    for x in range(7, 13):
        acts.append(Action("place", x, top, WOOD))
    return acts


def record(seed: int, out: Path, save_png: bool) -> dict:
    world = FormWorld(seed)
    states, frames, actions = [world.copy_grid()], [render(world.copy_grid())], []

    for a in demo_bridge():
        g = world.step(a)
        states.append(g)
        frames.append(render(g))
        actions.append(a.as_dict())

    # 逐块拆去左侧承重柱，让顶梁因悬挑超限而真正断裂。
    collapse = [Action("remove", 7, y, STONE)
                for y in range(H - 2, H - 6, -1)]
    collapse += [Action() for _ in range(8)]
    for a in collapse:
        g = world.step(a)
        states.append(g)
        frames.append(render(g))
        actions.append(a.as_dict())

    states_arr = np.stack(states)
    frames_arr = np.stack(frames)
    np.savez_compressed(out / "episode.npz",
                        states=states_arr, frames=frames_arr)
    (out / "actions.json").write_text(
        json.dumps(actions, ensure_ascii=False, indent=2), encoding="utf-8")

    parsed = np.stack([parse(f) for f in frames_arr])
    accuracy = float((parsed == states_arr).mean())
    report = {
        "seed": seed,
        "frames": int(len(frames_arr)),
        "render_parse_accuracy": accuracy,
        "state_shape": list(states_arr.shape),
        "frame_shape": list(frames_arr.shape),
        "final_change": float((states_arr[-1] != states_arr[len(demo_bridge())]).mean()),
    }
    (out / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    if save_png:
        try:
            from PIL import Image
            idx = np.linspace(0, len(frames_arr) - 1, 8).astype(int)
            strip = np.concatenate([frames_arr[i] for i in idx], axis=1)
            Image.fromarray(strip).resize(
                (strip.shape[1] * 6, strip.shape[0] * 6),
                Image.Resampling.NEAREST,
            ).save(out / "storyboard.png")
        except ImportError:
            print("提示：安装 pillow 后可导出 storyboard.png")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="FormWorld-2D")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("output"))
    ap.add_argument("--png", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--data", action="store_true", help="生成数据条件与评测集")
    ap.add_argument("--quick", action="store_true", help="小规模数据，用于快速验证")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--per-split", type=int, default=None)
    ap.add_argument("--frames", action="store_true", help="also store rendered RGB frames in train_*.npz (large)")
    ap.add_argument("--data-v3", action="store_true", help="生成 v3 数据（更复杂的场景/干预/建造过程），输出到 <out>/data_v3")
    ap.add_argument("--workers", type=int, default=0, help="v3 生成使用的进程数，0 = 自动")
    ap.add_argument("--p-mistake", type=float, default=None, help="v3 建造失误率，默认 0.08")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    did_something = False
    if args.data_v3:
        did_something = True
        import datasets_v3 as v3
        episodes = args.episodes or (24 if args.quick else 256)
        per_split = args.per_split or (12 if args.quick else 128)
        kw = {} if args.p_mistake is None else {"p_mistake": args.p_mistake}
        print("generating v3 data ...")
        result = v3.generate_all(args.out / "data_v3", episodes, 64, per_split, args.seed, args.workers, **kw)
        print(json.dumps({"train": result["train"]["conditions"], "bench": result["bench"]}, indent=2))
        print("audit written to", args.out / "data_v3" / "audit.json")
    if args.selftest:
        did_something = True
        print("physics:", json.dumps(physics_selftest(), indent=2))
        print("render→parse accuracy:", verify_render_parser(args.seed))

    if args.record or not (did_something or args.data or args.data_v3):
        did_something = True
        print("episode:", json.dumps(record(args.seed, args.out, args.png), indent=2))

    if args.data:
        from datasets import generate_all
        episodes = args.episodes or (24 if args.quick else 256)
        per_split = args.per_split or (12 if args.quick else 128)
        print("generating data ...")
        result = generate_all(args.out / "data", episodes, 64, per_split, args.seed, args.frames)
        print(json.dumps({"train": result["train"]["conditions"], "bench": result["bench"]}, indent=2))
        print("audit written to", args.out / "data" / "audit.json")


if __name__ == "__main__":
    main()
