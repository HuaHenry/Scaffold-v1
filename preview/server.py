"""FormWorld-2D 数据集预览服务。

用法（在仓库根目录）:
    game_code/.venv/bin/python preview/server.py
    game_code/.venv/bin/python preview/server.py --data game_code/output/data --port 8765
    （端口被占用时自动顺延到下一个空闲端口，实际地址以终端打印为准）

只依赖 numpy。启动后自动打开浏览器；--no-browser 可关闭。
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
CONDITIONS = ("state", "process", "partial", "shuffle")

MATERIAL_NAMES = ["EMPTY", "GROUND", "STONE", "WOOD", "LEAF", "SAND", "WATER", "CLOUD", "DIRT", "GRASS", "BRICK",
                  "GLASS", "ROOF", "DOOR", "WALL", "TORCH", "LADDER", "FLOWER", "OPEN_DOOR", "DOOR_TOP",
                  "OPEN_DOOR_TOP"]
FALLBACK_COLORS = ['#d9ebf1', '#454851', '#838a90', '#b78143', '#418b48', '#d9c18a', '#529bc0', '#f4fafb',
                   '#896044', '#68a849', '#bf6750', '#8bd6df', '#934b38', '#a4723e', '#806244', '#f2b04d',
                   '#c89c58', '#edbe65', '#69422b', '#a16a35', '#543724']


def load_palette() -> list[str]:
    try:
        sys.path.insert(0, str(REPO / "game_code"))
        import render  # type: ignore
        return list(render.HEX_COLORS)
    except Exception:
        return FALLBACK_COLORS


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode("ascii")


class Store:
    def __init__(self, data_dir: Path):
        self.dir = data_dir
        self.lock = threading.Lock()
        self._train: dict[str, dict] = {}
        self._actions: dict[str, dict] = {}
        self._bench: dict | None = None
        self._bench_json: dict | None = None
        self._scenes: list | None = None

    def exists(self, name: str) -> bool:
        return (self.dir / name).exists()

    def train(self, cond: str) -> dict:
        with self.lock:
            if cond not in self._train:
                z = np.load(self.dir / f"train_{cond}.npz")
                self._train[cond] = {k: z[k] for k in ("fg", "bg", "cont", "phase")}
            return self._train[cond]

    def actions(self, cond: str) -> dict:
        with self.lock:
            if cond not in self._actions:
                self._actions[cond] = json.loads((self.dir / f"train_{cond}_actions.json").read_text("utf-8"))
            return self._actions[cond]

    def scenes(self) -> list:
        with self.lock:
            if self._scenes is None:
                p = self.dir / "train_scenes.json"
                self._scenes = json.loads(p.read_text("utf-8")) if p.exists() else []
            return self._scenes

    def bench(self) -> tuple[dict, dict]:
        with self.lock:
            if self._bench is None:
                z = np.load(self.dir / "bench.npz")
                self._bench = {k: z[k] for k in z.files}
                self._bench_json = json.loads((self.dir / "bench.json").read_text("utf-8"))
            return self._bench, self._bench_json  # type: ignore


def make_info(store: Store) -> dict:
    info: dict = {"palette": load_palette(), "materials": MATERIAL_NAMES, "train": {}, "bench": None, "layout": {}}
    summ = store.dir / "train_summary.json"
    if summ.exists():
        s = json.loads(summ.read_text("utf-8"))
        info["layout"] = s.get("layout", {})
        for c in CONDITIONS:
            if c in s.get("conditions", {}) and store.exists(f"train_{c}.npz"):
                v = s["conditions"][c]
                info["train"][c] = {"episodes": v["episodes"], "frames": v["frames_per_episode"],
                                    "continuous_transitions": v["continuous_transitions"],
                                    "formation_frames": v["formation_frames"], "reset_frames": v["reset_frames"]}
    if store.exists("bench.npz") and store.exists("bench.json"):
        _, bj = store.bench()
        splits = bj["splits"]
        by_split = {s: [i for i, x in enumerate(bj["split_of_sample"]) if x == s] for s in splits}
        info["bench"] = {"samples": bj["samples"], "splits": splits, "indices": by_split}
    return info


def make_train(store: Store, cond: str, ep: int) -> dict:
    d = store.train(cond)
    n = d["fg"].shape[0]
    ep = max(0, min(n - 1, ep))
    A = store.actions(cond)
    scenes = store.scenes()
    extra = A.get("extra", [])
    return {
        "kind": "train", "cond": cond, "episode": ep, "episodes": n, "T": int(d["fg"].shape[1]),
        "H": int(d["fg"].shape[2]), "W": int(d["fg"].shape[3]),
        "fg": b64(d["fg"][ep]), "bg": b64(d["bg"][ep]),
        "cont": d["cont"][ep].astype(int).tolist(), "phase": d["phase"][ep].astype(int).tolist(),
        "actions": A["actions"][ep],
        "scene": scenes[ep] if ep < len(scenes) else None,
        "extra": extra[ep] if ep < len(extra) else None,
    }


def make_bench(store: Store, idx: int) -> dict:
    b, bj = store.bench()
    n = int(b["ctx_fg"].shape[0])
    idx = max(0, min(n - 1, idx))
    H, W = b["ctx_fg"].shape[1:]
    panels = []
    for k in range(2):
        fg = np.concatenate([b["ctx_fg"][idx][None], b["fut_fg"][idx, k]], 0)
        bg = np.concatenate([b["ctx_bg"][idx][None], b["fut_bg"][idx, k]], 0)
        T = fg.shape[0]
        acts = [[] for _ in range(T)]
        acts[1] = [bj["interventions"][idx][k]]
        panels.append({"k": k, "T": T, "fg": b64(fg), "bg": b64(bg), "actions": acts,
                       "label": float(b["labels"][idx][k]), "settled_at": int(b["settled_at"][idx][k]),
                       "n_changed": int(b["n_changed"][idx][k])})
    return {"kind": "bench", "index": idx, "samples": n, "H": int(H), "W": int(W),
            "split": bj["split_of_sample"][idx], "scene_kind": bj["scene_kinds"][idx],
            "scene": bj["scenes"][idx], "rank_valid": bool(b["rank_valid"][idx]), "panels": panels}


class Handler(BaseHTTPRequestHandler):
    store: Store

    def log_message(self, *a):  # 静默
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                self._send(200, (ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif u.path in ("/game/physics.js", "/game/painter.js"):
                # 直接复用游戏自己的贴图绘制（只读，白名单）
                self._send(200, (REPO / "game_code" / u.path.rsplit("/", 1)[1]).read_bytes(),
                           "application/javascript; charset=utf-8")
            elif u.path == "/api/info":
                self._json(make_info(self.store))
            elif u.path == "/api/train":
                cond = q.get("cond", "")
                if cond not in CONDITIONS or not self.store.exists(f"train_{cond}.npz"):
                    return self._json({"error": f"unknown condition {cond}"}, 404)
                self._json(make_train(self.store, cond, int(q.get("ep", 0))))
            elif u.path == "/api/bench":
                if not self.store.exists("bench.npz"):
                    return self._json({"error": "bench.npz not found"}, 404)
                self._json(make_bench(self.store, int(q.get("i", 0))))
            else:
                self._send(404, b"not found", "text/plain")
        except (ValueError, KeyError) as e:
            self._json({"error": str(e)}, 400)
        except Exception as e:  # noqa: BLE001
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=REPO / "game_code" / "output" / "data")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    if not a.data.exists():
        sys.exit(f"数据目录不存在: {a.data}")
    Handler.store = Store(a.data)
    srv, port = None, a.port
    for port in range(a.port, a.port + 50):
        try:
            srv = ThreadingHTTPServer((a.host, port), Handler)
            break
        except OSError:
            print(f"端口 {port} 已被占用，尝试 {port + 1}", flush=True)
    if srv is None:
        sys.exit(f"{a.port}–{a.port + 49} 都被占用，请用 --port 指定其他端口")
    url = f"http://{a.host}:{port}/"
    print(f"数据目录: {a.data}\n预览地址: {url}\nCtrl+C 退出", flush=True)
    if not a.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
