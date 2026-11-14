"""Full frozen-file identity checks, independent of training and evaluation."""
import argparse
from pathlib import Path

import numpy as np

from ..shared import CONDITIONS, H, W, VERSION, code_identity, digest, file_hash, files_identity, load_json, safe_path
from ..tasks.registry import TASKS, select
from .storage import Dataset


def verify(root, full=True, progress=None):
    root = safe_path(root)
    marker = load_json(root / "complete.json")
    manifest = load_json(root / "manifest.json")
    if marker.get("version") != VERSION or marker.get("manifest_sha256") != file_hash(root / "manifest.json"):
        raise ValueError("dataset manifest identity differs")
    if manifest["tasks"] != {t: TASKS[t] for t in manifest["config"]["tasks"]}:
        raise ValueError("task registry differs from frozen dataset")
    if manifest["code"] != code_identity():
        raise ValueError("data/code identity differs; keep frozen code or create a new dataset")
    if full and files_identity(root, ("complete.json",), progress=progress) != marker["hashes"]:
        raise ValueError("frozen data files changed or files were added/removed")
    for task, meta in manifest["task_data"].items():
        for split, conditions in (("train", CONDITIONS), ("validation", ("state",))):
            for c in conditions:
                data = Dataset(root, task, split, c)
                if data.count != meta[split + "_episodes"]:
                    raise ValueError("episode count differs from frozen metadata")
        with np.load(root / task / "bench.npz", allow_pickle=False) as z:
            n = meta["test_scenes"]
            expected = {"ctx_fg": (n, H, W), "ctx_bg": (n, H, W), "act": (n, 2, 32, H, W),
                        "target_fg": (n, 2, H, W), "target_bg": (n, 2, H, W)}
            for key, shape in expected.items():
                if z[key].shape != shape or z[key].dtype != np.uint8:
                    raise ValueError(f"{task}: invalid benchmark {key}")
            if z["group"].shape != (n,) or z["scene_ids"].shape != (n,):
                raise ValueError("benchmark scene IDs/groups differ")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, type=Path)
    a = ap.parse_args(argv)
    manifest = verify(a.data)
    print(f"verified {len(manifest['tasks'])} tasks; frozen data/code identities match")


if __name__ == "__main__":
    main()
