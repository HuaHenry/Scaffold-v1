"""Condition arrays are uncompressed NPY files, read through memory maps."""
from pathlib import Path
import zipfile

import numpy as np

from ..shared import CONDITIONS, H, W, common, integer

KEYS = ("fg", "bg", "act")


def legacy_count(source, condition="state"):
    counts = []
    for key in ("fg", "bg"):
        with zipfile.ZipFile(Path(source) / f"train_{condition}.npz") as archive, archive.open(f"{key}.npy") as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError("unsupported legacy array header")
        if len(shape) != 4 or shape[1:] != (64, H, W) or dtype != np.uint8 or fortran:
            raise ValueError("invalid legacy array shape or dtype")
        counts.append(shape[0])
    if counts[0] != counts[1]:
        raise ValueError("legacy foreground/background episode count differs")
    return counts[0]


def allocate(root, count, conditions=CONDITIONS):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    return {c: {k: np.lib.format.open_memmap(root / f"{c}_{k}.npy", mode="w+", dtype=np.uint8,
                shape=(count, 64, H, W)) for k in KEYS} for c in conditions}


def flush(arrays):
    for values in arrays.values():
        for array in values.values():
            array.flush()


def stream_npz_prefix(path, key, destination, count, chunk=64):
    with zipfile.ZipFile(path) as z, z.open(f"{key}.npy") as source:
        version = np.lib.format.read_magic(source)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(source)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(source)
        else:
            raise ValueError("unsupported NPY header in source archive")
        if fortran or dtype != np.dtype("uint8") or shape[1:] != (64, H, W) or count > shape[0]:
            raise ValueError("legacy prefix shape/dtype/count is invalid")
        per_episode = 64 * H * W
        for start in range(0, count, chunk):
            n = min(chunk, count - start)
            data = source.read(n * per_episode)
            if len(data) != n * per_episode:
                raise ValueError("truncated legacy archive")
            destination[start:start + n] = np.frombuffer(data, np.uint8).reshape(n, 64, H, W)


def import_legacy(root, source, count, conditions=CONDITIONS):
    arrays = allocate(root, count, conditions)
    for c in conditions:
        for k in ("fg", "bg"):
            stream_npz_prefix(Path(source) / f"train_{c}.npz", k, arrays[c][k], count)
        acts = common.load_json(Path(source) / f"train_{c}_actions.json")["actions"]
        if len(acts) < count:
            raise ValueError("legacy action count is smaller than requested prefix")
        for e in range(count):
            if len(acts[e]) != 64:
                raise ValueError("legacy episode action length differs")
            arrays[c]["act"][e] = np.stack([common.encode_actions(a) for a in acts[e]]).astype(np.uint8)
        del acts
    flush(arrays)
    return arrays


class Dataset:
    def __init__(self, root, task, split, condition, episodes=None):
        self.root = Path(root)
        path = self.root / task / split
        self.arrays = {k: np.load(path / f"{condition}_{k}.npy", mmap_mode="r", allow_pickle=False) for k in KEYS}
        self.count = len(self.arrays["fg"])
        if episodes is not None:
            n = integer(episodes, "episodes")
            if n > self.count:
                raise ValueError("requested episode count exceeds frozen task data")
            self.count = n
        if any(a.dtype != np.uint8 or a.shape != (len(self.arrays["fg"]), 64, H, W) for a in self.arrays.values()):
            raise ValueError("condition arrays have invalid shape or dtype")
        self.condition = condition

    def batch(self, e, t):
        return tuple(np.asarray(self.arrays[k][e, tt]).copy() for k, tt in
                     (("fg", t - 1), ("bg", t - 1), ("fg", t), ("bg", t), ("act", t)))
