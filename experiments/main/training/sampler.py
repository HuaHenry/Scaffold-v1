"""Exact task-balanced and phase-balanced exposure without allocating transition pools."""
import numpy as np

from ..shared import H, W, integer, quota
from ..data.storage import Dataset

FORMATION = np.arange(1, 32)
MIXED_OPERATION = np.concatenate([np.arange(a + 1, a + 8) for a in (32, 40, 48, 56)])
STATE_OPERATION = np.concatenate([np.arange(a + 1, a + 8) for a in range(0, 64, 8)])


class Sampler:
    def __init__(self, p, row, split="train"):
        self.tasks = list(row["tasks"])
        self.condition = row["condition"] if split == "train" else "state"
        self.data = {t: Dataset(p["data"], t, split if split == "train" else "validation", self.condition,
                                p["episodes_per_task"] if split == "train" else None) for t in self.tasks}
        self.ratio = p["formation_ratio"]
        self.rng = np.random.default_rng(row["seed"])
        self.total = 0
        self.seen = {t: 0 for t in self.tasks}
        self.formed = {t: 0 for t in self.tasks}

    def batch(self, batch):
        integer(batch, "batch")
        output = [np.empty((batch, H, W), np.uint8) for _ in range(5)]
        labels = (np.arange(batch) + self.total) % len(self.tasks)
        for i, task in enumerate(self.tasks):
            positions = np.flatnonzero(labels == i)
            n = len(positions)
            if not n:
                continue
            mixed = self.condition in ("process", "shuffle")
            nf = quota(self.ratio, self.seen[task] + n) - quota(self.ratio, self.seen[task]) if mixed else 0
            ts = np.concatenate((self.rng.choice(FORMATION, nf), self.rng.choice(MIXED_OPERATION if mixed else STATE_OPERATION, n - nf)))
            self.rng.shuffle(ts)
            es = self.rng.integers(self.data[task].count, size=n)
            values = self.data[task].batch(es, ts)
            for out, value in zip(output, values):
                out[positions] = value
            self.seen[task] += n
            self.formed[task] += nf
        self.total += batch
        order = self.rng.permutation(batch)
        return tuple(v[order] for v in output)

    def exposure(self):
        return {t: {"samples": self.seen[t], "formation": self.formed[t],
                    "operation": self.seen[t] - self.formed[t]} for t in self.tasks}
