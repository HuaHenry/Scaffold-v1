from importlib import import_module
import os
from pathlib import Path
import zipfile

import numpy as np
import pytest

S = import_module("preresearch.08_noisy_formation.exp.shared")
G = import_module("preresearch.08_noisy_formation.exp.generate")
N = import_module("preresearch.08_noisy_formation.exp.noise")


@pytest.mark.skipif(os.environ.get("RUN_08_REAL_SOURCE") != "1", reason="opt-in read-only frozen-source probe")
def test_frozen_prefix_noise_acceptance():
    root = S.PROJECT / "game_code/output/data_v3_simple_train_hard_eval_81920"
    n = 512
    arrays = {}
    with zipfile.ZipFile(root / "train_process.npz") as z:
        for key in ("fg", "bg"):
            with z.open(key + ".npy") as stream:
                version = np.lib.format.read_magic(stream)
                shape, order, dtype = (np.lib.format.read_array_header_1_0(stream) if version == (1, 0)
                                        else np.lib.format.read_array_header_2_0(stream))
                assert not order
                raw = stream.read(n * int(np.prod(shape[1:])) * dtype.itemsize)
                arrays[key] = np.frombuffer(raw, dtype=dtype).reshape(n, *shape[1:])
    actions = S.common.load_json(root / "train_process_actions.json")["actions"][:n]
    scenes = S.common.load_json(root / "train_scenes.json")[:n]
    seed = S.common.load_summary(root)["seed"]
    baseline, audits = None, {}
    for variant, p in S.NOISE.items():
        records = []
        tasks = ((arrays["fg"][e], arrays["bg"][e], actions[e], N.D.int_seed(seed, "world", e), p, 0,
                  not any(k in ("dune", "pond") for k in scenes[e]["kinds"]),
                  None if baseline is None else baseline[e]) for e in range(n))
        for e, (f, b, acts, meta) in enumerate(G.derive_results(tasks, 8)):
            G.check_pair(arrays["fg"][e], arrays["bg"][e], actions[e], f, b, acts)
            if p == 0:
                np.testing.assert_array_equal(f, arrays["fg"][e, :32])
                np.testing.assert_array_equal(b, arrays["bg"][e, :32])
                assert acts == actions[e][:32]
            records.append(meta)
        audits[variant] = G.report_audit(records, n)
        if p == 0:
            baseline = records
        print(variant, audits[variant], flush=True)
    assert S.qualifies(audits), audits
