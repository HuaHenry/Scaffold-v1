"""Preflight, equal-budget batch sampling, and paired crossing analysis."""
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from exp.audit import audit
from exp.analysis import load_scores, summarize
from exp.generate_data import build, check_prefix, disjoint_scenes, prepare_validation
from exp.make_jobs import epoch_steps, jobs
from exp.shared import LEVELS, PACKAGE, common


def test_equal_budget_jobs():
    joblist = jobs("test-run", common.DEFAULT_DATA, (0,))
    assert LEVELS[-1] == 4096 and len(joblist) == 34 and len({job["out"] for job in joblist}) == 34
    for job in joblist:
        cmd = job["cmd"]
        assert "--steps" in cmd and cmd[cmd.index("--steps") + 1] == "12000"
        assert "--batch" in cmd and cmd[cmd.index("--batch") + 1] == "256"
        assert "--select-best-val" in cmd and cmd[cmd.index("--val-every") + 1] == "500"
        assert cmd[cmd.index("--val") + 1] == common.rel(common.DEFAULT_VAL)
        cond = cmd[cmd.index("--condition") + 1]
        n = int(cmd[cmd.index("--episodes") + 1])
        assert n in LEVELS
        assert ("--balanced-phases" in cmd) == (cond == "process")


def test_full_epochs_train_every_transition_exactly_five_times():
    train = import_module(f"{PACKAGE}.train")
    for condition, count in (("state", 56*16), ("process", 59*16)):
        generator = torch.Generator().manual_seed(7)
        seen = []
        for _ in range(5):
            batches = list(train.epoch_batches(count, 256, generator, torch.device("cpu")))
            assert len(batches) == epoch_steps(16, condition, 256, 5) // 5
            assert sum(len(batch) for batch in batches) == count
            assert all(0 < len(batch) <= 256 for batch in batches)
            assert torch.equal(torch.sort(torch.cat(batches)).values, torch.arange(count))
            seen.extend(batches)
        assert torch.equal(torch.bincount(torch.cat(seen), minlength=count), torch.full((count,), 5))
    assert epoch_steps(4096, "state", 256, 5) == 4480
    assert epoch_steps(4096, "process", 256, 5) == 4720


def test_validation_is_at_each_epoch_boundary_and_legacy_schedule_is_unchanged():
    train = import_module(f"{PACKAGE}.train")
    assert [step for step in range(1, 21) if train.validation_due(step, 20, 500, 4)] == [4, 8, 12, 16, 20]
    assert [step for step in range(1, 1201) if train.validation_due(step, 1200, 500, None)] == [500, 1000, 1200]
    assert not train.validation_due(20, 20, 0, None)
    assert train.better_validation({"cascade_acc": .3}, {"cascade_acc": .2})
    assert not train.better_validation({"cascade_acc": .2}, {"cascade_acc": .2})


def test_full_epoch_jobs_keep_original_protocol_isolated():
    rows = jobs("test-run", common.DEFAULT_DATA, (0,), epochs=5)
    assert len(rows) == 34
    for j in rows:
        cmd = j["cmd"]
        condition, n = cmd[cmd.index("--condition")+1], int(cmd[cmd.index("--episodes")+1])
        assert int(cmd[cmd.index("--steps")+1]) == epoch_steps(n, condition, 256, 5)
        assert cmd[cmd.index("--epochs")+1] == "5"
        assert "--balanced-phases" not in cmd
    assert "--epochs" not in jobs("test-run", common.DEFAULT_DATA, (0,))[0]["cmd"]


def test_analysis_rejects_mixed_epoch_and_fixed_step_runs(tmp_path):
    common.dump_json({"training_protocol": {"mode": "full_epochs", "epochs": 5, "batch": 256}},
                     tmp_path / "audit.json")
    with pytest.raises(ValueError, match="training protocol"):
        load_scores(tmp_path, (0,), tmp_path, tmp_path / "bench.npz", steps=12000)


def test_epoch_resume_rejects_incomplete_coverage(tmp_path):
    queue = import_module(f"{PACKAGE}.run_queue")
    job = jobs("test-run", common.DEFAULT_DATA, (0,), epochs=5)[0]
    cmd = job["cmd"]
    args = {key: cmd[cmd.index("--" + flag) + 1] for flag, key in
            (("condition", "condition"), ("seed", "seed"), ("protocol", "protocol"),
             ("steps", "steps"), ("batch", "batch"), ("episodes", "episodes"),
             ("lr", "lr"), ("change-weight", "change_weight"), ("depth", "depth"),
             ("data", "data"), ("val", "val"), ("val-every", "val_every"))}
    args.update(seed=0, steps=20, batch=256, episodes=16, epochs=5, balanced_phases=False)
    common.dump_json({"args": args}, tmp_path / "config.json")
    common.dump_json({"epochs": 5, "n_pairs": 896, "steps_per_epoch": 4,
                      "steps": 20, "samples_seen": 895*5}, tmp_path / "done.json")
    with pytest.raises(ValueError, match="full-epoch sampling"):
        queue.validate_curve_resume(job, tmp_path, [])


def test_each_process_batch_exactly_half_formation():
    train = import_module(f"{PACKAGE}.train")
    pairs = torch.tensor([(0, t) for t in range(1, 11)], dtype=torch.long)
    phase = torch.ones((1, 11), dtype=torch.long)
    phase[0, 1:6] = 0
    pools = train.phase_pools(pairs, phase, 256)
    gen = torch.Generator().manual_seed(7)
    for _ in range(12):
        idx = train.sample_batch(pairs, 256, gen, pools)
        assert int((phase[pairs[idx, 0], pairs[idx, 1]] == 0).sum()) == 128
    assert torch.equal(train.sample_batch(pairs, 16, torch.Generator().manual_seed(7)),
                       torch.randint(10, (16,), generator=torch.Generator().manual_seed(7)))
    with pytest.raises(ValueError):
        train.phase_pools(pairs, phase, 255)


def test_crossing_requires_negative_and_positive_supported_endpoints():
    n = (16, 64, 128, 256)
    a = {k: np.full((3, 8), .4) for k in n}
    b = {k: np.full((3, 8), value) for k, value in zip(n, (.2, .3, .5, .6))}
    r = summarize(a, b, n_boot=300)
    assert r["conclusion"] == "supported_crossing"
    assert r["crossings"] == [{"between": [64, 128], "status": "supported"}]
    single = summarize({k: v[:1] for k, v in a.items()}, {k: v[:1] for k, v in b.items()}, n_boot=100)
    assert single["crossings"][0]["status"] == "descriptive_only"


def test_existing_v3_prefix_and_bench_do_not_leak():
    if not common.DEFAULT_BENCH.exists():
        pytest.skip("v3 data not present")
    expanded = common.GAME_CODE / "output/data_v3_curve_512"
    if expanded.exists():
        assert check_prefix(common.DEFAULT_DATA, expanded) == 256
    r = audit(common.DEFAULT_DATA, common.DEFAULT_VAL, common.DEFAULT_BENCH, levels=(16, 64, 128, 256))
    assert r["levels"]["256"]["conditions"]["state"]["video_pairs"] == 56*256
    assert r["levels"]["256"]["conditions"]["process"]["video_pairs"] == 59*256
    assert r["bench_overlap"] == r["validation_overlap"] == 0


def test_validation_checkpoint_rule_does_not_use_bench():
    train = import_module(f"{PACKAGE}.train")
    assert train.better_validation({"cascade_acc": .2}, None)
    assert train.better_validation({"cascade_acc": .3}, {"cascade_acc": .2})
    assert not train.better_validation({"cascade_acc": .2}, {"cascade_acc": .2})
    assert not train.better_validation({"cascade_acc": .1}, {"cascade_acc": .2})


def test_partial_expansion_is_rejected(tmp_path):
    if not (common.DEFAULT_DATA / "train_summary.json").exists():
        pytest.skip("v3 source absent")
    (tmp_path / "train_state.npz").touch()
    with pytest.raises(ValueError, match="partial dataset"):
        build(common.DEFAULT_DATA, tmp_path, common.DEFAULT_VAL, episodes=4096)


def test_expanded_validation_is_new_and_reused_without_overwriting(tmp_path, monkeypatch):
    mod = import_module("exp.generate_data")
    data, val = tmp_path / "data", tmp_path / "data" / "validation"
    calls = []

    def fake_generate(argv):
        calls.append(argv)
        val.mkdir(parents=True)
        (val / "train_summary.json").write_text("{}")

    monkeypatch.setattr(mod, "import_module", lambda name: SimpleNamespace(main=fake_generate))
    monkeypatch.setattr(mod, "input_hashes", lambda path: {"validation": "unchanged"})
    assert prepare_validation(data, val, 2) == {"validation": "unchanged"}
    assert calls[0][:4] == ["--data", str(data), "--out", str(val)]
    assert calls[0][calls[0].index("--extra") + 1] == "64"
    prepare_validation(data, val, 2)
    assert len(calls) == 1
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "_raw").mkdir()
    with pytest.raises(ValueError, match="incomplete validation"):
        prepare_validation(data, broken, 2)


def test_validation_collision_requires_another_directory(tmp_path):
    data, val = tmp_path / "data", tmp_path / "validation"
    data.mkdir()
    val.mkdir()
    common.dump_json([{"final_hash": "same"}, {"final_hash": "other"}], data / "train_scenes.json")
    common.dump_json([{"final_hash": "same"}], val / "train_scenes.json")
    with pytest.raises(ValueError, match="empty separate --val"):
        disjoint_scenes(data, val)
    common.dump_json([{"final_hash": "new"}], val / "train_scenes.json")
    assert disjoint_scenes(data, val) == ({"same", "other"}, {"new"})


def test_curve_resume_rejects_missing_checkpoint(tmp_path):
    queue = import_module(f"{PACKAGE}.run_queue")
    job = jobs("test-run", common.DEFAULT_DATA, (0,))[0]
    common.dump_json({"steps": 12000}, tmp_path / "done.json")
    with pytest.raises(FileNotFoundError):
        queue.validate_curve_resume(job, tmp_path, [])
