"""Main-experiment tests never generate production data or train a model."""
from importlib import import_module

import numpy as np
import pytest

from exp import analysis
from exp.audit import verify_video_controls
from exp.make_jobs import jobs
from exp.shared import (CONDITIONS, DEFAULT_DATA, EXP_NAME, METRICS, PACKAGE, budget,
                        check_separation, common, require_data, validate_completed, validate_evaluation)


def test_four_condition_budget_and_output_isolation():
    rows = jobs("test", DEFAULT_DATA, DEFAULT_DATA / "validation")
    assert len(rows) == 4 and {r["name"] for r in rows} == set(CONDITIONS)
    assert len({r["out"] for r in rows}) == 4
    for row in rows:
        c, cmd = row["name"], row["cmd"]
        assert row["out"].startswith("03_main/test/train/")
        assert cmd[cmd.index("--exp")+1] == EXP_NAME
        assert cmd[cmd.index("--epochs")+1] == "5" and "--balanced-phases" not in cmd
        assert cmd[cmd.index("--episodes")+1] == "81920"
        assert int(cmd[cmd.index("--steps")+1]) == budget(c)["steps"]
        assert budget(c)["frames"] == 5242880
    assert budget("state")["steps"] == budget("partial")["steps"] == 89600
    assert budget("process")["steps"] == budget("shuffle")["steps"] == 94400
    with pytest.raises(ValueError):
        budget("unknown")


def test_training_interfaces_support_all_four_epoch_layouts():
    train = import_module(f"{PACKAGE}.train")
    import torch
    for c in CONDITIONS:
        layout = {"clip": 8, "formation": 32}
        segments = common.segments(c, layout, 64)
        count = sum(b-a-1 for a, b in segments)
        assert count == (56 if c in ("state", "partial") else 59)
        seen = np.zeros(count, dtype=int)
        generator = torch.Generator().manual_seed(0)
        for _ in range(5):
            pieces = list(train.epoch_batches(count, 16, generator, torch.device("cpu")))
            order = torch.cat(pieces).numpy()
            assert sorted(order) == list(range(count))
            seen[order] += 1
        assert np.all(seen == 5)


def test_missing_production_data_fails_without_generating(tmp_path):
    with pytest.raises(FileNotFoundError, match="existing data only"):
        require_data(tmp_path / "absent")
    assert not (tmp_path / "absent").exists()


def test_all_three_scene_sets_must_be_disjoint():
    check_separation({"a"}, {"b"}, {"c"})
    for t, v, b in (({"a"}, {"a"}, {"c"}), ({"a"}, {"b"}, {"a"}), ({"a"}, {"b"}, {"b"})):
        with pytest.raises(ValueError, match="pairwise disjoint"):
            check_separation(t, v, b)


def test_video_content_controls_detect_changed_operation_and_shuffle(tmp_path):
    n = 2
    p = np.arange(n*64*4, dtype=np.int32).reshape(n, 64, 2, 2)
    state = np.concatenate((p[:, 32:], p[:, 32:]), axis=1)
    order = list(reversed(range(32)))
    shuffled = np.concatenate((p[:, order], p[:, 32:]), axis=1)
    for c, values in (("state", state), ("process", p), ("shuffle", shuffled)):
        np.savez_compressed(tmp_path / f"train_{c}.npz", fg=values, bg=values)
    acts = [[[{"type": "noop", "t": t}] for t in range(64)] for _ in range(n)]
    common.dump_json({"actions": acts}, tmp_path / "train_process_actions.json")
    common.dump_json({"actions": [a[32:]+a[32:] for a in acts]}, tmp_path / "train_state_actions.json")
    common.dump_json({"actions": [[a[t] for t in order]+a[32:] for a in acts],
                      "extra": [{"order": order} for _ in range(n)]}, tmp_path / "train_shuffle_actions.json")
    verify_video_controls(tmp_path, n)
    corrupted = shuffled.copy()
    corrupted[0, 0, 0, 0] += 1
    np.savez_compressed(tmp_path / "train_shuffle.npz", fg=corrupted, bg=shuffled)
    with pytest.raises(ValueError, match="permuted"):
        verify_video_controls(tmp_path, n)


def completed_fixture(tmp_path, condition):
    b = budget(condition)
    out = tmp_path / condition
    out.mkdir()
    (out / "ckpt.pt").write_text("test checkpoint placeholder")
    hashes = {k: "fixture" for k in ("train_summary.json", "train_state.npz", "train_state_actions.json",
                                     f"train_{condition}.npz", f"train_{condition}_actions.json")}
    audited = {"inputs": {"data": "data", "val": "val", "bench": "bench.npz"},
               "hashes": {"data": hashes, "val": hashes, "bench": {"bench.npz": "fixture", "bench.json": "fixture"}}}
    args = {"condition": condition, "episodes": 81920, "epochs": 5, "steps": b["steps"],
            "batch": 256, "protocol": "video", "lr": 3e-4, "change_weight": 8, "depth": 8,
            "unroll": 1, "seed": 0, "data": "data", "val": "val", "val_every": 500,
            "select_best_val": True, "balanced_phases": False, "log_batch_composition": True}
    common.dump_json({"args": args,
                      "data_sha1": {k: "fixture" for k in (f"train_{condition}.npz", f"train_{condition}_actions.json", "train_summary.json")},
                      "val_sha1": {k: "fixture" for k in ("train_state.npz", "train_state_actions.json", "train_summary.json")}},
                     out / "config.json")
    done = {"steps": b["steps"], "epochs": 5, "n_pairs": b["pairs"], "samples_seen": b["samples"],
            "steps_per_epoch": b["steps_per_epoch"], "selection": "val_cascade_acc",
            "best_step": b["steps_per_epoch"], "form_frac_mean": b["formation_fraction"],
            "ckpt_sha1": common.sha1_file(out / "ckpt.pt")}
    common.dump_json(done, out / "done.json")
    return out, audited, done


def test_partial_and_shuffle_completion_and_incomplete_epoch_rejection(tmp_path):
    for c in ("partial", "shuffle"):
        out, audited, done = completed_fixture(tmp_path, c)
        assert validate_completed(out, c, audited)["samples_seen"] == budget(c)["samples"]
        done["samples_seen"] -= 1
        common.dump_json(done, out / "done.json")
        with pytest.raises(ValueError, match="coverage"):
            validate_completed(out, c, audited)


def test_evaluation_rejects_different_checkpoint(tmp_path):
    out, audited, done = completed_fixture(tmp_path, "state")
    common.dump_json({"bench": common.rel(common.resolve("bench.npz")), "n_samples": 512,
                      "bench_sha1": "fixture", "bench_json_sha1": "fixture", "checkpoint_sha1": "wrong"},
                     out / "eval.json")
    with pytest.raises(ValueError, match="checkpoint"):
        validate_evaluation(out, audited, done)


def synthetic_results():
    splits = np.repeat(np.arange(4), 128)
    groups = {c: {m: np.full(512, .2 + j*.1) for m in METRICS} for j, c in enumerate(CONDITIONS)}
    return groups, splits


def test_main_summary_all_four_conditions_and_five_contrasts():
    groups, splits = synthetic_results()
    result = analysis.summarize(groups, splits, 100)
    for split in common.SPLITS:
        row = result[split]
        assert len(row["conditions"]) == 4 and len(row["comparisons"]) == 5
        assert row["comparisons"]["process-state"]["difference"] == pytest.approx(.1)
        assert row["comparisons"]["process-partial"]["difference"] == pytest.approx(-.1)
    del groups["shuffle"]
    with pytest.raises(ValueError, match="four conditions"):
        analysis.summarize(groups, splits, 100)


def test_all_four_results_load_and_different_case_order_is_rejected(tmp_path, monkeypatch):
    groups, splits = synthetic_results()
    bench = tmp_path / "bench.npz"
    np.savez_compressed(bench, split_ids=splits)
    folders, audited = {}, None
    for c in CONDITIONS:
        out, current, done = completed_fixture(tmp_path, c)
        audited = current if audited is None else audited
        audited["hashes"]["data"].update(current["hashes"]["data"])
        audited["inputs"]["bench"] = common.rel(bench)
        common.dump_json({"bench": common.rel(bench), "n_samples": 512, "bench_sha1": "fixture",
                          "bench_json_sha1": "fixture", "checkpoint_sha1": done["ckpt_sha1"]}, out / "eval.json")
        np.savez_compressed(out / "eval_samples.npz", split_ids=splits, **groups[c])
        folders[c] = out
    common.dump_json(audited, tmp_path / "audit.json")
    monkeypatch.setattr(analysis, "verify_inputs", lambda record: {"bench": bench})
    monkeypatch.setattr(common, "find_latest", lambda condition, *args: folders[condition])
    loaded, actual_splits, sources, _ = analysis.load_results(tmp_path)
    assert set(loaded) == set(CONDITIONS) and len(sources) == 4
    assert np.array_equal(actual_splits, splits)
    np.savez_compressed(folders["shuffle"] / "eval_samples.npz", split_ids=splits[::-1], **groups["shuffle"])
    with pytest.raises(ValueError, match="ordering"):
        analysis.load_results(tmp_path)


def test_completed_main_job_is_skipped_without_launching_training(tmp_path, monkeypatch):
    from exp import run_queue
    out, audited, done = completed_fixture(tmp_path, "partial")
    bench = common.resolve(audited["inputs"]["bench"])
    common.dump_json({"bench": common.rel(bench), "n_samples": 512,
                      "bench_sha1": "fixture", "bench_json_sha1": "fixture",
                      "checkpoint_sha1": done["ckpt_sha1"]}, out / "eval.json")
    (out / "eval_samples.npz").touch()
    monkeypatch.setattr(common, "resolve_out", lambda path: out)
    def never_launch(*args, **kwargs):
        pytest.fail("completed four-condition run must not launch a subprocess")
    monkeypatch.setattr(run_queue.subprocess, "call", never_launch)
    assert run_queue.run_job({"name": "partial", "out": "test", "cmd": []}, "cpu", {"bench": bench}, audited)


def test_main_table_and_figures_render_without_training(tmp_path):
    groups, splits = synthetic_results()
    report = analysis.summarize(groups, splits, 100)
    table = analysis.markdown(report, {"inputs": {}}, {})
    assert "partial" in table and "shuffle" in table and "process-partial" in table
    analysis.plot(report, tmp_path)
    for name in ("four_conditions.png", "four_conditions.pdf", "paired_differences.png", "paired_differences.pdf"):
        assert (tmp_path / name).stat().st_size > 1000
