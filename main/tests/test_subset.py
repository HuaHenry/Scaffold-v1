from pathlib import Path
from threading import Event

import numpy as np
import pytest
import torch

from main import shared
from main.data import generate, audit
from main.training import protocol, train, queue
from main.evaluation import evaluate
from scripts import main_subset as subset


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("subset-data") / "frozen"
    generate.generate(root, {"tasks": ["W02", "G02"], "episodes": 2, "val_episodes": 2,
                             "test_scenes": 2, "seed": 37, "max_attempts": 100})
    return root


@pytest.fixture
def source(data, tmp_path):
    manifest = audit.verify(data)
    p = protocol.make_protocol(manifest, data, ["W02", "G02"], 2, "0.4", 1,
                               modes=("single",),
                               overrides={"dim": 12, "depth": 1, "heads": 3, "batch": 4,
                                          "val_every": 1, "log_every": 1},
                               selection_note="CPU test fixture")
    root = protocol.prepare(tmp_path / "source", p, manifest)
    rows = subset.selected_jobs(p, ["W02"])
    for row in rows:
        train.train_job(root, p, row, "cpu")
        evaluate.evaluate_job(root, p, row, "cpu", batch=2)
    return root, p, rows


def tree_snapshot(root):
    return {p.relative_to(root).as_posix(): (shared.file_hash(p), p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


def test_default_five_tasks_twenty_models():
    p = {"tasks": list(subset.DEFAULT_TASKS), "training_seeds": [0], "modes": ["single"],
         "steps_single": 94400, "mixed_budget": "per_task"}
    rows = subset.selected_jobs(p, subset.DEFAULT_TASKS)
    assert len(rows) == 20 and {r["seed"] for r in rows} == {0}
    assert {r["tasks"][0] for r in rows} == set(subset.DEFAULT_TASKS)
    assert all(r["mode"] == "single" for r in rows)


def test_copy_byte_identity_no_source_writes_and_no_retraining(source, tmp_path, monkeypatch):
    old, p, rows = source
    before = tree_snapshot(old)
    target = tmp_path / "target"
    subset.prepare_subset(old, target, ["W02"])
    assert tree_snapshot(old) == before
    for name in subset.SNAPSHOTS:
        assert (old / name).read_bytes() == (target / name).read_bytes()
    for row in rows:
        out = target / "train" / row["id"]
        train.validate_done(out, p, row)
        evaluate.validate_eval(out, p, row, "W02")
        for n in subset.TRAIN_FILES:
            a, b = old / "train" / row["id"] / n, out / n
            assert a.read_bytes() == b.read_bytes()
            assert a.stat().st_ino != b.stat().st_ino

    def no_child(*args, **kwargs):
        pytest.fail("completed model/eval should not launch training or inference")

    monkeypatch.setattr(queue.subprocess, "Popen", no_child)
    subset.execute_subset(target, cards="cpu", interval=.01)
    subset.prepare_subset(old, target, ["W02"])
    assert tree_snapshot(old) == before
    result = subset.analyze_subset(target, boot=16)
    report = shared.load_json(result / "results.json")
    assert list(report["by_mode"]["single"]["tasks"]) == ["W02"]
    assert report["source_run"] == str(old) and len(report["copied_jobs"]) == 4
    assert report["protocol"]["tasks"] == ["W02"]
    assert shared.load_json(target / "protocol.json")["tasks"] == ["W02", "G02"]
    assert (result / "figs" / "single.png").exists()


def test_missing_models_only_new_jobs_train(source, tmp_path, monkeypatch):
    old, p, old_rows = source
    target = tmp_path / "target"
    partial = old / "train" / "single_G02_state_seed0"
    partial.mkdir()
    shared.write_json(partial / "config.json", {"protocol": p, "job": subset.selected_jobs(p, ["G02"])[0]})
    (partial / "train.log").write_text("interrupted at step50")
    subset.prepare_subset(old, target, ["W02", "G02"])
    assert not (target / "train" / "single_G02_state_seed0").exists()
    _, recorded, rows, _ = subset.verify_subset(target)
    assert len(recorded["copied"]) == 4 and len(rows) == 8
    with pytest.raises(FileNotFoundError):
        queue.preflight(target, p, rows)
    requested = []

    def launch(run, p, row, card, evaluate_only, stop, interval):
        requested.append(row["id"])
        out = run / "train" / row["id"]
        if row["tasks"] == ["W02"]:
            train.validate_done(out, p, row)
        else:
            train.train_job(run, p, row, "cpu")
            evaluate.evaluate_job(run, p, row, "cpu", batch=2)
        return True

    monkeypatch.setattr(subset.queue, "run_job", launch)
    subset.execute_subset(target, cards="cpu", interval=.01)
    assert set(requested) == {r["id"] for r in rows}
    assert len(list((target / "train").glob("*/train_complete.json"))) == 8
    subset.verify_subset(target)


def test_real_cpu_child_respects_subset(source, tmp_path):
    old, p, _ = source
    target = tmp_path / "target"
    subset.prepare_subset(old, target, ["W02", "G02"])
    row = subset.selected_jobs(p, ["G02"])[0]
    assert queue.run_job(target, p, row, "cpu", False, Event(), .1)
    train.validate_done(target / "train" / row["id"], p, row)
    evaluate.validate_eval(target / "train" / row["id"], p, row, "G02")


def test_tampered_copy_rejected(source, tmp_path):
    old, _, rows = source
    target = tmp_path / "target"
    subset.prepare_subset(old, target, ["W02"])
    path = target / "train" / rows[0]["id"] / "done.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="copied model"):
        subset.verify_subset(target)


def test_no_changes_to_frozen_selection(source, tmp_path):
    old, _, _ = source
    target = tmp_path / "target"
    subset.prepare_subset(old, target, ["W02"])
    with pytest.raises(ValueError, match="selection/source changed"):
        subset.prepare_subset(old, target, ["G02"])
    plan = shared.load_json(target / "subset.json")
    plan["tasks"] = ["G02"]
    shared.write_json(target / "subset.json", plan)
    with pytest.raises(ValueError, match="identity changed"):
        subset.verify_subset(target)


def test_source_active_refused_and_overlap_refused(source, tmp_path):
    old, _, _ = source
    with shared.lock(old / "pipeline.lock"):
        with pytest.raises(ValueError, match="still running"):
            subset.prepare_subset(old, tmp_path / "target", ["W02"])
    with pytest.raises(ValueError, match="overlaps"):
        subset.prepare_subset(old, old / "child", ["W02"])


@pytest.mark.parametrize("interrupted_name", ["ckpt.pt.copying", "protocol.json.copying", "jobs.json.copying", "run.json.copying"])
def test_incomplete_copy_restart(source, tmp_path, monkeypatch, interrupted_name):
    old, _, rows = source
    target = tmp_path / "target"
    copy = subset.shutil.copyfile
    fail_once = []

    def interrupted(src, dest):
        if str(dest).endswith(interrupted_name) and not fail_once:
            fail_once.append(True)
            Path(dest).write_text("partial")
            raise OSError("forced interruption")
        return copy(src, dest)

    monkeypatch.setattr(subset.shutil, "copyfile", interrupted)
    with pytest.raises(OSError, match="forced"):
        subset.prepare_subset(old, target, ["W02"])
    assert not (target / "subset_complete.json").exists()
    monkeypatch.setattr(subset.shutil, "copyfile", copy)
    subset.prepare_subset(old, target, ["W02"])
    _, plan, _, _ = subset.verify_subset(target)
    assert len(plan["copied"]) == len(rows)


def test_invalid_source_model_rejected_before_copy(source, tmp_path):
    old, _, rows = source
    (old / "train" / rows[0]["id"] / "done.json").write_text("{}")
    target = tmp_path / "target"
    with pytest.raises(ValueError):
        subset.prepare_subset(old, target, ["W02"])
    assert not (target / "train").exists()


def test_launcher_change_refused(source, tmp_path, monkeypatch):
    old, _, _ = source
    target = tmp_path / "target"
    subset.prepare_subset(old, target, ["W02"])
    monkeypatch.setattr(subset, "launcher_identity", lambda: "changed")
    with pytest.raises(ValueError, match="launcher changed"):
        subset.verify_subset(target)


@pytest.mark.parametrize("flag", ["--run", "--ru", "--r", "--run=", "--ru=", "--r="])
def test_full_launcher_refuses_subset(flag, tmp_path):
    import subprocess
    target = tmp_path / "subset"
    target.mkdir()
    (target / "subset.json").write_text("{}")
    args = [flag + str(target)] if flag.endswith("=") else [flag, str(target)]
    result = subprocess.run(["bash", str(shared.ROOT / "run_main.sh"), "--stage", "train", *args],
                            capture_output=True, text=True)
    assert result.returncode == 2 and "run_subset.sh" in result.stderr
