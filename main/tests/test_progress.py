from pathlib import Path
import queue
import subprocess
import sys
import time

import numpy as np
import pytest
import torch

from main import shared
from main.data import audit, generate, storage
from main.data.progress import Monitor, Reporter, duration, stage_estimate
from main.training import protocol, sampler, train


@pytest.fixture(scope="module")
def larger_data(tmp_path_factory):
    root = tmp_path_factory.mktemp("prefix-progress") / "data"
    config = {"tasks": ["W02", "G02"], "episodes": 3, "val_episodes": 2, "test_scenes": 2,
              "seed": 63, "max_attempts": 100}
    generate.generate(root, config, progress_every=.1)
    return root


def test_stage_speed_eta_and_unknown_time():
    event = {"started": 10., "done": 20, "total": 100}
    elapsed, rate, eta = stage_estimate(event, 20.)
    assert (elapsed, rate, eta) == (10., 2., 40.)
    assert duration(eta) == "0分40秒"
    assert stage_estimate({**event, "done": 0}, 20.)[2] is None
    assert stage_estimate({**event, "total": None}, 20.)[2] is None
    assert duration(None) == "估算中"
    assert duration(3601) == "1小时00分"


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_invalid_progress_interval_has_no_writes(tmp_path, interval):
    config = {"tasks": ["G01"], "episodes": 1, "val_episodes": 1, "test_scenes": 1,
              "seed": 1, "max_attempts": 100}
    out = tmp_path / "absent"
    with pytest.raises(ValueError, match="progress interval"):
        generate.generate(out, config, progress_every=interval)
    assert not out.exists()


def test_reporter_throttles_events_and_always_reports_tail():
    events = queue.Queue()
    report = Reporter("G01", events, interval=1000)
    report.start("train", 3, "集", "samples")
    report.update(1)
    report.update(2)
    report.update(3)
    report.finish()
    records = []
    while not events.empty():
        records.append(events.get())
    assert [v["done"] for v in records] == [0, 3, 3]
    assert records[-1]["status"] == "finished"


def test_monitor_counts_out_of_order_tasks_reuse_and_unknown_eta(capsys):
    config = {"episodes": 10, "val_episodes": 2, "test_scenes": 2}
    with Monitor(["T01", "W01", "G01"], config, 2, interval=100) as monitor:
        assert monitor.sample_summary()[3] is None
        monitor.apply({"kind": "committed", "task": "G01", "reused": True})
        event = {"kind": "progress", "task": "W01", "stage": "train", "started": time.monotonic()-5,
                 "done": 5, "total": 10, "category": "samples", "unit": "集", "status": "running"}
        monitor.apply(event)
        monitor.apply(event)
        done, total, rate, eta = monitor.sample_summary()
        assert done == 19 and total == 28 and rate > 0 and eta > 0
        monitor.apply({"kind": "committed", "task": "T01", "reused": False})
        assert len(monitor.committed_tasks) == 2
        monitor.snapshot()
    text = capsys.readouterr().out
    assert "阶段剩余" in text and "样本阶段粗估剩余" in text
    assert "不含T01导入、写盘和审计" in text


def test_unknown_blocking_stage_keeps_printing_elapsed(capsys):
    config = {"episodes": 1, "val_episodes": 1, "test_scenes": 1}
    with Monitor(["T01"], config, 1, interval=.01) as monitor:
        reporter = Reporter("T01", monitor.events, .01)
        with reporter.stage("读取与写盘"):
            time.sleep(.06)
    assert capsys.readouterr().out.count("读取与写盘") >= 2


def test_hash_progress_is_byte_exact_and_does_not_change_hashes(tmp_path):
    path = tmp_path / "file.bin"
    path.write_bytes(b"abc" * 5000)
    original = shared.files_identity(tmp_path)
    seen = []
    assert shared.files_identity(tmp_path, progress=lambda done, total: seen.append((done, total))) == original
    assert seen[0] == (0, 15000) and seen[-1] == (15000, 15000)
    assert all(a[0] <= b[0] for a, b in zip(seen, seen[1:]))


def test_legacy_import_reports_frames_actions_and_flush(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    fg = np.zeros((2, 64, 28, 56), np.uint8)
    np.savez_compressed(source / "train_state.npz", fg=fg, bg=fg)
    shared.write_json(source / "train_state_actions.json", {"actions": [[[] for _ in range(64)] for _ in range(2)]})
    events = queue.Queue()
    reporter = Reporter("T01", events, .01)
    storage.import_legacy(tmp_path / "train", source, 2, ("state",), reporter=reporter)
    records = []
    while not events.empty():
        records.append(events.get())
    finished = [v for v in records if v["status"] == "finished"]
    frame = next(v for v in finished if "fg 解压" in v["stage"])
    assert frame["done"] == frame["total"] == 2 * 64 * 28 * 56
    action = next(v for v in finished if "编码动作" in v["stage"])
    assert action["done"] == action["total"] == 2
    assert any("同步磁盘" in v["stage"] for v in finished)


def test_generate_finish_after_final_audit_and_progress_setting_is_not_identity(tmp_path, capsys):
    out = tmp_path / "data"
    config = {"tasks": ["G01"], "episodes": 1, "val_episodes": 1, "test_scenes": 1,
              "seed": 28, "max_attempts": 100}
    generate.generate(out, config, progress_every=.01)
    text = capsys.readouterr().out
    assert "1/1 (100.0%)" in text
    assert text.rindex("整体完成") > text.rindex("最终冻结审计")
    before = shared.files_identity(out)
    generate.generate(out, config, progress_every=20)
    assert shared.files_identity(out) == before
    assert "progress_every" not in shared.load_json(out / "request.json")["config"]
    assert not any("progress" in p.name or p.suffix == ".log" for p in out.rglob("*"))


def test_final_audit_failure_does_not_announce_overall_completion(tmp_path, monkeypatch, capsys):
    config = {"tasks": ["G01"], "episodes": 1, "val_episodes": 1, "test_scenes": 1,
              "seed": 28, "max_attempts": 100}
    def reject(*args, **kwargs):
        raise ValueError("simulated final audit failure")
    monkeypatch.setattr(generate, "verify", reject)
    with pytest.raises(ValueError, match="final audit"):
        generate.generate(tmp_path / "data", config, progress_every=.01)
    text = capsys.readouterr().out
    assert "生成失败" in text and "整体完成，冻结校验通过" not in text


def test_parallel_progress_and_serial_generate_identical_data(tmp_path):
    import os
    roots = [tmp_path / "serial", tmp_path / "parallel"]
    outputs = []
    for root, workers in zip(roots, [1, 2]):
        command = [sys.executable, "-m", "main.data.generate", "--out", str(root), "--tasks", "W02,G02",
                   "--episodes", "1", "--val-episodes", "1", "--test-scenes", "1", "--seed", "82",
                   "--workers", str(workers), "--progress-every", ".01"]
        p = subprocess.run(command, cwd=shared.PROJECT, capture_output=True, text=True,
                           env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        assert p.returncode == 0, p.stdout + p.stderr
        assert "已提交任务" in p.stdout and "阶段剩余" in p.stdout and "整体完成" in p.stdout
        outputs.append(p.stdout)
    assert shared.files_identity(roots[0]) == shared.files_identity(roots[1])
    assert "W02" in outputs[1] and "G02" in outputs[1]


@pytest.mark.parametrize("n", [1, 2])
def test_larger_frozen_dataset_selects_prefix_for_every_condition_and_mode(larger_data, tmp_path, n):
    manifest = audit.verify(larger_data)
    before = shared.files_identity(larger_data)
    p = protocol.make_protocol(manifest, larger_data, "all", n, ".4", steps=1, selection_note="test prefix")
    out = protocol.prepare(tmp_path / f"run_n{n}", p, manifest)
    assert p["training_seeds"] == [0]
    for task, spec in p["training_prefixes"].items():
        assert spec == {"available_episodes": 3, "used_episodes": n, "start": 0, "stop": n, "step": 1}
    for row in protocol.jobs(p):
        sample = sampler.Sampler(p, row)
        for task, data in sample.data.items():
            original = data.batch
            def restricted(e, t, original=original):
                assert np.min(e) >= 0 and np.max(e) < n
                return original(e, t)
            data.batch = restricted
            assert data.count == n and data.arrays["fg"].shape[0] == 3
            with pytest.raises(ValueError, match="selected training prefix"):
                original(np.array([n]), np.array([1]))
        for _ in range(3):
            sample.batch(17)
        validation = sampler.Sampler(p, row, "validation")
        assert all(d.count == 2 for d in validation.data.values())
    assert protocol.verify_run(out)[0] == p
    assert shared.files_identity(larger_data) == before


def test_prefix_training_runs_real_cpu_without_regenerating_larger_data(larger_data, tmp_path):
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        manifest = audit.verify(larger_data)
        p = protocol.make_protocol(manifest, larger_data, "all", 1, ".4", steps=1,
            overrides={"dim": 12, "depth": 1, "heads": 3, "batch": 4, "val_every": 1, "log_every": 1},
            selection_note="test prefix")
        run = protocol.prepare(tmp_path / "run", p, manifest)
        row = next(r for r in protocol.jobs(p) if r["mode"] == "mixed" and r["condition"] == "process")
        done = train.train_job(run, p, row, "cpu")
        assert done["samples_seen"] == 8
        assert train.validate_done(run / "train" / row["id"], p, row) == done
    finally:
        torch.set_num_threads(old)


def test_prefix_larger_than_frozen_dataset_is_rejected(larger_data):
    with pytest.raises(ValueError, match="exceeds"):
        protocol.make_protocol(audit.verify(larger_data), larger_data, "all", 4, ".4", selection_note="test")
