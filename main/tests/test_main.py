from copy import deepcopy
from importlib import import_module
from pathlib import Path
import subprocess
from threading import Event

import numpy as np
import pytest
import torch

from main import shared
from main.tasks import registry, physics_tasks
from main.data import generate, storage, audit
from main.training import protocol, sampler, train, queue, main
from main.evaluation import evaluate, analyze


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    out = tmp_path_factory.mktemp("main-data") / "frozen"
    config = {"tasks": ["W02", "G02"], "episodes": 2, "val_episodes": 2, "test_scenes": 2,
              "seed": 37, "max_attempts": 100}
    generate.generate(out, config)
    return out


@pytest.fixture
def study(generated, tmp_path):
    manifest = audit.verify(generated)
    overrides = {"dim": 12, "depth": 1, "heads": 3, "batch": 4, "val_every": 1, "log_every": 1}
    p = protocol.make_protocol(manifest, generated, ["W02", "G02"], 2, "0.4", 2, (0,),
                               overrides=overrides, selection_note="test fixture, not a selected scientific setting")
    out = protocol.prepare(tmp_path / "run", p, manifest)
    return out, p, protocol.jobs(p)


@pytest.mark.parametrize("task", list(registry.TASKS)[1:])
def test_all_sixteen_physics_tasks_form_and_generate_four_conditions(task):
    result = None
    for attempt in range(100):
        spec = physics_tasks.sample_spec(task, 7, "train", 0, attempt)
        try:
            result = physics_tasks.episode(spec)
            break
        except ValueError:
            pass
    assert result is not None, task
    arrays, meta, _, final = result
    assert len(registry.TASKS) == 17
    for c in shared.CONDITIONS:
        assert all(a.shape == (64, 28, 56) and a.dtype == np.uint8 for a in arrays[c].values())
    for k in storage.KEYS:
        assert np.array_equal(arrays["state"][k][:32], arrays["process"][k][32:])
        assert np.array_equal(arrays["shuffle"][k][32:], arrays["process"][k][32:])
        assert np.array_equal(arrays["shuffle"][k][:32], arrays["process"][k][meta["shuffle_order"]])
    material = shared.core.WATER if registry.TASKS[task]["family"] == "water" else shared.core.SAND
    opposite = shared.core.SAND if material == shared.core.WATER else shared.core.WATER
    assert material in final[0] and opposite not in final[0]
    assert np.count_nonzero(final[0] == material) == spec["fill"]
    assert sum(a["type"] == "place" and a["material"] == material for f in result[2] for a in f[2]) == spec["fill"]
    bench = physics_tasks.bench_item(spec, *final)
    assert bench["act"].shape == (2, 32, 28, 56) and bench["target_fg"].shape == (2, 28, 56)
    for j in range(2):
        world = shared.core.FormWorld(seed=shared.D.int_seed(spec["seed"], "bench", j), grid=final[0], background=final[1])
        acts = physics_tasks.intervention(spec, *final, shared.D.int_seed(spec["seed"], "bench", j))
        for step in range(32):
            physics_tasks.tick(world, acts if step == 0 else [])
        assert np.array_equal(world.grid, bench["target_fg"][j])


def test_task_registry_and_location_boundaries():
    assert registry.TASKS["T01"]["template"] == "legacy03"
    assert len(set((t["family"], t["template"]) for t in registry.TASKS.values())) == 17
    for task in list(registry.TASKS)[1:]:
        specs = [physics_tasks.sample_spec(task, 1, split, 0) for split in ("train", "validation", "test")]
        assert specs[0]["x"] < specs[1]["x"] < specs[2]["x"]


@pytest.mark.parametrize("tasks", [[], ["W01", "W01"], ["unknown"]])
def test_bad_tasks(tasks):
    with pytest.raises(ValueError):
        registry.select(tasks)


def test_dataset_frozen_and_splits_separated(generated):
    manifest = audit.verify(generated)
    for task in manifest["tasks"]:
        hashes = [{m["final_hash"] for m in shared.load_json(generated / task / f"{s}_scenes.json")}
                  for s in ("train", "validation", "test")]
        assert not hashes[0] & hashes[1] and not hashes[0] & hashes[2] and not hashes[1] & hashes[2]
        data = storage.Dataset(generated, task, "train", "process", 1)
        assert data.count == 1 and isinstance(data.arrays["fg"], np.memmap)
        with pytest.raises(ValueError, match="exceeds"):
            storage.Dataset(generated, task, "train", "process", 3)


def test_dry_run_writes_nothing(tmp_path):
    out = tmp_path / "absent"
    config = {"tasks": ["G01"], "episodes": 2, "val_episodes": 2, "test_scenes": 2, "seed": 0, "max_attempts": 100}
    generate.generate(out, config, dry_run=True)
    assert not out.exists()


def test_data_generation_never_starts_training(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("generation launched subprocess"))
    out = tmp_path / "data"
    config = {"tasks": ["G01"], "episodes": 1, "val_episodes": 1, "test_scenes": 1, "seed": 17, "max_attempts": 100}
    generate.generate(out, config)
    assert (out / "complete.json").is_file()
    assert not (out / "train").exists()
    assert generate.generate(out, config) == out
    changed = {**config, "episodes": 2}
    with pytest.raises(ValueError, match="changed"):
        generate.generate(out, changed)


def test_tampered_dataset_refused(tmp_path):
    out = tmp_path / "data"
    config = {"tasks": ["G01"], "episodes": 1, "val_episodes": 1, "test_scenes": 1, "seed": 12, "max_attempts": 100}
    generate.generate(out, config)
    path = out / "G01/train/state_fg.npy"
    data = np.load(path, mmap_mode="r+")
    data[0, 0, 0, 0] = 1
    data.flush()
    with pytest.raises(ValueError, match="changed"):
        audit.verify(out)


def test_partial_data_never_silently_overwritten(tmp_path):
    out = tmp_path / "data"
    (out / "G01").mkdir(parents=True)
    (out / "G01/user.txt").write_text("keep")
    config = {"tasks": ["G01"], "episodes": 1, "val_episodes": 1, "test_scenes": 1, "seed": 0, "max_attempts": 100}
    with pytest.raises(ValueError, match="uncommitted"):
        generate.generate(out, config)
    assert (out / "G01/user.txt").read_text() == "keep"


@pytest.mark.parametrize("value", ["../x", "", ".", "..", "a/b"])
def test_bad_name(value):
    with pytest.raises(ValueError):
        shared.name(value)


def test_symlink_output_rejected(tmp_path):
    target = tmp_path / "real"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic"):
        shared.safe_path(link)


def test_lock_prevents_concurrent_use(tmp_path):
    path = tmp_path / "pipeline.lock"
    with shared.lock(path):
        with pytest.raises(ValueError, match="another process"):
            with shared.lock(path):
                pass


def test_legacy_streaming_import_preserves_exact_prefix(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    arrays = {"fg": np.zeros((3, 64, 28, 56), np.uint8), "bg": np.zeros((3, 64, 28, 56), np.uint8)}
    arrays["fg"][1] = 2
    np.savez_compressed(source / "train_state.npz", **arrays)
    actions = [[[] for _ in range(64)] for _ in range(3)]
    actions[1][4] = [physics_tasks.action("place", 2, 3, 4)]
    shared.write_json(source / "train_state_actions.json", {"actions": actions})
    imported = storage.import_legacy(tmp_path / "import", source, 2, ("state",))
    assert np.array_equal(imported["state"]["fg"], arrays["fg"][:2])
    assert np.array_equal(imported["state"]["act"][1, 4], shared.common.encode_actions(actions[1][4]))
    with pytest.raises(ValueError):
        storage.stream_npz_prefix(source / "train_state.npz", "fg", imported["state"]["fg"], 4)


def test_jobs_include_single_and_mixed_four_conditions(study):
    out, p, jobs = study
    assert len(jobs) == 12
    for row in jobs:
        assert row["condition"] in shared.CONDITIONS
        assert row["steps"] == (4 if row["mode"] == "mixed" else 2)
    assert protocol.verify_run(out)[0] == p
    assert not (out / "train").exists()


def test_seventeen_tasks_have_72_default_single_seed_jobs(generated):
    manifest = audit.verify(generated)
    manifest["tasks"] = registry.TASKS
    manifest["task_data"] = {t: {"train_episodes": 2} for t in registry.TASKS}
    p = protocol.make_protocol(manifest, generated, "all", 2, "0.4", selection_note="test")
    rows = protocol.jobs(p)
    assert p["training_seeds"] == [0]
    assert len(rows) == 72 and len({r["id"] for r in rows}) == 72
    assert sum(r["mode"] == "mixed" for r in rows) == 4
    assert {r["seed"] for r in rows} == {0}


@pytest.mark.parametrize("r", [0, 1, -1, "nan", "inf", 2])
def test_main_ratio_must_be_mixed(generated, r):
    with pytest.raises(ValueError):
        protocol.make_protocol(audit.verify(generated), generated, "all", 2, r, selection_note="test")


def test_setting_selection_record_required(generated):
    with pytest.raises(ValueError, match="record"):
        protocol.make_protocol(audit.verify(generated), generated, "all", 2, "0.4")


def test_task_phase_sampler_exact_reproducible(study):
    _, p, jobs = study
    row = next(r for r in jobs if r["mode"] == "mixed" and r["condition"] == "process")
    first, second = sampler.Sampler(p, row), sampler.Sampler(p, row)
    for _ in range(13):
        a, b = first.batch(7), second.batch(7)
        assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert first.total == 91 and first.seen == {"W02": 46, "G02": 45}
    assert first.formed == {"W02": shared.quota("0.4", 46), "G02": shared.quota("0.4", 45)}
    assert not np.isin(sampler.MIXED_OPERATION, [0, 32, 40, 48, 56]).any()
    assert len(sampler.FORMATION) == 31 and len(sampler.MIXED_OPERATION) == 28 and len(sampler.STATE_OPERATION) == 56
    state = sampler.Sampler(p, {**row, "condition": "state"})
    state.batch(7)
    assert not any(state.formed.values())


@pytest.mark.parametrize("mode", ["single", "mixed"])
@pytest.mark.parametrize("condition", shared.CONDITIONS)
def test_real_cpu_training_evaluation_and_resume(study, mode, condition):
    out, p, jobs = study
    row = next(r for r in jobs if r["mode"] == mode and r["condition"] == condition)
    done = train.train_job(out, p, row, "cpu")
    folder = out / "train" / row["id"]
    ck = torch.load(folder / "ckpt.pt", map_location="cpu", weights_only=True)
    assert ck["step"] == row["steps"] and ck["selection"] == "last" and ck["weights"] == "ema"
    assert done["samples_seen"] == row["steps"] * 4
    assert train.validate_done(folder, p, row) == done
    before = shared.file_hash(folder / "ckpt.pt")
    assert train.train_job(out, p, row, "cpu") == done
    assert shared.file_hash(folder / "ckpt.pt") == before
    evaluate.evaluate_job(out, p, row, "cpu", batch=2)
    for task in row["tasks"]:
        samples = evaluate.validate_eval(folder, p, row, task)
        assert set(shared.METRICS).issubset(samples)
        assert samples["change_f1"].shape == (2, 2)
    evaluate.evaluate_job(out, p, row, "cpu", batch=2)
    assert shared.file_hash(folder / "ckpt.pt") == before


def test_corrupt_completed_training_refused(study):
    out, p, rows = study
    row = rows[0]
    train.train_job(out, p, row, "cpu")
    folder = out / "train" / row["id"]
    with (folder / "ckpt.pt").open("ab") as stream:
        stream.write(b"x")
    with pytest.raises(ValueError, match="identity"):
        train.validate_done(folder, p, row)


def test_modified_run_budget_refused(study):
    out, p, _ = study
    p = deepcopy(p)
    p["steps_single"] = 3
    shared.write_json(out / "protocol.json", p)
    with pytest.raises(ValueError):
        protocol.verify_run(out)


def test_final_only_preflight_no_partial_child_launch(study, monkeypatch):
    out, p, rows = study
    monkeypatch.setattr(queue, "verify_run", lambda *a: (p, {}))
    calls = []
    def validate(path, protocol, row):
        calls.append(row["id"])
        if row == rows[2]:
            raise ValueError("missing final checkpoint")
        return {}
    monkeypatch.setattr(queue, "validate_done", validate)
    monkeypatch.setattr(queue.subprocess, "Popen", lambda *a, **k: pytest.fail("child launched"))
    with pytest.raises(ValueError, match="missing"):
        queue.execute(out, evaluate_only=True)
    assert calls == [r["id"] for r in rows[:3]]


def test_eval_only_never_falls_back_to_training(study, monkeypatch):
    out, p, rows = study
    monkeypatch.setattr(queue.subprocess, "Popen", lambda *a, **k: pytest.fail("training launched"))
    with pytest.raises(ValueError, match="cannot train"):
        queue.run_job(out, p, rows[0], "cpu", True, Event(), .01)


def test_shared_metrics_perfect_and_empty_change():
    fg = shared.D.terrain()
    bg = np.zeros_like(fg)
    score = shared.score((fg, bg), (fg, bg), (fg, bg))
    assert set(score) == set(shared.METRICS)
    assert score["change_f1"] == score["cell_acc_mask"] == 1


def test_paired_statistics_seed_std_and_task_macro(generated, tmp_path):
    manifest = audit.verify(generated)
    p = protocol.make_protocol(manifest, generated, "all", 2, "0.4", selection_note="test")
    p["training_seeds"] = [0, 1, 2]
    samples = {}
    for mode in p["modes"]:
        for task in p["tasks"]:
            for c in shared.CONDITIONS:
                for s in p["training_seeds"]:
                    value = .4 + .01 * s + (.1 if c == "process" else 0)
                    samples[(mode, task, c, s)] = {m: np.full((4, 2), value) for m in shared.METRICS}
    report = analyze.summarize(samples, p, boot=100)
    for block in report["by_mode"].values():
        for task in p["tasks"]:
            metric = block["tasks"][task]["change_f1"]
            assert metric["means"]["process"]["seed_std"] == pytest.approx(.01)
            assert metric["contrasts"]["process_minus_state"]["mean"] == pytest.approx(.1)
            assert metric["contrasts"]["process_minus_state"]["n_cases"] == 4
        assert block["equal_task_macro"]["change_f1"]["process"] == pytest.approx(.51)
    text = analyze.markdown(report)
    assert "单任务训练是核心" in text and "first-fixed-point" in text
    analyze.plot(report, tmp_path)
    assert all((tmp_path / f"{mode}.{ext}").exists() for mode in p["modes"] for ext in ("png", "pdf"))


def test_missing_scores_not_treated_as_zero(generated):
    p = protocol.make_protocol(audit.verify(generated), generated, "all", 2, "0.4", selection_note="test")
    with pytest.raises(KeyError):
        analyze.summarize({}, p, boot=10)


def test_prepare_cli_and_resume_parameter_rejection(generated, tmp_path):
    out = tmp_path / "cli-run"
    args = ["--stage", "prepare", "--run", str(out), "--data", str(generated), "--episodes", "2",
            "--formation-ratio", "0.4", "--selection-note", "test fixture"]
    main.main(args + ["--dry-run"])
    assert not out.exists()
    main.main(args)
    assert len(shared.load_json(out / "jobs.json")) == 12
    main.main(args)
    main.main(["--stage", "train", "--run", str(out), "--dry-run"])
    with pytest.raises(SystemExit):
        main.main(["--stage", "train", "--run", str(out), "--steps", "1", "--dry-run"])


def test_parallel_generation_cli_and_all_tasks_round_trip(tmp_path):
    import os
    out = tmp_path / "all-physics"
    tasks = list(registry.TASKS)[1:]
    command = [__import__("sys").executable, "-m", "main.data.generate", "--out", str(out),
               "--tasks", ",".join(tasks), "--episodes", "1", "--val-episodes", "1", "--test-scenes", "1",
               "--seed", "71", "--workers", "2"]
    completed = subprocess.run(command, cwd=shared.PROJECT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    manifest = audit.verify(out)
    assert len(manifest["tasks"]) == 16
    assert all(m["train_validation_overlap"] == m["train_test_overlap"] == m["validation_test_overlap"] == 0
               for m in manifest["task_data"].values())


def test_real_legacy_anchor_import(tmp_path):
    if not (shared.DEFAULT_SOURCE / "audit.json").is_file():
        pytest.skip("current experiment03 source unavailable")
    out = tmp_path / "legacy"
    config = {"tasks": ["T01"], "episodes": 1, "val_episodes": 1, "test_scenes": 1, "seed": 1, "max_attempts": 100}
    generate.generate(out, config)
    manifest = audit.verify(out)
    assert manifest["task_data"]["T01"]["test_scenes"] == 512
    source_audit = shared.load_json(shared.DEFAULT_SOURCE / "audit.json")
    source_path = Path(source_audit["inputs"]["data"])
    if not source_path.is_absolute():
        source_path = shared.PROJECT / source_path
    source = shared.common.load_condition(source_path, "process", 1)
    imported = storage.Dataset(out, "T01", "train", "process")
    assert np.array_equal(imported.arrays["fg"], source["fg"])
    assert np.array_equal(imported.arrays["bg"], source["bg"])
    val_path = Path(source_audit["inputs"]["val"])
    if not val_path.is_absolute():
        val_path = shared.PROJECT / val_path
    assert storage.legacy_count(source_path) == 81920 and storage.legacy_count(val_path) == 32
    dry_out = tmp_path / "default-readonly"
    generate.main(["--out", str(dry_out), "--episodes", "5120", "--dry-run"])
    assert not dry_out.exists()
    with pytest.raises(ValueError, match="exceeds"):
        generate.main(["--out", str(dry_out), "--episodes", "5120", "--val-episodes", "33", "--dry-run"])
    assert not dry_out.exists()


@pytest.mark.parametrize("task", ["W02", "G02"])
def test_container_geometry_does_not_collapse(task):
    spec = physics_tasks.sample_spec(task, 1, "train", 0)
    spec["width"] = 10
    _, _, _, final = physics_tasks.episode(spec)
    for a in physics_tasks.geometry(spec):
        if a["type"] == "place":
            assert final[0][a["y"], a["x"]] == a["material"]
    bench = physics_tasks.bench_item(spec, *final)
    for j in range(2):
        expected = final[0].copy()
        acts = physics_tasks.intervention(spec, *final, shared.D.int_seed(spec["seed"], "bench", j))
        for a in acts:
            expected[a["y"], a["x"]] = 0
        solid = ~np.isin(expected, [shared.core.EMPTY, shared.core.SAND, shared.core.WATER])
        assert np.array_equal(bench["target_fg"][j][solid], expected[solid])
        predicted_solid = ~np.isin(bench["target_fg"][j], [shared.core.EMPTY, shared.core.SAND, shared.core.WATER])
        assert np.array_equal(solid, predicted_solid)


def test_data_output_rejects_unrelated_files_before_any_write(tmp_path):
    out = tmp_path / "unrelated"
    out.mkdir()
    (out / "user.txt").write_text("keep")
    config = {"tasks": ["G01"], "episodes": 1, "val_episodes": 1, "test_scenes": 1, "seed": 0, "max_attempts": 100}
    with pytest.raises(ValueError, match="unrelated"):
        generate.generate(out, config)
    assert {p.name for p in out.iterdir()} == {"user.txt"}


def test_complete_single_mixed_study_publishes_analysis(study):
    out, p, rows = study
    for row in rows:
        train.train_job(out, p, row, "cpu")
        evaluate.evaluate_job(out, p, row, "cpu", batch=2)
    result = analyze.analyze(out, boot=20)
    report = shared.load_json(result / "results.json")
    assert set(report["by_mode"]) == {"single", "mixed"}
    assert all(set(v["tasks"]) == {"W02", "G02"} for v in report["by_mode"].values())
    assert len(report["identities"]) == 16
    assert (result / "results.md").is_file() and (result / "figs/mixed.pdf").is_file()


@pytest.mark.parametrize("task", ["W08", "G08"])
def test_ledge_fill_is_exact_and_oversized_requests_rejected(task):
    for index in range(20):
        spec = physics_tasks.sample_spec(task, 11, "train", index)
        assert 4 <= spec["fill"] <= 12
    spec["fill"] = 12
    frames, world = physics_tasks.formation(spec)
    material = shared.core.WATER if task == "W08" else shared.core.SAND
    assert np.count_nonzero(world.grid == material) == 12
    assert sum(a["material"] == material for f in frames for a in f[2] if a["type"] == "place") == 12
    spec["fill"] = 13
    with pytest.raises(ValueError, match="declared material count"):
        physics_tasks.formation(spec)


@pytest.mark.parametrize("relative", ["preresearch/03_main/exp/audit.py", "preresearch/03_main/exp/shared.py",
                                      "preresearch/04_ood_generalization/exp/statistics.py"])
def test_dependency_changes_invalidate_frozen_run(study, monkeypatch, relative):
    out, _, _ = study
    before = shared.code_identity()
    assert relative in before["dependencies"]
    original = shared.file_hash
    target = shared.PROJECT / relative
    monkeypatch.setattr(shared, "file_hash", lambda path: "changed" if Path(path).resolve() == target else original(path))
    assert shared.code_identity()["dependencies"][relative] != before["dependencies"][relative]
    with pytest.raises(ValueError, match="code identity"):
        protocol.verify_run(out)


@pytest.mark.parametrize("variable", ["FW_PROJECT", "FW_GAME_CODE"])
def test_external_game_override_rejected_at_import(tmp_path, variable):
    import os
    env = {**os.environ, variable: str(tmp_path / "external"), "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([__import__("sys").executable, "-c", "import main.shared"],
                            cwd=shared.PROJECT, env=env, capture_output=True, text=True)
    assert result.returncode != 0 and variable + " must point to this repository" in result.stderr


@pytest.mark.parametrize("module", [shared.core, shared.D])
def test_cached_physics_module_origin_rejected(tmp_path, monkeypatch, module):
    monkeypatch.setattr(module, "__file__", str(tmp_path / "shadow.py"))
    with pytest.raises(ValueError, match="actual module source"):
        shared.code_identity()


@pytest.mark.parametrize("relative", ["W02/train/state_fg.npy", "W02/validation/state_bg.npy", "W02/bench.npz"])
def test_completed_single_job_reuse_rechecks_inputs(generated, tmp_path, relative):
    import shutil
    data = tmp_path / "copy"
    shutil.copytree(generated, data)
    overrides = {"dim": 12, "depth": 1, "heads": 3, "batch": 4, "val_every": 1, "log_every": 1}
    manifest = audit.verify(data)
    p = protocol.make_protocol(manifest, data, "all", 2, "0.4", 1, (0,), overrides=overrides, selection_note="test")
    run = protocol.prepare(tmp_path / "run", p, manifest)
    row = protocol.jobs(p)[0]
    train.train_job(run, p, row, "cpu")
    if relative.endswith(".npy"):
        values = np.load(data / relative, mmap_mode="r+")
        values[0, 0, 0, 0] = (int(values[0, 0, 0, 0]) + 1) % 21
        values.flush()
        del values
    else:
        with np.load(data / relative, allow_pickle=False) as z:
            values = {k: z[k] for k in z.files}
        values["ctx_fg"][0, 0, 0] = (int(values["ctx_fg"][0, 0, 0]) + 1) % 21
        np.savez_compressed(data / relative, **values)
    assert protocol.verify_run(run, full_data=False)[0] == p
    with pytest.raises(ValueError, match="task input files changed"):
        train.main(["--run", str(run), "--job", row["id"], "--device", "cpu"])


def test_prepare_recovers_after_only_pipeline_lock_remains(generated, tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "pipeline.lock").touch()
    main.main(["--stage", "prepare", "--run", str(out), "--data", str(generated), "--episodes", "2",
               "--formation-ratio", "0.4", "--selection-note", "test fixture", "--seeds", "0"])
    assert len(shared.load_json(out / "jobs.json")) == 12
    assert protocol.verify_run(out)[0]["training_seeds"] == [0]


@pytest.mark.parametrize("stage", ["manifest", "manifest_tmp", "complete_tmp"])
def test_data_final_commit_interruption_recovers_without_regeneration(tmp_path, monkeypatch, stage):
    out = tmp_path / "data"
    config = {"tasks": ["G01"], "episodes": 1, "val_episodes": 1, "test_scenes": 1, "seed": 27, "max_attempts": 100}
    original_json, original_frozen = generate.write_json, generate.frozen_write
    def frozen(path, value):
        if stage == "manifest_tmp" and Path(path).name == "manifest.json":
            (out / "manifest.json.tmp").write_text("partial")
            raise RuntimeError("simulated interruption")
        return original_frozen(path, value)
    def write(path, value):
        if Path(path).name == "complete.json":
            if stage == "complete_tmp":
                (out / "complete.json.tmp").write_text("partial")
            raise RuntimeError("simulated interruption")
        return original_json(path, value)
    with monkeypatch.context() as m:
        m.setattr(generate, "write_json", write)
        m.setattr(generate, "frozen_write", frozen)
        with pytest.raises(RuntimeError, match="interruption"):
            generate.generate(out, config)
    assert (out / "G01/task_complete.json").is_file() and not (out / "complete.json").exists()
    before = shared.files_identity(out / "G01")
    monkeypatch.setattr(generate, "build_task", lambda *a: pytest.fail("regenerated committed task"))
    generate.generate(out, config)
    assert shared.files_identity(out / "G01") == before
    assert audit.verify(out)["config"] == config
    assert not list(out.glob("*.tmp"))


@pytest.mark.parametrize("completed", [True, False])
def test_t01_source_audit_changes_reject_same_path_reuse(tmp_path, monkeypatch, completed):
    source, out = tmp_path / "source", tmp_path / "data"
    source.mkdir()
    paths = {k: tmp_path / k for k in ("data", "val", "bench")}
    paths["data"] = tmp_path / "source-data"
    audited = {"inputs": {k: str(v) for k, v in paths.items()}, "hashes": {k: {"file": "original"} for k in paths},
               "protocol": {"episodes": 81920}, "train_validation_overlap": 0, "train_bench_overlap": 0,
               "validation_bench_overlap": 0}
    shared.write_json(source / "audit.json", audited)
    api = import_module("preresearch.03_main.exp.audit")
    monkeypatch.setattr(api, "verify_inputs", lambda a: paths)
    monkeypatch.setattr(generate.common, "load_summary", lambda p: {"profile": "simple_to_complex"})
    monkeypatch.setattr(generate, "legacy_count", lambda p: 81920 if p == paths["data"] else 32)
    monkeypatch.setattr(generate, "legacy_task", lambda *a: {"test_scenes": 512})
    monkeypatch.setattr(generate, "verify", lambda p: shared.load_json(Path(p) / "manifest.json"))
    config = {"tasks": ["T01"], "episodes": 1, "val_episodes": 1, "test_scenes": 1, "seed": 1, "max_attempts": 100}
    generate.generate(out, config, source)
    request = shared.load_json(out / "request.json")
    assert request["source_identity"]["audit_sha256"] == shared.file_hash(source / "audit.json")
    assert request["source_identity"]["hashes"] == audited["hashes"]
    if not completed:
        (out / "complete.json").unlink()
    before = shared.files_identity(out)
    audited["revision"] = "new legal audit at the same path"
    shared.write_json(source / "audit.json", audited)
    monkeypatch.setattr(generate, "build_task", lambda *a: pytest.fail("source changed but task launched"))
    with pytest.raises(ValueError, match="request changed"):
        generate.generate(out, config, source)
    assert shared.files_identity(out) == before


def test_migrated_source_and_result_roots():
    assert shared.ROOT == shared.PROJECT / "main"
    assert shared.RESULTS_ROOT == shared.PROJECT / "experiments/main"
    assert shared.DEFAULT_SOURCE == shared.PROJECT / "experiments/preresearch/03_main/20261111-044057_four_conditions"
    assert protocol.run_root(shared.RESULTS_ROOT / "runs/new") == shared.RESULTS_ROOT / "runs/new"
    assert shared.code_identity()["main"]["shared.py"] == shared.file_hash(shared.ROOT / "shared.py")


@pytest.mark.parametrize("relative", ["datasets/bad", "runs/bad"])
def test_source_tree_cannot_receive_generated_results(relative):
    target = shared.ROOT / relative
    config = {"tasks": ["G01"], "episodes": 1, "val_episodes": 1, "test_scenes": 1, "seed": 1, "max_attempts": 100}
    assert not target.exists()
    with pytest.raises(ValueError, match="overlaps"):
        generate.generate(target, config, dry_run=True)
    with pytest.raises(ValueError, match="overlaps"):
        protocol.run_root(target)
    assert not target.exists()


@pytest.mark.parametrize("script", ["generate_data.sh", "run_main.sh"])
def test_migrated_shell_entry_from_unrelated_directory(tmp_path, script):
    import os, sys
    env = {**os.environ, "PY": sys.executable, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run(["bash", str(shared.ROOT / script), "--help"], cwd=tmp_path,
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--" in result.stdout


def test_migrated_cpu_queue_launches_real_new_package(study, monkeypatch):
    out, p, rows = study
    row = rows[0]
    calls = []
    real_popen = queue.subprocess.Popen
    def launch(command, **kwargs):
        calls.append(command)
        return real_popen(command, **kwargs)
    monkeypatch.setattr(queue.subprocess, "Popen", launch)
    assert queue.run_job(out, p, row, "cpu", False, Event(), .1)
    assert [command[2] for command in calls] == ["main.training.train", "main.evaluation.evaluate"]
    folder = out / "train" / row["id"]
    train.validate_done(folder, p, row)
    for task in row["tasks"]:
        evaluate.validate_eval(folder, p, row, task)
    before = shared.files_identity(folder)
    assert queue.run_job(out, p, row, "cpu", True, Event(), .1)
    assert len(calls) == 2
    assert shared.files_identity(folder) == before


@pytest.mark.parametrize("seeds", [(), (0, 1), (0, 1, 2), (0, 0), (-1,), (True,)])
def test_main_training_rejects_invalid_or_multiple_seeds(generated, tmp_path, seeds):
    manifest = audit.verify(generated)
    with pytest.raises(ValueError):
        protocol.make_protocol(manifest, generated, "all", 2, "0.4", seeds=seeds, selection_note="test")
    p = protocol.make_protocol(manifest, generated, "all", 2, "0.4", selection_note="test")
    row = protocol.jobs(p)[0]
    p["training_seeds"] = list(seeds)
    with pytest.raises(ValueError):
        protocol.jobs(p)
    run = tmp_path / "blocked"
    with pytest.raises(ValueError):
        protocol.prepare(run, p, manifest)
    with pytest.raises(ValueError):
        train.train_job(run, p, row, "cpu")
    assert not run.exists()


def test_main_prepare_defaults_to_seed_zero_and_accepts_one_other_seed(generated, tmp_path):
    run = tmp_path / "default"
    main.main(["--stage", "prepare", "--run", str(run), "--data", str(generated), "--episodes", "2",
               "--formation-ratio", "0.4", "--selection-note", "test fixture"])
    p = shared.load_json(run / "protocol.json")
    rows = shared.load_json(run / "jobs.json")
    assert p["training_seeds"] == [0] and len(rows) == 12 and {r["seed"] for r in rows} == {0}
    other = protocol.make_protocol(audit.verify(generated), generated, "all", 2, "0.4", seeds=(7,), selection_note="test")
    assert {r["seed"] for r in protocol.jobs(other)} == {7}
    with pytest.raises(ValueError, match="job seed differs"):
        train.train_job(tmp_path / "wrong-job", other, rows[0], "cpu")
    assert not (tmp_path / "wrong-job").exists()


@pytest.mark.parametrize("seeds", ["0,1", "0,1,2", ""])
def test_cli_rejects_multiseed_before_reading_data(tmp_path, monkeypatch, seeds):
    monkeypatch.setattr(main, "verify", lambda *a: pytest.fail("invalid seed read dataset"))
    with pytest.raises(ValueError):
        main.main(["--stage", "prepare", "--run", str(tmp_path / "blocked"), "--data", str(tmp_path / "missing"),
                   "--episodes", "2", "--formation-ratio", "0.4", "--selection-note", "test", "--seeds", seeds])
    assert not (tmp_path / "blocked").exists()


def test_frozen_multiseed_run_rejected_before_dataset_access(study, monkeypatch):
    out, p, _ = study
    p = deepcopy(p)
    p["training_seeds"] = [0, 1, 2]
    shared.write_json(out / "protocol.json", p)
    monkeypatch.setattr(protocol, "verify", lambda *a, **k: pytest.fail("multiseed resume read dataset"))
    with pytest.raises(ValueError, match="exactly one"):
        protocol.verify_run(out)
