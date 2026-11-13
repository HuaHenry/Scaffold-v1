from copy import deepcopy
from importlib import import_module
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from exp import analysis, audit, evaluate_final, generate_data, main, make_jobs, run_queue, shared, train_ratio as trainer
import datasets as D
import datasets_v3 as v3


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def write_small_data(path, episodes=2):
    path.mkdir()
    formed, operated, scenes = [], [], []
    for i in range(episodes):
        actions = [D.act_dict(D.place(10 + i, D.FLOOR - j, D.STONE)) for j in range(5)]
        script = [[(v3.to_action(a), "build")] for a in actions] + [[] for _ in range(22)]
        snapshots, world, ok = v3.run_formation(script, D.int_seed(0, "world", i), 32)
        assert ok
        f = D.pack(snapshots, [0] * 32)
        operation = D.op_clip(world.grid, world.background, D.Action("remove", 10 + i, D.FLOOR), 0)
        o = D.pack(operation * 8, [1] * 64)
        formed.append(f)
        operated.append(o)
        scenes.append({"kind": "tower", "n_struct": 1, "final_hash": D.state_hash(world.grid, world.background)})
    arrays = {key: np.stack([np.concatenate((f[key], o[key][:32])) for f, o in zip(formed, operated)])
              for key in ("fg", "bg", "phase", "cont")}
    arrays["phase"] = arrays["phase"].astype(np.int8)
    shared.write_npz(path / "train_process.npz", arrays)
    shared.write_json(path / "train_process_actions.json", {"scenes": ["tower"] * episodes,
        "actions": [f["actions"] + o["actions"][:32] for f, o in zip(formed, operated)]})
    shared.write_npz(path / "train_state.npz", {k: np.stack([o[k] for o in operated]) for k in arrays})
    shared.write_json(path / "train_state_actions.json", {"actions": [o["actions"] for o in operated]})
    shared.write_json(path / "train_scenes.json", scenes)
    shared.write_json(path / "train_summary.json", {"layout": {"formation": 32, "clip": 8},
        "conditions": {"process": {"episodes": episodes}, "state": {"episodes": episodes}}})
    return arrays


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setattr(shared.common, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(shared, "TRAIN_DEFAULTS", {**shared.TRAIN_DEFAULTS, "dim": 12, "depth": 1, "heads": 3})
    data = tmp_path / "frozen"
    arrays = write_small_data(data)
    p = shared.protocol(steps=1000)
    p["episodes"] = 2
    inputs = {"data": str(data), "val": str(data), "bench": str(tmp_path / "bench.npz")}
    hashes = {"data": {f.name: shared.common.sha1_file(f) for f in data.iterdir()},
              "val": shared.common.data_hashes(data, "state"), "bench": {}}
    manifest = {"version": shared.VERSION, "source_run": "source03", "inputs": inputs, "hashes": hashes,
                "source_audit_sha1": "frozen", "protocol": p}
    bundle = {"source": tmp_path / "source03", "manifest": manifest,
              "inputs": {k: Path(v) for k, v in inputs.items()}, "audited": {"inputs": inputs, "hashes": hashes}}
    out = shared.bind_run("unit", bundle)
    return bundle, out, arrays


@pytest.mark.parametrize("ratio,n", [(0, 0), (.125, 32), (.25, 64), (.5, 128), (.75, 192)])
def test_equal_update_budget_and_exact_exposure(ratio, n):
    b = shared.budget(ratio)
    assert b["steps"] == 94400 and b["batch"] == 256
    assert b["samples"] == 24166400
    assert b["formation_per_batch"] == n
    assert b["formation_samples"] == 94400 * n
    assert b["formation_samples"] + b["operation_samples"] == b["samples"]
    assert b["epochs"] is None


@pytest.mark.parametrize("value", [-1, 1.1, np.nan, np.inf])
def test_invalid_ratios_are_rejected(value):
    with pytest.raises(ValueError):
        shared.formation_count(value)


@pytest.mark.parametrize("values", [[], [0], [.125, .5], [0, .5, .5], [.5, 0], [0, np.nan, .5]])
def test_invalid_ratio_design_is_not_silently_changed(values):
    with pytest.raises(ValueError):
        shared.validate_ratios(values)


@pytest.mark.parametrize("value", [0, -1, 1.5])
def test_invalid_update_budget(value):
    with pytest.raises(ValueError):
        shared.budget(.5, value)


@pytest.mark.parametrize("name", ["", ".", "..", "a/b", "../escape"])
def test_names_cannot_escape_output_root(name):
    with pytest.raises(ValueError):
        shared.run_path(name)


@pytest.mark.parametrize("ratio", [0, .125, .25, .5, .75, 1])
def test_stratified_sampler_exact_ratio_and_reproducibility(ratio):
    pools = (torch.arange(31), torch.arange(31, 59))
    g1, g2 = (torch.Generator().manual_seed(9) for _ in range(2))
    for _ in range(3):
        a = trainer.sample_ratio_batch(pools, 256, ratio, g1)
        b = trainer.sample_ratio_batch(pools, 256, ratio, g2)
        assert torch.equal(a, b)
        assert int((a < 31).sum()) == shared.formation_count(ratio)
        assert len(a) == 256 and int(a.min()) >= 0 and int(a.max()) < 59
    assert len(torch.unique(a)) < len(a)


def test_zero_ratio_does_not_require_a_formation_pool():
    pools = (torch.empty(0, dtype=torch.long), torch.arange(3))
    ids = trainer.sample_ratio_batch(pools, 16, 0, torch.Generator().manual_seed(0))
    assert len(ids) == 16
    with pytest.raises(ValueError, match="empty"):
        trainer.sample_ratio_batch(pools, 16, .5, torch.Generator())


def test_phases_are_checked_and_no_reset_pairs_are_drawn(source):
    bundle, _, _ = source
    data = trainer.pilot.Data(bundle["inputs"]["data"], "process", "video", 2, torch.device("cpu"))
    pools = trainer.phase_pools(data.pairs, data.phase)
    assert [len(p) for p in pools] == [62, 56]
    assert not torch.isin(data.pairs[:, 1], torch.tensor([0, 32, 40, 48, 56])).any()
    before = data.phase.clone()
    data.phase[data.pairs[0, 0], data.pairs[0, 1]] = 3
    with pytest.raises(ValueError, match="phases"):
        trainer.phase_pools(data.pairs, data.phase)
    data.phase.copy_(before)


def test_preparation_audits_original_data_without_frame_outputs(source):
    bundle, out, arrays = source
    before = {p.name: shared.common.sha1_file(p) for p in bundle["inputs"]["data"].iterdir()}
    audited = audit.audit_sampling(bundle, "unit")
    assert audited["pools"]["formation"] == 62 and audited["pools"]["operation"] == 56
    assert audited["pools"]["natural_formation_share"] == pytest.approx(31 / 59)
    assert audited["scene_distribution"]["scene_types"] == {"tower": 2}
    assert audited["inherited_action_encoding"]["collision_frames"] == 0
    assert audit.audit_sampling(bundle, "unit") == audited
    assert not (out / "data").exists() and not (out / "cache").exists()
    assert before == {p.name: shared.common.sha1_file(p) for p in bundle["inputs"]["data"].iterdir()}


@pytest.mark.parametrize("key", ["phase", "cont", "shape", "actions", "scenes"])
def test_population_rejects_invalid_original_boundaries_or_identity(source, key):
    bundle, _, arrays = source
    data = bundle["inputs"]["data"]
    if key == "phase":
        arrays["phase"][0, 1] = 1
        shared.write_npz(data / "train_process.npz", arrays)
    elif key == "cont":
        arrays["cont"][0, 32] = True
        shared.write_npz(data / "train_process.npz", arrays)
    elif key == "shape":
        arrays["fg"] = arrays["fg"][:1]
        shared.write_npz(data / "train_process.npz", arrays)
    elif key == "actions":
        payload = shared.common.load_json(data / "train_process_actions.json")
        payload["actions"][0].pop()
        shared.write_json(data / "train_process_actions.json", payload)
    else:
        payload = shared.common.load_json(data / "train_process_actions.json")
        payload["scenes"][0] = "changed"
        shared.write_json(data / "train_process_actions.json", payload)
    with pytest.raises(ValueError):
        audit.population(data, 2)


def test_inherited_same_cell_encoding_is_recorded_not_newly_merged(source):
    bundle, _, _ = source
    path = bundle["inputs"]["data"] / "train_process_actions.json"
    payload = shared.common.load_json(path)
    frame = payload["actions"][0][1]
    frame.append({**frame[0], "layer": "back"})
    shared.write_json(path, payload)
    before = path.read_bytes()
    result = audit.population(bundle["inputs"]["data"], 2)
    assert result["inherited_action_encoding"]["collision_frames"] == 1
    assert path.read_bytes() == before


def test_pool_fingerprints_match_actual_training_data(source):
    bundle, _, _ = source
    population = audit.population(bundle["inputs"]["data"], 2)
    data = trainer.pilot.Data(bundle["inputs"]["data"], "process", "video", 2, torch.device("cpu"))
    pools = trainer.phase_pools(data.pairs, data.phase)
    trainer.verify_pools(data, pools, population["pools"])
    broken = deepcopy(population["pools"])
    broken["pairs_sha1"] = "changed"
    with pytest.raises(ValueError, match="pools"):
        trainer.verify_pools(data, pools, broken)


def test_all_eleven_jobs_are_new_and_use_the_same_original_pools(source):
    bundle, out, _ = source
    audited = audit.audit_sampling(bundle, "unit")
    rows = make_jobs.save_jobs("unit", audited)
    assert [r["formation_ratio"] for r in rows] == list(shared.RATIOS)
    assert len({r["out"] for r in rows}) == 11
    for row in rows:
        assert row["cmd"][2] == "preresearch.05_formation_ratio.exp.train_ratio"
        assert row["cwd"] == str(shared.PROJECT)
        assert shared.resolve_file(row["out"]).is_relative_to(out / "train")
        expected = shared.training_args("unit", row["formation_ratio"], audited)
        assert expected["data"] == audited["inputs"]["data"]
        assert expected["steps"] == 1000 and expected["epochs"] is None
    assert make_jobs.save_jobs("unit", audited) == rows
    rows[0]["steps"] = 999
    shared.write_json(out / "jobs/formation_ratio.json", rows)
    with pytest.raises(ValueError, match="job list changed"):
        make_jobs.save_jobs("unit", audited)


def test_old_compression_run_is_rejected_without_overwrite(source):
    bundle, out, _ = source
    old = {"version": 1, "protocol": {"formation_mode": "compress", "levels": [0, 16, 32, 48]}}
    shared.write_json(out / "source.json", old)
    before = (out / "source.json").read_bytes()
    with pytest.raises(ValueError, match="retired"):
        shared.bind_run("unit", bundle)
    assert (out / "source.json").read_bytes() == before


def test_changed_sampling_budget_requires_new_run(source):
    bundle, out, _ = source
    bundle = deepcopy(bundle)
    bundle["manifest"]["protocol"]["steps"] += 1
    before = (out / "source.json").read_bytes()
    with pytest.raises(ValueError, match="budget changed"):
        shared.bind_run("unit", bundle)
    assert (out / "source.json").read_bytes() == before


def test_symlink_output_cannot_point_to_frozen_source(source):
    bundle, out, _ = source
    (out / "train").symlink_to(bundle["inputs"]["data"], target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        shared.bind_run("unit", bundle)
    with pytest.raises(ValueError, match="symbolic link"):
        make_jobs.jobs("unit", {"protocol": bundle["manifest"]["protocol"], "inputs": bundle["manifest"]["inputs"]})


def test_shared_run_lock_blocks_other_entrypoints(source, monkeypatch):
    _, out, _ = source
    with shared.run_lock(out):
        for module in (audit, make_jobs, run_queue, analysis):
            with pytest.raises(ValueError, match="another"):
                module.main(["--run", "unit"])


def completion_record(source, ratio=.2):
    bundle, _, _ = source
    audited = audit.audit_sampling(bundle, "unit")
    row = next(row for row in make_jobs.jobs("unit", audited) if row["formation_ratio"] == ratio)
    out = shared.resolve_file(row["out"])
    out.mkdir(parents=True)
    cfg = {"version": shared.VERSION, "args": shared.training_args("unit", ratio, audited), "pools": audited["pools"],
           "data_sha1": shared.expected_data_hashes(audited, "data"), "val_sha1": shared.expected_data_hashes(audited, "val")}
    shared.write_json(out / "config.json", cfg)
    model = trainer.models.build_model({k: cfg["args"][k] for k in ("dim", "depth", "heads")})
    torch.save({"model": model.state_dict(), "ema": model.state_dict(), "cfg": model.cfg,
        "step": 1000, "selection": "last", "weights": "ema", "args": cfg["args"]}, out / "ckpt.pt")
    b = shared.budget(ratio, 1000)
    done = {"version": shared.VERSION, "sampling_mode": shared.MODE, "steps": 1000, "epochs": None, "steps_per_epoch": None,
            "samples_seen": b["samples"], "formation_samples_seen": b["formation_samples"],
            "operation_samples_seen": b["operation_samples"], "formation_ratio": ratio, "form_frac_mean": ratio,
            "n_pairs": audited["pools"]["total"], "selection": "last", "checkpoint_step": 1000, "weights": "ema",
            "best_step": 500, "best_val": {"cascade_acc": .7}, "val": {"cascade_acc": .6},
            "ckpt_sha1": shared.common.sha1_file(out / "ckpt.pt")}
    shared.write_json(out / "done.json", done)
    return out, row, audited, cfg, done


def test_training_completion_contract(source):
    out, row, audited, _, done = completion_record(source)
    assert shared.validate_done(out, row, audited) == done


@pytest.mark.parametrize("key", ["formation_samples_seen", "operation_samples_seen", "steps", "epochs", "best_step", "ckpt_sha1"])
def test_incorrect_exposure_or_checkpoint_is_not_reused(source, key):
    out, row, audited, _, done = completion_record(source)
    done[key] = "changed" if key == "ckpt_sha1" else 1
    shared.write_json(out / "done.json", done)
    with pytest.raises(ValueError):
        shared.validate_done(out, row, audited)


@pytest.mark.parametrize("key", ["seed", "dim", "formation_ratio", "data", "steps"])
def test_changed_model_or_budget_settings_rejected(source, key):
    out, row, audited, cfg, _ = completion_record(source)
    cfg["args"][key] = "changed" if key == "data" else 99
    shared.write_json(out / "config.json", cfg)
    with pytest.raises(ValueError, match="configuration"):
        shared.validate_done(out, row, audited)


def tiny_fit(source, monkeypatch, ratio=.5):
    bundle, out, _ = source
    data = trainer.pilot.Data(bundle["inputs"]["data"], "process", "video", 2, torch.device("cpu"))
    val = trainer.pilot.Data(bundle["inputs"]["data"], "state", "video", 2, torch.device("cpu"))
    population = audit.population(bundle["inputs"]["data"], 2)
    args = SimpleNamespace(**{**shared.TRAIN_DEFAULTS, "dim": 12, "depth": 1, "heads": 3,
        "formation_ratio": ratio, "batch": 8, "steps": 3, "seed": 0, "log_every": 1, "val_every": 1})
    original_model = trainer.models.build_model
    monkeypatch.setattr(trainer.models, "build_model", lambda cfg: original_model({**cfg, "emb": 4}))
    target = out / "tiny"
    target.mkdir()
    config = {"version": shared.VERSION, "pools": population["pools"], "args": vars(args)}
    done = trainer.fit(data, val, args, target, config)
    return target, done, args


@pytest.mark.parametrize("ratio", [0, .125, .25, .5, .75])
def test_actual_cpu_training_ema_and_sample_counts(source, monkeypatch, ratio):
    out, done, args = tiny_fit(source, monkeypatch, ratio)
    assert done["samples_seen"] == 24
    assert done["formation_samples_seen"] == 24 * ratio
    assert done["operation_samples_seen"] == 24 * (1 - ratio)
    assert done["epochs"] is None
    checkpoint = torch.load(out / "ckpt.pt", map_location="cpu", weights_only=False)
    assert checkpoint["step"] == done["checkpoint_step"] == args.steps
    assert checkpoint["selection"] == "last" and checkpoint["weights"] == "ema"
    assert all(torch.isfinite(v).all() for v in checkpoint["ema"].values())
    logs = [json.loads(line) for line in (out / "train_log.jsonl").read_text().splitlines()]
    assert len([r for r in logs if r.get("type") == "val"]) == 3
    assert all(r["form_frac"] == ratio for r in logs if "form_frac" in r)


def test_validation_ties_do_not_select_early_ema_and_evaluator_loads_final(source, monkeypatch):
    monkeypatch.setattr(trainer.pilot, "transition_accuracy", lambda *a: {"cascade_acc": .5, "chg_acc": .5, "n_cascade": 2})
    out, done, _ = tiny_fit(source, monkeypatch)
    assert done["best_step"] == 1
    evaluator = import_module(f"{shared.PACKAGE}.evaluate")
    model, checkpoint = evaluator.load_run(out, "cpu")
    assert checkpoint["step"] == done["checkpoint_step"] == 3
    assert checkpoint["step"] != done["best_step"]
    assert checkpoint["selection"] == "last" and checkpoint["weights"] == "ema"
    assert model.cfg["dim"] == 12 and model.cfg["emb"] == 4
    initial = torch.zeros((1, 28, 56), dtype=torch.long)
    fg, bg = model.step(initial, initial, initial)
    assert fg.shape == initial.shape and bg.shape == initial.shape


def test_validation_failure_does_not_write_done(source, monkeypatch):
    monkeypatch.setattr(trainer.pilot, "transition_accuracy", lambda *a: {"cascade_acc": .5, "n_cascade": 0})
    with pytest.raises(ValueError, match="validation requires"):
        tiny_fit(source, monkeypatch)
    assert not (source[1] / "tiny/done.json").exists()


def synthetic_scores(ratios=shared.RATIOS):
    ids = np.repeat(np.arange(4), 8)
    samples = {r: {key: np.full(32, .2 + r * .2) for key in analysis.METRICS} for r in ratios}
    return samples, ids


def test_paired_statistics_and_supervision_trend():
    scores, ids = synthetic_scores()
    report = analysis.summarize(scores, ids, shared.common.SPLITS, boot=40, seed=9)
    metric = report["by_split"]["id"]["metrics"]["cascade_f1"]
    assert metric["trend"]["spearman"] == pytest.approx(1)
    assert metric["trend"]["slope_per_unit_fraction"] == pytest.approx(.2)
    value = metric["contrasts"][f"vs_{shared.dose_key(0)}"][shared.dose_key(.2)]
    assert value["mean"] == pytest.approx(.04)
    np.testing.assert_allclose(value["ci"], [.04, .04])
    assert metric["monotonic_increasing_means"] is True
    assert report["settings"]["seed_std"] is None
    assert report["settings"]["reuse_main_endpoints"] is False
    assert report == analysis.summarize(scores, ids, shared.common.SPLITS, boot=40, seed=9)
    assert analysis.spearman([0, 1, 2], [2, 2, 2]) is None


def test_nonmonotonic_and_undefined_rank_outcomes():
    scores, ids = synthetic_scores()
    scores[.2]["cascade_f1"][:] = .9
    scores[.8]["rank_acc"][:] = np.nan
    report = analysis.summarize(scores, ids, shared.common.SPLITS, boot=20)
    metric = report["by_split"]["id"]["metrics"]["cascade_f1"]
    assert metric["descriptive_best_ratio"] == .2
    assert metric["monotonic_increasing_means"] is False
    assert report["by_split"]["id"]["metrics"]["rank_acc"]["trend"]["n_cases"] == 0
    scores.pop(.1)
    with pytest.raises(ValueError, match="all selected ratios"):
        analysis.summarize(scores, ids, shared.common.SPLITS, boot=20)


@pytest.mark.parametrize("ratios", [shared.RATIOS, (0, .25, .5, .75)])
def test_report_and_plot_label_supervision_not_frames(tmp_path, ratios):
    scores, ids = synthetic_scores(ratios)
    report = analysis.summarize(scores, ids, shared.common.SPLITS, boot=20, ratios=ratios, steps=12000)
    text = analysis.markdown(report)
    assert "监督采样比例" in text and "12,000" not in text
    assert "F=16" not in text and "完整5轮" not in text
    assert ("12.5%" in text) == (.125 in ratios)
    assert "50.0% − 0.0%" in text
    analysis.plot(report, tmp_path)
    assert all((tmp_path / ("dose_response." + suffix)).stat().st_size > 100 for suffix in ("png", "pdf"))


def test_repeat_analysis_preserves_previous_report(source, monkeypatch):
    bundle, out, _ = source
    audited = audit.audit_sampling(bundle, "unit")
    scores, ids = synthetic_scores()
    shared.write_json(bundle["inputs"]["bench"].with_suffix(".json"), {"splits": list(shared.common.SPLITS)})
    monkeypatch.setattr(analysis, "verify_audit", lambda *a: audited)
    monkeypatch.setattr(analysis, "load_results", lambda *a: (scores, ids, {"r000": "new_model"}, {}))
    first = analysis.analyze("unit", boot=20)
    before = (first / "formation_ratio.json").read_bytes()
    second = analysis.analyze("unit", boot=20)
    assert first != second and second.is_relative_to(out)
    assert (first / "formation_ratio.json").read_bytes() == before


def test_analysis_never_reads_old_endpoints(source, monkeypatch):
    bundle, out, _ = source
    audited = audit.audit_sampling(bundle, "unit")
    audited["endpoints"] = {"0": "must_not_read", "32": "must_not_read"}
    monkeypatch.setattr(analysis.common, "load_bench", lambda *a: {"split_ids": np.zeros(512, int)})
    seen = []

    def validate(path, row, audited):
        seen.append(path)
        raise ValueError("new model missing")

    monkeypatch.setattr(analysis, "validate_done", validate)
    with pytest.raises(ValueError, match="new model missing"):
        analysis.load_results("unit", audited)
    assert seen == [out / "train/r000_seed0"]


def test_dry_run_never_creates_outputs_or_trains(source, monkeypatch):
    bundle, _, _ = source
    monkeypatch.delenv("FW_RUN", raising=False)
    monkeypatch.setattr(main, "validate_source", lambda value, ratios, steps: bundle)
    monkeypatch.setattr(main, "bind_run", lambda *a: pytest.fail("dry run should not write"))
    monkeypatch.setattr(main, "execute_queue", lambda *a: pytest.fail("dry run should not train"))
    before = sorted(shared.common.RESULTS_ROOT.rglob("*"))
    main.main(["--dry-run"])
    assert before == sorted(shared.common.RESULTS_ROOT.rglob("*"))


def test_prepare_pipeline_writes_audit_and_jobs_only(source, monkeypatch):
    bundle, out, _ = source
    monkeypatch.delenv("FW_RUN", raising=False)
    monkeypatch.setattr(main, "validate_source", lambda *a: bundle)
    monkeypatch.setattr(main, "execute_queue", lambda *a: pytest.fail("prepare must not train"))
    main.main(["--run", "unit", "--stage", "prepare"])
    assert (out / "audit.json").exists() and (out / "jobs/formation_ratio.json").exists()
    assert not (out / "cache").exists() and not (out / "data").exists()


def test_resume_inherits_ratio_source_and_step_budget(source, monkeypatch):
    bundle, _, _ = source
    calls = []

    def validate(value, ratios, steps):
        calls.append((value, ratios, steps))
        return bundle

    monkeypatch.setattr(main, "validate_source", validate)
    main.main(["--run", "unit", "--dry-run"])
    assert calls == [("source03", shared.RATIOS, 1000)]


@pytest.mark.parametrize("args", [["--levels", "0,16,32,48"], ["--workers", "4"], ["--stage", "preflight"]])
def test_retired_arguments_fail_before_source_loading(monkeypatch, args):
    monkeypatch.setattr(main, "validate_source", lambda *a: pytest.fail("old CLI should not run"))
    with pytest.raises(SystemExit):
        main.main(args)


def test_retired_data_generator_cannot_replay_or_write():
    with pytest.raises(SystemExit, match="retired"):
        generate_data.main([])


@pytest.mark.parametrize("cards", ["", "0,0", "0,", "-1", "not_a_gpu"])
def test_invalid_devices_fail_before_audit(monkeypatch, cards):
    monkeypatch.setattr(run_queue, "verify_audit", lambda *a: pytest.fail("should reject cards before audit"))
    with pytest.raises(ValueError):
        run_queue.execute_queue("unit", cards)


def test_completed_training_and_eval_skip_subprocess(source, monkeypatch):
    out, row, audited, _, done = completion_record(source)
    (out / "eval_complete.json").write_text("{}")
    monkeypatch.setattr(run_queue, "validate_eval", lambda *a: None)
    monkeypatch.setattr(run_queue.subprocess, "Popen", lambda *a, **k: pytest.fail("completed job must not rerun"))
    assert run_queue.run_job(row, "cpu", audited)


def test_invalid_completed_job_records_failure_without_retraining(source, monkeypatch):
    out, row, audited, _, done = completion_record(source)
    done["samples_seen"] = 0
    shared.write_json(out / "done.json", done)
    monkeypatch.setattr(run_queue.subprocess, "Popen", lambda *a, **k: pytest.fail("invalid completion must not be overwritten"))
    assert not run_queue.run_job(row, "cpu", audited)
    assert shared.common.load_json(out / "failed.json")["formation_ratio"] == .2


def test_queue_launches_new_trainer_and_existing_evaluator_from_project_root(source, monkeypatch):
    bundle, out, _ = source
    audited = audit.audit_sampling(bundle, "unit")
    row = make_jobs.jobs("unit", audited)[0]
    calls, checks = [], []

    class Process:
        returncode = 0

        def __init__(self, command, **kwargs):
            calls.append(command)
            assert kwargs["cwd"] == shared.PROJECT
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "2"

        def poll(self):
            return 0

    monkeypatch.setattr(run_queue.subprocess, "Popen", Process)
    monkeypatch.setattr(run_queue, "validate_done", lambda *a: {"ckpt_sha1": "selected"})
    monkeypatch.setattr(run_queue, "record_eval", lambda *a: checks.append("committed"))
    assert run_queue.run_job(row, "2", audited, interval=.01)
    assert [c[2] for c in calls] == ["preresearch.05_formation_ratio.exp.train_ratio", "preresearch.01_pilot_validity.exp.evaluate"]
    assert checks == ["committed"]


def test_uncommitted_eval_is_repeated_without_retraining(source, monkeypatch):
    out, row, audited, _, _ = completion_record(source)
    (out / "eval_samples.npz").write_bytes(b"partial")
    calls = []

    class Process:
        returncode = 0

        def __init__(self, command, **kwargs):
            calls.append(command)

        def poll(self):
            return 0

    monkeypatch.setattr(run_queue.subprocess, "Popen", Process)
    monkeypatch.setattr(run_queue, "record_eval", lambda *a: None)
    assert run_queue.run_job(row, "cpu", audited, interval=.01)
    assert len(calls) == 1 and calls[0][2].endswith("exp.evaluate")


def test_queue_stop_prevents_child_start(source, monkeypatch):
    bundle, out, _ = source
    audited = audit.audit_sampling(bundle, "unit")
    row = make_jobs.jobs("unit", audited)[0]
    stopped = Event()
    stopped.set()
    monkeypatch.setattr(run_queue.subprocess, "Popen", lambda *a, **k: pytest.fail("cancelled task started"))
    assert not run_queue.run_job(row, "cpu", audited, stop=stopped)


@pytest.mark.parametrize("key", ["args", "cfg", "step", "selection", "ema", "unloadable"])
def test_internal_checkpoint_identity_and_nonfinite_weights_are_rejected(source, key):
    out, row, audited, cfg, done = completion_record(source)
    ckpt = torch.load(out / "ckpt.pt", map_location="cpu", weights_only=True)
    if key == "unloadable":
        (out / "ckpt.pt").write_bytes(b"not a checkpoint")
    else:
        if key == "args":
            ckpt[key]["formation_ratio"] = .5
        elif key == "cfg":
            ckpt[key]["dim"] = 24
        elif key == "step":
            ckpt[key] = 500
        elif key == "selection":
            ckpt[key] = "val_cascade_acc"
        else:
            next(iter(ckpt["ema"].values())).fill_(float("nan"))
        torch.save(ckpt, out / "ckpt.pt")
    done["ckpt_sha1"] = shared.common.sha1_file(out / "ckpt.pt")
    shared.write_json(out / "done.json", done)
    with pytest.raises(ValueError, match="checkpoint"):
        shared.validate_done(out, row, audited)


def perfect_bench(tmp_path, n=512):
    initial = D.terrain()
    initial[D.FLOOR, 10] = D.STONE
    target = D.terrain()
    background = np.zeros_like(initial)
    pred_fg = np.stack((target, target))[None]
    pred_bg = np.zeros_like(pred_fg)
    mask = np.zeros((1, 2, 28, 56), bool)
    mask[0, :, D.FLOOR, 10] = True
    score = shared.metrics.score_sample((target, background), (target, background), (initial, background), mask[0, 0])
    bench = {"split_ids": np.zeros(1, int), "rank_valid": np.array([False]), "cascade": np.array([[0, 0]]),
             "ctx_fg": initial[None], "ctx_bg": background[None], "fut_fg": pred_fg[:, :, None],
             "fut_bg": pred_bg[:, :, None], "labels": np.array([[0, 0]])}
    raw = {k: np.array([[score[k], score[k]]]) for k in ("cascade_f1", "change_f1", "cell_acc_mask", "any_violation")}
    raw.update(split_ids=bench["split_ids"], rank_valid=bench["rank_valid"], n_cascade_true=bench["cascade"],
        rank_acc=np.array([np.nan]), pred_fg=pred_fg, pred_bg=pred_bg)
    bench = {k: np.repeat(v, n, axis=0) for k, v in bench.items()}
    raw = {k: np.repeat(v, n, axis=0) for k, v in raw.items()}
    shared.write_npz(tmp_path / "bench.npz", bench)
    event = v3.mk("single", (0, [{"type": "remove", "x": 10, "y": D.FLOOR}]))
    shared.write_json(tmp_path / "bench.json", {"scene_kinds": ["tower"] * n, "splits": list(shared.common.SPLITS),
        "interventions": [[event, event] for _ in range(n)]})
    return raw


def test_real_eval_metadata_predictions_and_atomic_commit(source):
    out, row, audited, cfg, done = completion_record(source)
    bench = shared.resolve_file(audited["inputs"]["bench"])
    raw = perfect_bench(bench.parent)
    audited["hashes"]["bench"] = {p.name: shared.common.sha1_file(p) for p in (bench, bench.with_suffix(".json"))}
    metadata = {"bench": shared.common.rel(bench), "n_samples": 512,
        "bench_sha1": audited["hashes"]["bench"][bench.name],
        "bench_json_sha1": audited["hashes"]["bench"][bench.with_suffix(".json").name],
        "checkpoint_sha1": done["ckpt_sha1"]}
    shared.write_json(out / "eval.json", metadata)
    shared.write_npz(out / "eval_samples.npz", raw)
    shared.record_eval(out, audited, done)
    before = (out / "eval_complete.json").read_bytes()
    shared.record_eval(out, audited, done)
    assert before == (out / "eval_complete.json").read_bytes()
    raw["change_f1"][0, 0] = 0
    shared.write_npz(out / "eval_samples.npz", raw)
    with pytest.raises(ValueError, match="identity changed"):
        shared.validate_eval(out, audited, done)


@pytest.mark.parametrize("key", ["cascade_f1", "change_f1", "cell_acc_mask", "any_violation"])
def test_all_reported_metrics_must_match_prediction_states(tmp_path, key):
    raw = perfect_bench(tmp_path, n=2)
    bench = shared.common.load_bench(tmp_path / "bench.npz")
    shared.validate_metrics(raw, bench)
    raw[key][0, 0] = 1 - raw[key][0, 0]
    with pytest.raises(ValueError, match="cached"):
        shared.validate_metrics(raw, bench)


def test_old_validation_best_run_is_rejected_without_any_writes(source, monkeypatch):
    bundle, out, _ = source
    old = deepcopy(bundle["manifest"])
    old["version"] = 2
    old["protocol"].pop("checkpoint_selection")
    old["protocol"].pop("checkpoint_weights")
    shared.write_json(out / "source.json", old)
    before = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
    monkeypatch.setattr(evaluate_final, "execute_queue", lambda *a, **k: pytest.fail("old run must not launch inference or training"))
    with pytest.raises(SystemExit) as exc:
        evaluate_final.cli(["--run", "unit"])
    assert exc.value.code == 2
    assert before == {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}


def test_final_eval_preflights_all_checkpoints_before_any_inference(source, monkeypatch):
    bundle, out, _ = source
    audited = audit.audit_sampling(bundle, "unit")
    monkeypatch.setattr(evaluate_final, "verify_audit", lambda *a: audited)
    checked = []

    def validate(path, row, audited):
        checked.append(row["formation_ratio"])
        if row["formation_ratio"] == .2:
            raise ValueError("final checkpoint missing")
        return {"checkpoint_step": 1000}

    monkeypatch.setattr(evaluate_final, "validate_done", validate)
    monkeypatch.setattr(evaluate_final, "execute_queue", lambda *a, **k: pytest.fail("must not partially evaluate a run missing final weights"))
    with pytest.raises(ValueError, match="final checkpoint missing"):
        evaluate_final.main(["--run", "unit"])
    assert checked == [0, .1, .2]
    assert not (out / "pipeline.lock").exists()


def test_final_eval_dry_run_writes_nothing(source, monkeypatch):
    bundle, out, _ = source
    audited = audit.audit_sampling(bundle, "unit")
    monkeypatch.setattr(evaluate_final, "verify_audit", lambda *a: audited)
    monkeypatch.setattr(evaluate_final, "validate_done", lambda *a: {"checkpoint_step": 1000})
    monkeypatch.setattr(evaluate_final, "execute_queue", lambda *a, **k: pytest.fail("dry run must not infer or train"))
    before = sorted(out.rglob("*"))
    evaluate_final.main(["--run", "unit", "--dry-run"])
    assert before == sorted(out.rglob("*"))


def test_final_eval_calls_evaluation_only_queue_then_analysis(source, monkeypatch):
    bundle, out, _ = source
    audited = audit.audit_sampling(bundle, "unit")
    monkeypatch.setattr(evaluate_final, "verify_audit", lambda *a: audited)
    monkeypatch.setattr(evaluate_final, "validate_done", lambda *a: {"checkpoint_step": 1000})
    calls = []

    def queue(name, cards, interval, evaluate_only=False):
        assert evaluate_only is True
        calls.append((name, cards, "eval_only"))

    monkeypatch.setattr(evaluate_final, "execute_queue", queue)
    monkeypatch.setattr(evaluate_final, "analyze", lambda *a: calls.append("analysis"))
    evaluate_final.main(["--run", "unit", "--gpus", "2,3"])
    assert calls == [("unit", "2,3", "eval_only"), "analysis"]


def test_evaluation_only_task_with_missing_training_never_starts_training(source, monkeypatch):
    bundle, out, _ = source
    audited = audit.audit_sampling(bundle, "unit")
    row = make_jobs.jobs("unit", audited)[0]
    monkeypatch.setattr(run_queue.subprocess, "Popen", lambda *a, **k: pytest.fail("eval-only must not start any training"))
    with pytest.raises(FileNotFoundError):
        run_queue.run_job(row, "cpu", audited, evaluate_only=True)
    assert not shared.resolve_file(row["out"]).exists()


@pytest.mark.parametrize("ratio", shared.RATIOS)
def test_decile_cumulative_quota_and_full_budget_are_exact(ratio):
    from fractions import Fraction
    fraction = Fraction(str(ratio))
    counts = [shared.formation_count(ratio, step=s) for s in range(1, shared.STEPS + 1)]
    budget = shared.budget(ratio)
    assert sum(counts) == shared.STEPS * shared.BATCH * fraction
    assert sum(counts) == budget["formation_samples"]
    assert budget["formation_samples"] + budget["operation_samples"] == budget["samples"]
    assert all(0 <= n <= shared.BATCH for n in counts)
    assert max(counts) - min(counts) <= 1
    assert shared.cumulative_formation(ratio, 256, 0) == 0


@pytest.mark.parametrize("ratio", shared.RATIOS)
def test_decile_sampler_preserves_exact_cumulative_counts(ratio):
    pools = (torch.arange(31), torch.arange(31, 59))
    g1, g2 = (torch.Generator().manual_seed(7) for _ in range(2))
    total = 0
    for step in range(1, 26):
        a = trainer.sample_ratio_batch(pools, 256, ratio, g1, step=step)
        b = trainer.sample_ratio_batch(pools, 256, ratio, g2, step=step)
        assert torch.equal(a, b)
        total += int((a < 31).sum())
        assert int((a < 31).sum()) == shared.formation_count(ratio, 256, step)
        assert total == shared.cumulative_formation(ratio, 256, step)


def test_hundred_percent_does_not_require_operation_samples():
    pools = (torch.arange(3), torch.empty(0, dtype=torch.long))
    ids = trainer.sample_ratio_batch(pools, 256, 1.0, torch.Generator())
    assert len(ids) == 256 and ids.max() < 3
    with pytest.raises(ValueError, match="empty"):
        trainer.sample_ratio_batch(pools, 256, .9, torch.Generator())


@pytest.mark.parametrize("ratio", shared.RATIOS)
def test_actual_decile_cpu_training_with_final_checkpoint(source, monkeypatch, ratio):
    bundle, out, _ = source
    data = trainer.pilot.Data(bundle["inputs"]["data"], "process", "video", 2, torch.device("cpu"))
    val = trainer.pilot.Data(bundle["inputs"]["data"], "state", "video", 2, torch.device("cpu"))
    population = audit.population(bundle["inputs"]["data"], 2)
    args = SimpleNamespace(**{**shared.TRAIN_DEFAULTS, "dim": 12, "depth": 1, "heads": 3,
        "formation_ratio": ratio, "batch": 8, "steps": 5, "seed": 0, "log_every": 1, "val_every": 1})
    original = trainer.models.build_model
    monkeypatch.setattr(trainer.models, "build_model", lambda cfg: original({**cfg, "emb": 4}))
    config = {"version": shared.VERSION, "pools": population["pools"], "args": vars(args)}
    target = out / "decile_cpu"
    target.mkdir()
    done = trainer.fit(data, val, args, target, config)
    assert done["samples_seen"] == 40
    assert done["formation_samples_seen"] == round(40 * ratio)
    assert done["operation_samples_seen"] == 40 - round(40 * ratio)
    assert done["form_frac_mean"] == ratio
    checkpoint = torch.load(target / "ckpt.pt", map_location="cpu", weights_only=True)
    assert checkpoint["step"] == 5 and checkpoint["selection"] == "last"
    rows = [json.loads(line) for line in (target / "train_log.jsonl").read_text().splitlines() if "samples_seen" in line]
    for record in rows:
        step = record["step"]
        assert record["formation_samples_seen"] == shared.cumulative_formation(ratio, 8, step)


def test_short_custom_budget_reports_actual_quota_not_target_rounding():
    result = shared.budget(.1, steps=1)
    assert result["formation_samples"] == 25
    assert result["actual_formation_fraction"] == 25 / 256
    assert shared.formation_count(.1, step=5) == 26
    for step in (0, -1, 1.5):
        with pytest.raises(ValueError):
            shared.formation_count(.1, step=step)


def test_eleven_dose_keys_do_not_collide_and_legacy_keys_stay_stable():
    assert len({shared.dose_key(r) for r in shared.RATIOS}) == 11
    assert shared.dose_key(.1) == "p010" and shared.dose_key(1) == "r256"
    assert [shared.dose_key(r) for r in (0, .125, .25, .5, .75)] == ["r000", "r032", "r064", "r128", "r192"]


def test_legacy_five_dose_protocol_remains_unchanged_and_cannot_be_overwritten(source, monkeypatch):
    bundle, out, _ = source
    old_ratios = (0, .125, .25, .5, .75)
    legacy = deepcopy(bundle)
    legacy["manifest"]["protocol"] = shared.protocol(old_ratios, steps=1000)
    legacy["manifest"]["protocol"]["episodes"] = 2
    assert "batch_quota" not in legacy["manifest"]["protocol"]
    shared.write_json(out / "source.json", legacy["manifest"])
    before = (out / "source.json").read_bytes()
    old_audit = audit.audit_sampling(legacy, "unit")
    old_rows = make_jobs.save_jobs("unit", old_audit)
    assert len(old_rows) == 5
    monkeypatch.setattr(shared, "validate_source", lambda *a: legacy)
    assert shared.verify_manifest("unit") == legacy
    assert audit.verify_audit("unit") == old_audit
    with pytest.raises(ValueError, match="budget changed"):
        shared.bind_run("unit", bundle)
    assert before == (out / "source.json").read_bytes()


def test_eleven_dose_report_covers_endpoints_and_integer_ranges():
    scores, ids = synthetic_scores()
    report = analysis.summarize(scores, ids, shared.common.SPLITS, boot=20)
    assert report["settings"]["dose_count"] == 11
    assert report["settings"]["full_eleven_dose_design"] is True
    text = analysis.markdown(report)
    assert "完整11档" in text and "100.0%" in text and "25–26/230–231" in text
    legacy = (0, .125, .25, .5, .75)
    scores, ids = synthetic_scores(legacy)
    report = analysis.summarize(scores, ids, shared.common.SPLITS, boot=20, ratios=legacy)
    assert report["settings"]["dose_count"] == 5
    assert "本次为5档" in analysis.markdown(report)
