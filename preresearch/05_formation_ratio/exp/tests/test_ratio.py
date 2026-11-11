from contextlib import ExitStack
from copy import deepcopy
from importlib import import_module
import io
from pathlib import Path
import zipfile

import numpy as np
import pytest

from exp import analysis, audit, generate_data as gen, main, make_jobs, run_queue, shared
import datasets as D
import datasets_v3 as v3


def tower_episode(index=0):
    actions = [D.act_dict(D.place(10 + index, D.FLOOR - j, D.STONE)) for j in range(5)]
    script = [[(v3.to_action(a), "build")] for a in actions] + [[] for _ in range(22)]
    snapshots, world, ok = v3.run_formation(script, D.int_seed(0, "world", index), 32)
    assert ok
    formed = D.pack(snapshots, [0] * 32)
    operation = D.op_clip(world.grid, world.background, D.Action("remove", 10 + index, D.FLOOR), 0)
    state = D.pack(operation * 8, [1] * 64)
    return formed, state


@pytest.fixture
def small_source(tmp_path, monkeypatch):
    n = 4
    original_budget = shared.budget
    for module in (shared, gen, audit):
        monkeypatch.setattr(module, "EPISODES", n)
        monkeypatch.setattr(module, "budget", lambda F: original_budget(F, n))
    monkeypatch.setattr(shared.common, "RESULTS_ROOT", tmp_path / "results")
    data = tmp_path / "frozen"
    data.mkdir()
    episodes = [tower_episode(i) for i in range(n)]
    arrays = {k: np.stack([state[k] for _, state in episodes]) for k in ("fg", "bg", "cont", "phase")}
    state_actions = [state["actions"] for _, state in episodes]
    formation = [formed["actions"] + state["actions"][:32] for formed, state in episodes]
    shared.write_npz(data / "train_state.npz", arrays)
    shared.write_json(data / "train_state_actions.json", {"actions": state_actions})
    shared.write_json(data / "train_process_actions.json", {"actions": formation})
    shared.write_json(data / "train_scenes.json", [{"kind": "tower", "n_struct": 1,
        "final_hash": D.state_hash(state["fg"][0], state["bg"][0])} for _, state in episodes])
    hashes = {p.name: shared.common.sha1_file(p) for p in data.iterdir()}
    inputs = {"data": str(data), "val": str(tmp_path / "validation"), "bench": str(tmp_path / "bench.npz")}
    manifest = {"version": shared.VERSION, "hashes": {"data": hashes}, "endpoints": {"0": "source0", "32": "source32"},
                "protocol": {"levels": list(shared.LEVELS), "training_seeds": [0]}}
    bundle = {"source": tmp_path / "source03", "inputs": {k: Path(v) for k, v in inputs.items()},
              "manifest": manifest, "audited": {"inputs": inputs, "hashes": {"data": hashes}}}
    out = shared.bind_run("unit", bundle)
    return bundle, out, arrays, state_actions, formation


@pytest.mark.parametrize("F,pairs,steps", [(0, 56, 89600), (8, 56, 89600), (16, 57, 91200), (32, 59, 94400), (48, 61, 97600)])
def test_exact_budget_and_real_video_boundaries(F, pairs, steps):
    b = shared.budget(F)
    assert b["pairs_per_episode"] == pairs
    assert b["steps"] == steps
    assert b["samples"] == 81920 * pairs * 5
    assert sum(end - start - 1 for start, end in shared.common.segments("process", {"formation": F, "clip": 8}, 64)) == pairs
    assert b["formation_fraction"] == pytest.approx(max(0, F - 1) / pairs)


@pytest.mark.parametrize("levels", [(), (0, 32), (0, 8, 16), (0, 32, 64), (0, 32, 16), (0, 16, 16, 32)])
def test_invalid_levels_are_not_silently_changed(levels):
    with pytest.raises(ValueError):
        shared.validate_levels(levels)


def test_explicit_fallback_preserves_both_endpoints():
    assert shared.validate_levels((0, 16, 32, 48)) == (0, 16, 32, 48)
    assert shared.new_levels({"protocol": {"levels": [0, 16, 32, 48]}}) == (16, 48)


@pytest.mark.parametrize("name", ["", ".", "..", "../bad", "a/b"])
def test_run_names_cannot_escape_result_folder(name):
    with pytest.raises(ValueError):
        shared.run_path(name)


def test_balanced_partition_preserves_all_actions():
    actions = [{"type": "place", "x": i, "y": 0} for i in range(23)]
    frames = gen.partition_actions(actions, 3)
    assert sorted(map(len, frames)) == [7, 8, 8]
    assert sum(frames, []) == actions
    assert gen.partition_actions([], 3) == [[], [], []]
    assert len(gen.partition_actions(actions[:2], 43)) == 43


def test_front_back_same_cell_must_use_different_tokens():
    actions = [{"type": "place", "x": 3, "y": 4, "layer": layer} for layer in ("front", "back")]
    assert gen.partition_actions(actions, 2) == [[actions[0]], [actions[1]]]
    with pytest.raises(ValueError, match="repeated-cell"):
        gen.partition_actions(actions * 2, 3)


@pytest.mark.parametrize("F", [8, 16, 48])
def test_replay_complete_formation_and_nested_operations(F):
    formed, state = tower_episode()
    episode, stats = gen.build_episode(F, 0, formed["actions"], state)
    assert stats["n_actions"] == 5
    audit.validate_episode(episode, episode["actions"], state, state["actions"], formed["actions"], F)
    assert episode["fg"].shape == (64, 28, 56)
    assert sum(episode["cont"]) == shared.budget(F)["pairs_per_episode"]
    assert not any(episode["actions"][F - 4:F])


@pytest.mark.parametrize("mutation", ["operation", "action_order", "final", "phase", "reset"])
def test_episode_audit_rejects_protocol_changes(mutation):
    formed, state = tower_episode()
    episode, _ = gen.build_episode(16, 0, formed["actions"], state)
    if mutation == "operation":
        episode["fg"][16, 0, 0] = 9
    elif mutation == "action_order":
        frame = next(frame for frame in episode["actions"][:16] if frame)
        frame.append(frame[0])
    elif mutation == "final":
        episode["fg"][15, 0, 0] = 9
    elif mutation == "phase":
        episode["phase"][1] = 1
    else:
        episode["cont"][16] = True
    with pytest.raises(ValueError):
        audit.validate_episode(episode, episode["actions"], state, state["actions"], formed["actions"], 16)


def test_replay_rejects_final_changes_and_source_layout():
    formed, state = tower_episode()
    wrong = state["fg"][0].copy()
    wrong[0, 0] = 1
    with pytest.raises(ValueError, match="final structure"):
        gen.replay_formation(formed["actions"], 8, 0, wrong, state["bg"][0])
    with pytest.raises(ValueError, match="exactly 32"):
        gen.replay_formation(formed["actions"][:8], 8, 0, wrong, state["bg"][0])


def test_source_cache_integrity_and_manifest_reuse(small_source):
    bundle, out, arrays, _, _ = small_source
    cached, _, _ = gen.cache_arrays(bundle, out)
    assert all(isinstance(v, np.memmap) for v in cached.values())
    for k in arrays:
        np.testing.assert_array_equal(cached[k], arrays[k])
    before = (out / "source.json").read_bytes()
    shared.bind_run("unit", bundle)
    assert (out / "source.json").read_bytes() == before
    changed = deepcopy(bundle)
    changed["manifest"]["protocol"]["levels"] = [0, 16, 32, 48]
    with pytest.raises(ValueError, match="changed"):
        shared.bind_run("unit", changed)
    with (out / "cache/fg.npy").open("ab") as file:
        file.write(b"changed")
    with pytest.raises(ValueError, match="cache changed"):
        gen.cache_arrays(bundle, out)


def test_preflight_failure_blocks_all_generation_and_training(small_source, monkeypatch):
    bundle, out, arrays, actions, formation = small_source
    original = gen.replay_formation

    def fail_on_one_scene(frames, F, index, fg, bg):
        if F == 8 and index == 2:
            raise ValueError("repeated-cell construction slots")
        return original(frames, F, index, fg, bg)

    monkeypatch.setattr(gen, "replay_formation", fail_on_one_scene)
    monkeypatch.setattr(gen, "create_variant", lambda *a, **k: pytest.fail("generation must not start"))
    with pytest.raises(ValueError, match="No training started"):
        gen.prepare_all(bundle, "unit", workers=1, chunk=2, preflight_count=4)
    report = shared.common.load_json(out / "formation_preflight.json")
    assert report["status"] == "blocked"
    assert report["failures"][0]["episode"] == 2
    assert not (out / "data").exists()


def test_variants_full_assembly_audit_and_no_replay_on_resume(small_source, monkeypatch):
    bundle, out, arrays, actions, formation = small_source
    gen.prepare_all(bundle, "unit", workers=1, chunk=2, preflight_count=4)
    report = audit.audit_variants(bundle, "unit")
    assert set(report["variants"]) == {"8", "16", "48"}
    assert report["scene_distribution"]["scene_types"] == {"tower": 4}
    monkeypatch.setattr(gen, "replay_formation", lambda *a, **k: pytest.fail("completed data must not be replayed"))
    gen.prepare_all(bundle, "unit", workers=1, chunk=2, preflight_count=4)
    assert audit.audit_variants(bundle, "unit") == report
    target = out / "data/f16/train_process.npz"
    with np.load(target) as z:
        broken = {k: z[k] for k in z.files}
    broken["fg"][0, 0, 0, 0] = 5
    shared.write_npz(target, broken)
    with pytest.raises(ValueError, match="identity changed"):
        shared.verify_variant(target.parent, bundle["manifest"], 16)
    marker = shared.common.load_json(target.parent / "complete.json")
    marker["hashes"] = shared.variant_hashes(target.parent)
    shared.write_json(target.parent / "complete.json", marker)
    with pytest.raises(ValueError, match="arrays differ"):
        audit.audit_variants(bundle, "unit")


def test_explicit_reduced_generation_does_not_create_f8(small_source):
    bundle, out, arrays, actions, formation = small_source
    bundle["manifest"]["protocol"]["levels"] = [0, 16, 32, 48]
    gen.prepare_all(bundle, "unit", workers=1, chunk=2, preflight_count=4)
    assert not (out / "data/f8").exists()
    assert set(audit.audit_variants(bundle, "unit")["variants"]) == {"16", "48"}


@pytest.mark.parametrize("error", ["truncated", "different"])
def test_stream_comparison_catches_incomplete_and_mismatched_arrays(error):
    saved = np.arange(12, dtype=np.uint8).reshape(3, 4)
    payload = saved.tobytes()[:-1] if error == "truncated" else bytes(12)
    with pytest.raises(ValueError):
        audit.compare_array_batch(io.BytesIO(payload), saved)


def test_stream_reader_rejects_fortran_layout(tmp_path):
    path = tmp_path / "arrays.npz"
    values = np.asfortranarray(np.ones((4, 3), dtype=np.uint8))
    shared.write_npz(path, {"fg": values})
    with ExitStack() as stack:
        z = stack.enter_context(zipfile.ZipFile(path))
        with pytest.raises(ValueError, match="layout"):
            audit.open_array_stream(z, stack, "fg", values)


def test_jobs_use_absolute_inputs_and_only_selected_new_doses(tmp_path, monkeypatch):
    monkeypatch.setattr(shared.common, "RESULTS_ROOT", tmp_path)
    audited = {"protocol": {"levels": [0, 16, 32, 48]}, "inputs": {"val": str(tmp_path / "val")},
               "variants": {str(F): {"data": shared.common.rel(tmp_path / f"f{F}")} for F in (16, 48)}}
    rows = make_jobs.jobs("unit", audited)
    assert [r["F"] for r in rows] == [16, 48]
    trainer = import_module(f"{shared.PACKAGE}.train")
    for row in rows:
        args = trainer.get_args(row["cmd"][3:])
        assert args.data.is_absolute()
        assert args.epochs == 5 and args.seed == 0
        assert args.select_best_val and not args.balanced_phases
        assert args.steps == shared.budget(row["F"])["steps"]
        assert args.dim == 384 and args.heads == 6
        assert "05_formation_ratio/unit/" in row["out"]
    make_jobs.save_jobs("unit", audited)
    original = shared.run_path("unit") / "jobs/formation_ratio.json"
    rows = shared.common.load_json(original)
    rows[0]["F"] = 8
    shared.write_json(original, rows)
    with pytest.raises(ValueError, match="changed"):
        make_jobs.save_jobs("unit", audited)


def synthetic_scores(levels=shared.LEVELS):
    ids = np.repeat(np.arange(4), 8)
    scores = {F: {k: np.full(32, .2 + F / 320) for k in analysis.METRICS} for F in levels}
    return scores, ids


def test_paired_statistics_slope_ties_and_seed_boundary():
    scores, ids = synthetic_scores()
    report = analysis.summarize(scores, ids, shared.common.SPLITS, boot=40, seed=9)
    metric = report["by_split"]["id"]["metrics"]["cascade_f1"]
    assert metric["trend"]["spearman"] == pytest.approx(1)
    assert metric["trend"]["slope_per_unit_fraction"] == pytest.approx(.2)
    assert metric["contrasts"]["vs_f0"]["16"]["mean"] == pytest.approx(.05)
    np.testing.assert_allclose(metric["contrasts"]["vs_f0"]["16"]["ci"], [.05, .05])
    assert metric["monotonic_increasing_means"] is True
    assert report["settings"]["seed_std"] is None
    assert report == analysis.summarize(scores, ids, shared.common.SPLITS, boot=40, seed=9)
    assert analysis.spearman([0, 1, 2], [2, 2, 2]) is None
    np.testing.assert_allclose(analysis.average_ranks([3, 1, 1]), [2, .5, .5])


def test_missing_dose_empty_groups_and_nonmonotonic_outcome():
    scores, ids = synthetic_scores()
    scores[16]["cascade_f1"][:] = .9
    report = analysis.summarize(scores, ids, shared.common.SPLITS, boot=20)
    row = report["by_split"]["id"]["metrics"]["cascade_f1"]
    assert row["descriptive_best_F"] == 16
    assert row["monotonic_increasing_means"] is False
    scores[48]["rank_acc"][:] = np.nan
    report = analysis.summarize(scores, ids, shared.common.SPLITS, boot=20)
    assert report["by_split"]["id"]["metrics"]["rank_acc"]["trend"]["n_cases"] == 0
    scores.pop(8)
    with pytest.raises(ValueError, match="all selected doses"):
        analysis.summarize(scores, ids, shared.common.SPLITS, boot=20)


@pytest.mark.parametrize("levels", [shared.LEVELS, (0, 16, 32, 48)])
def test_report_and_plots_match_actual_levels(tmp_path, levels):
    scores, ids = synthetic_scores(levels)
    report = analysis.summarize(scores, ids, shared.common.SPLITS, boot=20, levels=levels)
    text = analysis.markdown(report)
    assert ("显式缩减" in text) == (levels != shared.LEVELS)
    assert ("| F=8 |" in text) == (8 in levels)
    analysis.plot(report, tmp_path)
    assert all((tmp_path / f"dose_response.{suffix}").stat().st_size > 100 for suffix in ("png", "pdf"))


def test_analysis_saves_new_reports_without_overwriting(tmp_path, monkeypatch):
    monkeypatch.setattr(shared.common, "RESULTS_ROOT", tmp_path)
    scores, ids = synthetic_scores((0, 16, 32, 48))
    shared.write_json(tmp_path / "bench.json", {"splits": list(shared.common.SPLITS)})
    audited = {"protocol": {"levels": [0, 16, 32, 48]}, "inputs": {"bench": str(tmp_path / "bench.npz")}, "source": {"frozen": True}}
    monkeypatch.setattr(analysis, "verify_audit", lambda name: audited)
    monkeypatch.setattr(analysis, "load_results", lambda *a: (scores, ids, {"0": "source"}, {}))
    first = analysis.analyze("unit", boot=20)
    before = (first / "formation_ratio.json").read_bytes()
    second = analysis.analyze("unit", boot=20)
    assert first != second
    assert (first / "formation_ratio.json").read_bytes() == before
    assert second.is_relative_to(shared.run_path("unit"))


def test_dry_run_never_creates_or_trains(tmp_path, monkeypatch):
    monkeypatch.setattr(shared.common, "RESULTS_ROOT", tmp_path)
    monkeypatch.delenv("FW_RUN", raising=False)
    monkeypatch.setattr(main, "validate_source", lambda source, levels: {"manifest": {"protocol": {"levels": list(levels)}}})
    monkeypatch.setattr(main, "bind_run", lambda *a: pytest.fail("dry-run must not bind or write"))
    monkeypatch.setattr(main, "execute_queue", lambda *a: pytest.fail("dry-run must not train"))
    main.main(["--dry-run", "--levels", "0,16,32,48"])
    assert not list(tmp_path.iterdir())


def test_run_lock_rejects_concurrent_pipelines(tmp_path):
    with main.run_lock(tmp_path):
        with pytest.raises(ValueError, match="another"):
            with main.run_lock(tmp_path):
                pytest.fail("concurrent pipeline acquired same run lock")


@pytest.mark.parametrize("cards", ["", "0,0", "0,", "-1", "foo"])
def test_invalid_devices_are_rejected_before_source_check(cards, monkeypatch):
    monkeypatch.setattr(run_queue, "verify_audit", lambda *a: pytest.fail("invalid devices should fail first"))
    with pytest.raises(ValueError):
        run_queue.execute_queue("unit", cards)


def test_completed_task_is_reused_without_subprocess(tmp_path, monkeypatch):
    for name in ("done.json", "eval.json", "eval_samples.npz", "eval_complete.json"):
        (tmp_path / name).write_text("{}")
    done = {"ckpt_sha1": "checked"}
    monkeypatch.setattr(run_queue, "validate_done", lambda *a: done)
    monkeypatch.setattr(run_queue, "validate_eval", lambda out, audited, value: None if value == done else pytest.fail("wrong checkpoint"))
    monkeypatch.setattr(run_queue.subprocess, "Popen", lambda *a, **k: pytest.fail("completed task must not launch"))
    assert run_queue.run_job({"F": 16, "out": str(tmp_path)}, "cpu", {})


def test_invalid_completed_task_is_not_overwritten(tmp_path, monkeypatch):
    (tmp_path / "done.json").write_text("{}")
    monkeypatch.setattr(run_queue, "validate_done", lambda *a: (_ for _ in ()).throw(ValueError("changed checkpoint")))
    monkeypatch.setattr(run_queue.subprocess, "Popen", lambda *a, **k: pytest.fail("invalid completion must not rerun"))
    assert not run_queue.run_job({"F": 16, "out": str(tmp_path)}, "cpu", {})
    assert shared.common.load_json(tmp_path / "failed.json")["error"] == "changed checkpoint"


def test_queue_launches_training_then_evaluates_selected_checkpoint(tmp_path, monkeypatch):
    calls, stages = [], []
    done = {"ckpt_sha1": "selected"}

    class Process:
        returncode = 0

        def __init__(self, command, **kwargs):
            calls.append(command)
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "2"

        def poll(self):
            return 0

    def validate_done(*args):
        stages.append("train")
        return done

    def validate_eval(out, audited, value):
        assert value == done
        stages.append("eval")

    monkeypatch.setattr(run_queue.subprocess, "Popen", Process)
    monkeypatch.setattr(run_queue, "validate_done", validate_done)
    monkeypatch.setattr(run_queue, "record_eval", validate_eval)
    row = {"F": 16, "out": str(tmp_path), "cmd": ["python", "-m", "exp.train"]}
    assert run_queue.run_job(row, "2", {"inputs": {"bench": str(tmp_path / "bench.npz")}}, interval=.01)
    assert [command[2] for command in calls] == ["exp.train", "exp.evaluate"]
    assert stages == ["train", "eval"]


def test_source_and_child_symlinks_are_rejected_without_writes(small_source):
    bundle, out, arrays, actions, formation = small_source
    data = out / "data"
    data.mkdir()
    (data / "f16").symlink_to(bundle["inputs"]["data"], target_is_directory=True)
    before = {p.name: p.read_bytes() for p in bundle["inputs"]["data"].iterdir()}
    with pytest.raises(ValueError, match="symbolic link"):
        gen.create_variant(bundle, out, 16, arrays, actions, formation, workers=1, chunk=2)
    with pytest.raises(ValueError, match="symbolic link"):
        shared.bind_run("unit", bundle)
    assert before == {p.name: p.read_bytes() for p in bundle["inputs"]["data"].iterdir()}


def test_interrupted_generation_preserves_shards_and_rejects_changed_shard_size(small_source, monkeypatch):
    bundle, out, arrays, actions, formation = small_source
    original = gen._worker

    def fail_later(args):
        if args[1] == 2:
            return 2, None, None, "construction failed"
        return original(args)

    monkeypatch.setattr(gen, "_worker", fail_later)
    with pytest.raises(ValueError, match="completed shards retained"):
        gen.create_variant(bundle, out, 16, arrays, actions, formation, workers=1, chunk=2)
    shard = out / "data/f16/shards/000000_000002/complete.json"
    before = shard.read_bytes()
    with pytest.raises(ValueError, match="shard size changed"):
        gen.create_variant(bundle, out, 16, arrays, actions, formation, workers=1, chunk=1)
    assert shard.read_bytes() == before
    monkeypatch.setattr(gen, "_worker", original)
    gen.create_variant(bundle, out, 16, arrays, actions, formation, workers=1, chunk=2)
    assert shard.read_bytes() == before
    assert not (out / "data/f16/blocked.json").exists()


def test_resume_inherits_source_and_doses_before_read_only_check(tmp_path, monkeypatch):
    monkeypatch.setattr(shared.common, "RESULTS_ROOT", tmp_path)
    manifest = {"source_run": "alternative_source", "protocol": {"levels": [0, 16, 32, 48]}}
    shared.write_json(shared.run_path("unit") / "source.json", manifest)
    calls = []

    def source(value, levels):
        calls.append((value, levels))
        return {"manifest": manifest}

    monkeypatch.setattr(main, "validate_source", source)
    main.main(["--run", "unit", "--dry-run"])
    assert calls == [("alternative_source", (0, 16, 32, 48))]


@pytest.mark.parametrize("module", [gen, audit, make_jobs, run_queue, analysis])
def test_all_independent_entrypoints_share_pipeline_lock(tmp_path, monkeypatch, module):
    monkeypatch.setattr(shared.common, "RESULTS_ROOT", tmp_path)
    with shared.run_lock(shared.run_path("unit")):
        with pytest.raises(ValueError, match="another"):
            module.main(["--run", "unit"])


def prediction_fixture():
    initial = D.terrain()
    initial[D.FLOOR, 10] = D.STONE
    background = np.zeros_like(initial)
    truth = D.terrain()
    pred_fg = np.stack((truth, truth))[None]
    pred_bg = np.zeros_like(pred_fg)
    mask = np.zeros((1, 2, 28, 56), bool)
    mask[0, :, D.FLOOR, 10] = True
    scores = [shared.metrics.score_sample((truth, background), (truth, background), (initial, background), mask[0, k]) for k in range(2)]
    bench = {"split_ids": np.zeros(1, int), "rank_valid": np.array([False]), "cascade": np.array([[0, 0]]),
             "ctx_fg": initial[None], "ctx_bg": background[None], "fut_fg": pred_fg[:, :, None],
             "fut_bg": pred_bg[:, :, None], "A": mask, "labels": np.array([[0, 0]])}
    samples = {key: np.array([[score[key] for score in scores]]) for key in ("cascade_f1", "change_f1", "cell_acc_mask", "any_violation")}
    samples.update(split_ids=bench["split_ids"], rank_valid=bench["rank_valid"], n_cascade_true=bench["cascade"],
                   rank_acc=np.array([np.nan]), pred_fg=pred_fg, pred_bg=pred_bg)
    return samples, bench


@pytest.mark.parametrize("key", ["cascade_f1", "change_f1", "cell_acc_mask", "any_violation"])
def test_all_reported_prediction_metrics_are_recomputed(key):
    samples, bench = prediction_fixture()
    shared.validate_metrics(samples, bench)
    samples[key][0, 0] = 1 - samples[key][0, 0]
    with pytest.raises(ValueError, match="cached"):
        shared.validate_metrics(samples, bench)


def test_eval_commit_rejects_changed_prediction_bytes(tmp_path, monkeypatch):
    samples, bench = prediction_fixture()
    n = 512
    samples = {k: np.repeat(v, n, axis=0) for k, v in samples.items()}
    bench = {k: np.repeat(v, n, axis=0) for k, v in bench.items()}
    shared.write_npz(tmp_path / "bench.npz", {k: v for k, v in bench.items() if k != "A"})
    event = v3.mk("single", (0, [{"type": "remove", "x": 10, "y": D.FLOOR}]))
    shared.write_json(tmp_path / "bench.json", {"scene_kinds": ["tower"] * n,
        "splits": list(shared.common.SPLITS), "interventions": [[event, event] for _ in range(n)]})
    shared.write_npz(tmp_path / "eval_samples.npz", samples)
    shared.write_json(tmp_path / "eval.json", {"checked": True})
    monkeypatch.setattr(shared.main_shared, "validate_evaluation", lambda *a: {"checked": True})
    done = {"ckpt_sha1": "selected"}
    audited = {"inputs": {"bench": str(tmp_path / "bench.npz")}}
    shared.record_eval(tmp_path, audited, done)
    before = (tmp_path / "eval_complete.json").read_bytes()
    shared.record_eval(tmp_path, audited, done)
    assert (tmp_path / "eval_complete.json").read_bytes() == before
    samples["change_f1"][0, 0] = 0
    shared.write_npz(tmp_path / "eval_samples.npz", samples)
    with pytest.raises(ValueError, match="identity changed"):
        shared.validate_eval(tmp_path, audited, done)


def test_uncommitted_eval_restarts_without_retraining(tmp_path, monkeypatch):
    (tmp_path / "done.json").write_text("{}")
    (tmp_path / "eval.json").write_text("{}")
    (tmp_path / "eval_samples.npz").write_bytes(b"truncated")
    calls = []
    done = {"ckpt_sha1": "selected"}

    class Process:
        returncode = 0

        def __init__(self, command, **kwargs):
            calls.append(command)

        def poll(self):
            return 0

    monkeypatch.setattr(run_queue, "validate_done", lambda *a: done)
    monkeypatch.setattr(run_queue, "record_eval", lambda *a: None)
    monkeypatch.setattr(run_queue.subprocess, "Popen", Process)
    row = {"F": 16, "out": str(tmp_path), "cmd": ["python", "-m", "exp.train"]}
    assert run_queue.run_job(row, "cpu", {"inputs": {"bench": str(tmp_path / "bench.npz")}}, interval=.01)
    assert len(calls) == 1 and calls[0][2] == "exp.evaluate"


@pytest.fixture
def complete_training(tmp_path):
    hashes = {name: "frozen" for name in ("train_process.npz", "train_process_actions.json", "train_summary.json")}
    val_hashes = {name: "validation" for name in ("train_state.npz", "train_state_actions.json", "train_summary.json")}
    audited = {"protocol": {"levels": [0, 16, 32]}, "inputs": {"val": str(tmp_path / "val")},
        "hashes": {"val": val_hashes}, "variants": {"16": {"data": str(tmp_path / "data"), "hashes": hashes}}}
    row = make_jobs.jobs("unit", audited)[0]
    args = import_module(f"{shared.PACKAGE}.train").get_args(row["cmd"][3:])
    config = {"args": {k: shared.common.rel(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "data_sha1": hashes, "val_sha1": val_hashes}
    shared.write_json(tmp_path / "config.json", config)
    (tmp_path / "ckpt.pt").write_bytes(b"selected EMA")
    b = shared.budget(16)
    done = {"steps": b["steps"], "epochs": 5, "steps_per_epoch": b["steps_per_epoch"], "n_pairs": b["pairs"],
            "samples_seen": b["samples"], "selection": "val_cascade_acc", "best_step": b["steps_per_epoch"],
            "form_frac_mean": b["formation_fraction"], "ckpt_sha1": shared.common.sha1_file(tmp_path / "ckpt.pt")}
    shared.write_json(tmp_path / "done.json", done)
    return tmp_path, row, audited, config, done


def test_real_training_completion_contract(complete_training):
    path, row, audited, _, done = complete_training
    assert shared.validate_done(path, row, audited) == done


@pytest.mark.parametrize("field", ["seed", "dim", "samples_seen", "best_step", "form_frac_mean", "data_sha1", "ckpt_sha1"])
def test_completion_rejects_budget_settings_or_identity_changes(complete_training, field):
    path, row, audited, config, done = complete_training
    if field in ("seed", "dim"):
        config["args"][field] += 1
    elif field == "data_sha1":
        config[field] = {}
    else:
        done[field] = "changed" if field == "ckpt_sha1" else 1
    shared.write_json(path / "config.json", config)
    shared.write_json(path / "done.json", done)
    with pytest.raises(ValueError):
        shared.validate_done(path, row, audited)


def test_sparse_formation_uses_full_construction_time_span():
    actions = [{"type": "place", "x": i, "y": 0} for i in range(5)]
    frames = gen.partition_actions(actions, 43)
    occupied = [i for i, frame in enumerate(frames) if frame]
    assert occupied[-1] == 42 and occupied[0] > 0
    assert sum(frames, []) == actions
