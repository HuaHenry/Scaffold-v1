from copy import deepcopy
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from exp import analysis, evaluate_final, main, make_jobs, run_queue, shared, train_scale

main_shared = import_module("preresearch.03_main.exp.shared")


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small_data(path, episodes=3, offset=0):
    path.mkdir()
    fg = np.zeros((episodes, 64, shared.common.H, shared.common.W), dtype=np.uint8)
    bg = np.zeros_like(fg)
    fg[:, :, -1, :] = 1
    for e in range(episodes):
        fg[e, :32, -2, 10 + offset + e] = 2
        fg[e, 32:, -2, 10 + offset + e] = 3
    for c in shared.CONDITIONS:
        phase = np.ones((episodes, 64), dtype=np.int8)
        if c in ("process", "shuffle"):
            phase[:, :32] = 0
        shared.helpers.write_npz(path / f"train_{c}.npz", {
            "fg": fg, "bg": bg, "phase": phase, "cont": np.ones((episodes, 64), bool)})
        shared.write_json(path / f"train_{c}_actions.json", {"actions": [[[] for _ in range(64)] for _ in range(episodes)]})
    shared.helpers.write_npz(path / "train_finals.npz", {"fg": fg[:, -1], "bg": bg[:, -1]})
    shared.write_json(path / "train_scenes.json", [
        {"final_hash": f"scene{offset + e}", "kinds": ["tower"], "n_struct": 1} for e in range(episodes)])
    shared.write_json(path / "train_summary.json", {
        "profile": "simple_to_complex", "layout": {"formation": 32, "clip": 8, "state_clips": 8, "mixed_clips": 4},
        "conditions": {c: {"episodes": episodes} for c in shared.CONDITIONS}})


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setattr(shared, "EPISODES", 3)
    monkeypatch.setattr(shared, "TRAIN_DEFAULTS", {**shared.TRAIN_DEFAULTS, "dim": 12, "depth": 1, "heads": 3})
    monkeypatch.setattr(shared.common, "RESULTS_ROOT", tmp_path / "results")
    data, val = tmp_path / "data", tmp_path / "val"
    small_data(data)
    small_data(val, offset=8)
    bench = tmp_path / "bench.npz"
    shared.helpers.write_npz(bench, {"split_ids": np.arange(4)})
    shared.write_json(bench.with_suffix(".json"), {"splits": list(shared.common.SPLITS)})
    src = tmp_path / "source03"
    src.mkdir()
    shared.write_json(src / "run.json", {"exp": "03_main"})
    audited = {"inputs": {"data": str(data), "val": str(val), "bench": str(bench)},
        "hashes": {"data": main_shared.data_hashes(data), "val": main_shared.data_hashes(val),
                   "bench": main_shared.bench_hashes(bench)}, "protocol": {"episodes": 3},
        "train_validation_overlap": 0, "train_bench_overlap": 0, "validation_bench_overlap": 0}
    shared.write_json(src / "audit.json", audited)
    bundle = shared.validate_source(str(src), (1, 3), (0, 1), 2)
    audit = shared.prefix_audit(bundle)
    out = shared.bind_run("unit", bundle, audit)
    rows = make_jobs.save_jobs("unit", audit)
    return bundle, audit, out, rows


@pytest.mark.parametrize("condition", shared.CONDITIONS)
@pytest.mark.parametrize("n", shared.SIZES)
def test_default_equal_updates(condition, n):
    b = shared.budget(condition, n)
    assert b["steps"] == 94400 and b["samples"] == 24166400 and b["epochs"] is None
    assert b["pairs"] == n * (56 if condition in ("state", "partial") else 59)


@pytest.mark.parametrize("sizes", [[], [5120], [81921, 90000], [0, 1], [2, 1], [1, 1], [1, 2.5], [1, np.inf]])
def test_invalid_sizes(sizes):
    with pytest.raises(ValueError):
        shared.protocol(sizes=sizes)


@pytest.mark.parametrize("seeds", [[], [-1], [1, 0], [0, 0], [True], [np.nan], [None]])
def test_invalid_seeds(seeds):
    with pytest.raises(ValueError):
        shared.protocol(seeds=seeds)


@pytest.mark.parametrize("steps", [0, -1, 2.5, True, np.inf])
def test_invalid_steps(steps):
    with pytest.raises(ValueError):
        shared.protocol(steps=steps)


def test_default_design_has_sixty_distinct_new_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(shared.common, "RESULTS_ROOT", tmp_path)
    audited = {"protocol": shared.protocol(), "inputs": {"data": "/frozen/data", "val": "/frozen/val"}}
    rows = make_jobs.jobs("new", audited)
    assert len(rows) == len({r["id"] for r in rows}) == 60
    assert sum(r["episodes"] == 81920 for r in rows) == 12
    for row in rows:
        args = shared.expected_args(row, audited)
        assert not args["select_best_val"] and args["epochs"] is None and not args["balanced_phases"]
        assert args["protocol"] == "video" and args["val_every"] == 500
        assert args["steps"] == 94400 and args["episodes"] == row["episodes"]


@pytest.mark.parametrize("name", ["", ".", "..", "a/b", "../escape"])
def test_bad_output_name(name):
    with pytest.raises(ValueError):
        shared.run_path(name)


def test_output_symlink_rejected(tmp_path, monkeypatch):
    root, target = tmp_path / "results", tmp_path / "target"
    (root / shared.EXP_NAME).mkdir(parents=True)
    target.mkdir()
    (root / shared.EXP_NAME / "bad").symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(shared.common, "RESULTS_ROOT", root)
    with pytest.raises(ValueError, match="symbolic"):
        shared.run_path("bad")


def test_source_data_is_reused_and_prefixes_nested(source):
    bundle, audited, _, rows = source
    assert shared.verify_audit("unit") == audited
    assert audited["prefixes"]["1"]["selection"] == {"start": 0, "stop": 1, "step": 1}
    assert audited["prefixes"]["3"]["unique_final_structures"] == 3
    for c in shared.CONDITIONS:
        small = shared.common.load_condition(bundle["inputs"]["data"], c, 1)
        large = shared.common.load_condition(bundle["inputs"]["data"], c, 3)
        assert np.array_equal(small["fg"], large["fg"][:1])
        pairs = shared.common.make_pairs(bundle["inputs"]["data"], c, "video", 1)
        assert len(pairs) == (56 if c in ("state", "partial") else 59)
        boundaries = [0, 8, 16, 24, 32, 40, 48, 56] if c in ("state", "partial") else [0, 32, 40, 48, 56]
        assert not np.isin(pairs[:, 1], boundaries).any()
    assert len(rows) == 16


def test_changed_source_refused(source):
    bundle, _, _, _ = source
    path = bundle["inputs"]["data"] / "train_state_actions.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="changed"):
        shared.verify_audit("unit")


def test_changed_run_budget_refused(source):
    bundle, audited, _, _ = source
    altered = deepcopy(bundle)
    altered["manifest"]["protocol"]["steps"] = 3
    with pytest.raises(ValueError, match="changed"):
        shared.bind_run("unit", altered, audited)


def test_changed_job_list_refused(source):
    _, audit, out, rows = source
    rows[0]["steps"] += 1
    shared.write_json(out / "jobs/data_scale.json", rows)
    with pytest.raises(ValueError, match="job list"):
        run_queue.checked_jobs("unit", audit)


@pytest.mark.parametrize("condition", shared.CONDITIONS)
@pytest.mark.parametrize("n", [1, 3])
@pytest.mark.parametrize("seed", [0, 1])
def test_real_cpu_training_all_small_cells(source, condition, n, seed):
    _, audited, _, rows = source
    row = next(r for r in rows if (r["condition"], r["episodes"], r["seed"]) == (condition, n, seed))
    done = train_scale.train_task(row, audited)
    out = shared.resolve_file(row["out"])
    cfg = shared.common.load_json(out / "config.json")
    ck = torch.load(out / "ckpt.pt", map_location="cpu", weights_only=True)
    assert done["samples_seen"] == 512 and cfg["episodes_used"] == n
    assert ck["step"] == done["checkpoint_step"] == 2
    assert ck["selection"] == done["selection"] == "last" and ck["weights"] == "ema"
    assert not cfg["args"]["select_best_val"] and not cfg["args"]["balanced_phases"]
    assert shared.validate_done(out, row, audited) == done
    before = shared.common.sha1_file(out / "ckpt.pt")
    assert train_scale.train_task(row, audited) == done
    assert shared.common.sha1_file(out / "ckpt.pt") == before


@pytest.mark.parametrize("file", ["config.json", "done.json", "ckpt.pt", "train_complete.json"])
def test_changed_training_identity_is_rejected(source, file):
    _, audit, _, rows = source
    row = rows[0]
    train_scale.train_task(row, audit)
    path = shared.resolve_file(row["out"]) / file
    with path.open("ab") as stream:
        stream.write(b" ")
    if file == "train_complete.json":
        marker = shared.common.load_json(path)
        marker["checkpoint_step"] = 1
        shared.write_json(path, marker)
    with pytest.raises(ValueError):
        shared.validate_done(path.parent, row, audit)


def test_final_only_preflight_has_no_partial_launch(source, monkeypatch):
    _, audited, _, rows = source
    calls = []
    monkeypatch.setattr(run_queue, "verify_audit", lambda name: audited)
    def validate(out, row, audit):
        calls.append(row["id"])
        if row == rows[2]:
            raise ValueError("missing final checkpoint")
        return {}
    monkeypatch.setattr(run_queue, "validate_done", validate)
    monkeypatch.setattr(run_queue.subprocess, "Popen", lambda *a, **k: pytest.fail("preflight started a child"))
    with pytest.raises(ValueError, match="missing"):
        run_queue.execute_queue("unit", evaluate_only=True)
    assert calls == [r["id"] for r in rows[:3]]


def test_eval_only_never_falls_back_to_training(source, monkeypatch):
    _, audit, _, rows = source
    monkeypatch.setattr(run_queue, "validate_done", lambda *args: {})
    monkeypatch.setattr(run_queue.subprocess, "Popen", lambda *a, **k: pytest.fail("training launched"))
    assert not run_queue.run_job(rows[0], "cpu", audit, evaluate_only=True)


def test_eval_only_launches_evaluation_and_preserves_checkpoint(source, monkeypatch):
    _, audit, _, rows = source
    row = rows[0]
    done = train_scale.train_task(row, audit)
    out = shared.resolve_file(row["out"])
    commands = []
    before = shared.common.sha1_file(out / "ckpt.pt")
    def process(command, **kwargs):
        commands.append((command, kwargs))
        return SimpleNamespace(poll=lambda: 0, returncode=0)
    monkeypatch.setattr(run_queue.subprocess, "Popen", process)
    monkeypatch.setattr(run_queue, "record_eval", lambda p, a, d: shared.write_json(p / "eval_complete.json", {"done": True}))
    assert run_queue.run_job(row, "3", audit, evaluate_only=True)
    assert len(commands) == 1 and commands[0][0][2].endswith(".evaluate")
    assert commands[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == "3"
    assert shared.common.sha1_file(out / "ckpt.pt") == before
    assert done["selection"] == "last"


def test_main_dry_run_writes_nothing(source, monkeypatch):
    bundle, audit, _, _ = source
    monkeypatch.setattr(main, "validate_source", lambda *a: bundle)
    monkeypatch.setattr(main, "prefix_audit", lambda b: audit)
    monkeypatch.setattr(main, "protocol", lambda: audit["protocol"])
    monkeypatch.setattr(main, "bind_run", lambda *a: pytest.fail("dry-run bound a run"))
    monkeypatch.setattr(main, "save_jobs", lambda *a: pytest.fail("dry-run wrote jobs"))
    monkeypatch.setattr(main, "execute_queue", lambda *a: pytest.fail("dry-run started training"))
    main.main(["--run", "readonly", "--dry-run"])
    assert not shared.run_path("readonly").exists()


def test_evaluation_dry_run_never_executes_queue(source, monkeypatch):
    _, audit, _, _ = source
    monkeypatch.setattr(evaluate_final, "verify_audit", lambda name: audit)
    monkeypatch.setattr(evaluate_final, "preflight_final", lambda *a: None)
    monkeypatch.setattr(evaluate_final, "execute_queue", lambda *a, **k: pytest.fail("dry-run executed queue"))
    evaluate_final.main(["--run", "unit", "--dry-run"])


def artificial_scores(sizes=(5120, 10240, 20480), seeds=(0, 1, 2)):
    ids = np.repeat(np.arange(4), 8)
    samples = {}
    for j, n in enumerate(sizes):
        for c in shared.CONDITIONS:
            slope = .04 if c == "process" else .01
            for s in seeds:
                score = .25 + j * slope + s * .01 + np.linspace(0, .02, len(ids))
                samples[(n, c, s)] = {m: score.copy() for m in shared.METRICS}
    return samples, ids, sizes, seeds


def test_seed_average_paired_gains_and_slope():
    samples, ids, sizes, seeds = artificial_scores()
    report = analysis.summarize(samples, ids, sizes, seeds, 94400, boot=100)
    block = report["by_split"]["id"]["metrics"]["cascade_f1"]
    assert block["slopes_per_doubling"]["process"]["mean"] == pytest.approx(.04)
    assert block["slopes_per_doubling"]["state"]["mean"] == pytest.approx(.01)
    gap = block["contrasts"]["process_minus_state"]
    assert gap["by_size"]["20480"]["mean"] == pytest.approx(.06)
    assert gap["slope_difference"]["mean"] == pytest.approx(.03)
    assert gap["largest_minus_smallest_gain"]["mean"] == pytest.approx(.06)
    assert block["curves"]["state"]["5120"]["seed_std"] == pytest.approx(.01)
    assert gap["by_size"]["20480"]["seed_std"] == pytest.approx(0, abs=1e-12)
    assert gap["by_size"]["20480"]["n_cases"] == 8


def test_single_seed_std_missing_and_bootstrap_reproducible():
    args = artificial_scores(seeds=(0,))
    first = analysis.summarize(*args[:2], args[2], args[3], 94400, boot=100)
    assert first == analysis.summarize(*args[:2], args[2], args[3], 94400, boot=100)
    block = first["by_split"]["id"]["metrics"]["cascade_f1"]
    assert block["curves"]["state"]["5120"]["seed_std"] is None
    assert block["contrasts"]["process_minus_state"]["largest_minus_smallest_gain"]["seed_std"] is None


@pytest.mark.parametrize("empty", [False, True])
def test_endpoint_gain_change_reports_seed_std(empty):
    samples, ids, sizes, seeds = artificial_scores()
    for s in seeds:
        for metric in shared.METRICS:
            samples[(sizes[-1], "process", s)][metric] += s * .02
    if empty:
        for values in samples.values():
            values["rank_acc"][:] = np.nan
    report = analysis.summarize(samples, ids, sizes, seeds, 94400, boot=20)
    for block in report["by_split"].values():
        for metric in shared.METRICS:
            for contrast in block["metrics"][metric]["contrasts"].values():
                endpoint = contrast["largest_minus_smallest_gain"]
                if empty and metric == "rank_acc":
                    assert endpoint["seed_std"] is None and endpoint["mean"] is None
                else:
                    assert endpoint["seed_std"] == pytest.approx(.02)
                    assert endpoint["mean"] == pytest.approx(.08)


def test_missing_cell_not_treated_as_zero():
    samples, ids, sizes, seeds = artificial_scores()
    samples.pop(next(iter(samples)))
    with pytest.raises(ValueError, match="present"):
        analysis.summarize(samples, ids, sizes, seeds, 94400)


def test_invalid_scores_refused():
    samples, ids, sizes, seeds = artificial_scores()
    samples[next(iter(samples))]["cascade_f1"][0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        analysis.summarize(samples, ids, sizes, seeds, 94400)


def test_invalid_rank_cases_use_common_scene_mask():
    samples, ids, sizes, seeds = artificial_scores()
    for value in samples.values():
        value["rank_acc"][0] = np.nan
    report = analysis.summarize(samples, ids, sizes, seeds, 94400, boot=20)
    block = report["by_split"]["id"]["metrics"]
    assert block["rank_acc"]["n_cases"] == 7 and block["cascade_f1"]["n_cases"] == 8


def test_markdown_and_both_plots(tmp_path):
    samples, ids, sizes, seeds = artificial_scores()
    report = analysis.summarize(samples, ids, sizes, seeds, 94400, boot=20)
    report.update(planned_training_seeds=list(seeds), missing_cells=[], ignored_completed_cells=[],
                  prefixes={str(n): {"unique_final_structures": n - 10, "duplicate_episodes": 10} for n in sizes})
    text = analysis.markdown(report)
    assert "process − state" in text and "log2(N)" in text and "不复用旧模型" in text
    analysis.plot(report, tmp_path)
    assert all((tmp_path / f"{name}.{ext}").is_file() for name in ("data_scale", "process_state_gain") for ext in ("png", "pdf"))


@pytest.mark.parametrize("cards", ["", "0,0", "-1", "gpu", "0,"])
def test_invalid_cards(cards):
    with pytest.raises(ValueError):
        run_queue.devices(cards)


def test_allow_incomplete_uses_only_globally_complete_seeds(source, monkeypatch):
    _, audit, _, rows = source
    ids = np.repeat(np.arange(4), 2)
    raw = {m: np.full((8, 2), .5) for m in shared.METRICS}
    raw["rank_acc"] = np.ones(8)
    raw["split_ids"] = ids
    for row in rows:
        if row == rows[-1]:
            continue
        out = shared.resolve_file(row["out"])
        out.mkdir(parents=True, exist_ok=True)
        for name in ("train_complete.json", "eval_complete.json", "config.json", "done.json", "eval.json"):
            shared.write_json(out / name, {})
        (out / "ckpt.pt").write_text("fixture")
        shared.helpers.write_npz(out / "eval_samples.npz", raw)
    monkeypatch.setattr(analysis, "validate_done", lambda *a: {"val": {}, "train_raw": {}, "train_ema": {}, "seconds": 1, "samples_seen": 512})
    monkeypatch.setattr(analysis, "validate_eval", lambda *a: None)
    monkeypatch.setattr(analysis.common, "load_bench", lambda *a: {"split_ids": ids})
    with pytest.raises(ValueError, match="incomplete"):
        analysis.load_results("unit", audit)
    samples, _, seeds, missing, ignored, _, _ = analysis.load_results("unit", audit, True)
    assert seeds == [0] and missing == [rows[-1]["id"]] and len(ignored) == 7
    assert len(samples) == 8 and all(key[2] == 0 for key in samples)
    (shared.resolve_file(rows[0]["out"]) / "eval_complete.json").unlink()
    with pytest.raises(ValueError, match="no training seed"):
        analysis.load_results("unit", audit, True)


@pytest.mark.parametrize("evaluated", [True, False])
def test_incomplete_flag_does_not_hide_corrupt_committed_results(source, monkeypatch, evaluated):
    _, audit, _, rows = source
    out = shared.resolve_file(rows[0]["out"])
    out.mkdir(parents=True)
    shared.write_json(out / "train_complete.json", {})
    if evaluated:
        shared.write_json(out / "eval_complete.json", {})
    monkeypatch.setattr(analysis, "validate_done", lambda *a: (_ for _ in ()).throw(ValueError("corrupt committed result")))
    with pytest.raises(ValueError, match="corrupt"):
        analysis.load_results("unit", audit, True)


def test_committed_evaluation_without_training_is_not_plain_missing(source):
    _, audit, _, rows = source
    out = shared.resolve_file(rows[0]["out"])
    out.mkdir(parents=True)
    shared.write_json(out / "eval_complete.json", {})
    with pytest.raises(ValueError, match="without committed training"):
        analysis.load_results("unit", audit, True)


@pytest.mark.parametrize("field,value", [("selection", "val_cascade_acc"), ("checkpoint_step", 1), ("samples_seen", 256)])
def test_budget_and_selection_rechecked_even_with_fresh_marker(source, field, value):
    _, audit, _, rows = source
    row = rows[0]
    train_scale.train_task(row, audit)
    out = shared.resolve_file(row["out"])
    done = shared.common.load_json(out / "done.json")
    done[field] = value
    shared.write_json(out / "done.json", done)
    shared.write_json(out / "train_complete.json", shared.training_identity(out, row, audit))
    with pytest.raises(ValueError, match="budget"):
        shared.validate_done(out, row, audit)


@pytest.mark.parametrize("field,value", [("n_samples", 511), ("checkpoint_sha1", "wrong"), ("bench_sha1", "wrong"), ("bench_json_sha1", "wrong")])
def test_wrong_evaluation_identity_rejected_before_scoring(source, field, value):
    _, audit, _, rows = source
    row = rows[0]
    out = shared.resolve_file(row["out"])
    out.mkdir(parents=True)
    bp = shared.resolve_file(audit["inputs"]["bench"])
    done = {"ckpt_sha1": "final", "checkpoint_step": 2}
    ev = {"bench": shared.common.rel(bp), "n_samples": 512, "checkpoint_sha1": "final",
          "bench_sha1": audit["hashes"]["bench"][bp.name], "bench_json_sha1": audit["hashes"]["bench"][bp.with_suffix(".json").name]}
    ev[field] = value
    shared.write_json(out / "eval.json", ev)
    with pytest.raises(ValueError, match="evaluation differs"):
        shared.validate_eval(out, audit, done)


def test_cpu_checkpoint_rollout_and_metric_recomputation(source):
    _, audit, _, rows = source
    row = rows[0]
    train_scale.train_task(row, audit)
    evaluator = import_module(f"{shared.PILOT_PACKAGE}.evaluate")
    fg = np.zeros((4, shared.common.H, shared.common.W), dtype=np.uint8)
    fg[:, -1] = 1
    bg = np.zeros_like(fg)
    target_f = np.repeat(np.repeat(fg[:, None, None], 2, axis=1), 32, axis=2)
    target_b = np.zeros_like(target_f)
    bench = {"ctx_fg": fg, "ctx_bg": bg, "fut_fg": target_f, "fut_bg": target_b,
        "ivs": [[{"steps": []}, {"steps": []}] for _ in range(4)], "t_last": np.zeros((4, 2), int),
        "A": np.zeros((4, 2, shared.common.H, shared.common.W), bool), "split_ids": np.arange(4),
        "rank_valid": np.zeros(4, bool), "labels": np.zeros((4, 2)), "cascade": np.zeros((4, 2))}
    model, ck = evaluator.load_run(shared.resolve_file(row["out"]), torch.device("cpu"))
    pf, pb = evaluator.predict_bench_model(model, bench, torch.device("cpu"), batch=4)
    raw = evaluator.score_bench(bench, pf, pb)
    validator = import_module("preresearch.05_formation_ratio.exp.shared").validate_metrics
    validator(raw, bench)
    assert ck["step"] == 2 and raw["cascade_f1"].shape == (4, 2)
    raw["cascade_f1"][0, 0] = 1 - raw["cascade_f1"][0, 0]
    with pytest.raises(ValueError, match="reproduced"):
        validator(raw, bench)


def test_analysis_publishes_json_markdown_and_metadata(source, monkeypatch):
    _, audit, _, _ = source
    samples, ids, _, seeds = artificial_scores(sizes=(1, 3), seeds=(0, 1))
    monkeypatch.setattr(analysis, "verify_audit", lambda name: audit)
    monkeypatch.setattr(analysis, "load_results", lambda *a: (samples, ids, list(seeds), [], [], {}, {}))
    out = analysis.analyze("unit", boot=20)
    report = shared.common.load_json(out / "data_scale.json")
    meta = shared.common.load_json(out / "meta.json")
    assert report["complete_design"] and report["missing_cells"] == []
    assert meta["exp"] == shared.EXP_NAME and meta["run"] == "unit"
    assert (out / "data_scale.md").is_file() and (out / "figs/data_scale.pdf").is_file()
