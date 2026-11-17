from importlib import import_module
import json
from pathlib import Path

import numpy as np
import pytest
import torch

PACKAGE = "preresearch.09_physics_rollout.exp"
shared = import_module(f"{PACKAGE}.shared")
measurement = import_module(f"{PACKAGE}.measurement")
evaluation = import_module(f"{PACKAGE}.evaluation")
analysis = import_module(f"{PACKAGE}.analysis")
sources = import_module(f"{PACKAGE}.sources")
runner = import_module(f"{PACKAGE}.run")
import datasets as D
from core import SAND, STONE, check_violations


class LiteralModel(torch.nn.Module):
    def __init__(self, ignore_actions=False):
        super().__init__()
        self.calls = 0
        self.inputs = []
        self.ignore_actions = ignore_actions

    def step(self, fg, bg, action):
        self.calls += 1
        self.inputs.append(fg.clone())
        out = fg.clone()
        if not self.ignore_actions:
            place = (action > 0) & (((action - 1) // 42) == 0)
            material = ((action - 1) // 2) % 21
            out[place] = material[place]
        return out, bg.clone()


@pytest.fixture
def formation():
    initial = D.terrain()
    bg = np.zeros_like(initial)
    actions = [[[{"type": "place", "x": 10 + index, "y": D.SURF - 1, "material": STONE}]] +
               [[] for _ in range(30)] for index in range(2)]
    results = [measurement.engine_script(initial, bg, script) for script in actions]
    return {"fg": np.stack([np.concatenate([initial[None], row[0]]) for row in results]),
            "bg": np.stack([np.concatenate([bg[None], row[1]]) for row in results]),
            "actions": actions, "episode_ids": np.arange(2)}


@pytest.fixture
def bench(formation):
    fg, bg = formation["fg"][:, -1], formation["bg"][:, -1]
    ivs = [[{"steps": [{"t": 0, "actions": []}]} for _ in range(2)] for _ in fg]
    target_f = np.broadcast_to(fg[:, None, None], (2, 2, 32, shared.common.H, shared.common.W)).copy()
    target_b = np.broadcast_to(bg[:, None, None], target_f.shape).copy()
    return {"ctx_fg": fg, "ctx_bg": bg, "fut_fg": target_f, "fut_bg": target_b,
            "ivs": ivs, "A": np.zeros((2, 2, shared.common.H, shared.common.W), dtype=bool),
            "case_ids": np.arange(2), "split_ids": np.zeros(2, dtype=np.int8), "splits": ["id"]}


def test_full_rollout_never_stops_at_a_fixed_point():
    fg = D.terrain()[None]
    bg = np.zeros_like(fg)
    scripts = [[[] for _ in range(32)]]
    scripts[0][10] = [{"type": "place", "x": 10, "y": D.SURF - 1, "material": STONE}]
    model = LiteralModel()
    _, pf, pb = next(evaluation.predict_batches(model, fg, bg, scripts, 32, 1, "cpu"))
    assert model.calls == 32
    assert pf[0, 9, D.SURF - 1, 10] == 0
    assert pf[0, 10, D.SURF - 1, 10] == STONE
    assert pf.shape == pb.shape == (1, 32, shared.common.H, shared.common.W)


def test_noop_runs_all_128_steps(bench, formation):
    model = LiteralModel()
    scores, _ = evaluation.evaluate_task("drift", bench, formation, model, batch=2)
    assert model.calls == 128
    assert scores["exact"].shape == (2, 1, 128)
    assert not scores["ever_changed"].any()
    assert not scores["any_violation"].any()


def test_formation_actions_align_and_no_teacher_forcing(bench, formation):
    correct = LiteralModel()
    scores, trajectories = evaluation.evaluate_task("formation", bench, formation, correct, batch=2, save_trajectories=True)
    assert correct.calls == 31
    np.testing.assert_array_equal(trajectories["fg"][:, 0], formation["fg"][:, 1:])
    assert scores["exact"][..., -1].mean() == 1
    incorrect = LiteralModel(ignore_actions=True)
    scores, _ = evaluation.evaluate_task("formation", bench, formation, incorrect, batch=2)
    assert incorrect.calls == 31
    assert not scores["exact"][..., -1].any()
    assert all(torch.equal(row, incorrect.inputs[0]) for row in incorrect.inputs)


def test_ground_truth_audit_zero_and_oracle_exact(bench, formation):
    arrays, report = evaluation.audit_targets(bench, formation)
    assert report["intervention"]["all_ticks_zero"]
    assert not arrays["intervention"]["any_violation"].any()
    outputs = {task: evaluation.evaluate_task(task, bench, formation)[0] for task in runner.TASKS}
    checks = evaluation.oracle_checks(outputs)
    assert checks["formation_exact"] == 1
    assert checks["formation_probe_passed"]
    assert checks["noop_engine_static"]
    assert not checks["formation_is_h1_evidence"]


def test_real_engine_transients_are_not_silently_declared_violation_free():
    initial = D.terrain()
    initial[2, 5] = SAND
    f, b, _ = measurement.engine_script(initial, np.zeros_like(initial), [[]])
    assert check_violations(f[0], b[0])["floating_sand"] > 0
    arrays = measurement.audit_trajectory(f[None, None], b[None, None], initial[None], np.zeros_like(initial)[None])
    assert not measurement.audit_summary(arrays)["all_ticks_zero"]


def test_score_matches_existing_metrics_and_preserves_arrays(formation):
    fg, bg = formation["fg"][0, 1:], formation["bg"][0, 1:]
    initial_f, initial_b = formation["fg"][0, 0], formation["bg"][0, 0]
    pf, pb = fg.copy(), bg.copy()
    pf[0, D.SURF - 1, 10] = 0
    original = pf.copy()
    result = measurement.score_trajectory(pf, pb, fg, bg, initial_f, initial_b)
    expected = shared.metrics.score_sample((pf[0], pb[0]), (fg[0], bg[0]),
                                          (initial_f, initial_b), np.zeros_like(initial_f, dtype=bool))
    for key in ("cell_acc_all", "cell_acc_mask", "cascade_f1", "change_f1", "mass_error", "exact"):
        assert result[key][0] == expected[key]
    assert result["mass_error"][0] == 1
    np.testing.assert_array_equal(original, pf)


def test_ever_changed_detects_return_to_initial():
    initial = D.terrain()
    f = np.stack([initial.copy(), initial.copy()])
    f[0, D.SURF - 1, 7] = STONE
    bg = np.zeros_like(f)
    targets = np.stack([initial, initial])
    scores = measurement.score_trajectory(f, bg, targets, bg, initial, np.zeros_like(initial))
    assert scores["state_changed"].tolist() == [1, 0]
    assert scores["ever_changed"].tolist() == [1, 1]


def test_load_formation_uses_destination_frame_actions(tmp_path, formation):
    fg, bg = formation["fg"], formation["bg"]
    np.savez(tmp_path / "train_process.npz", fg=fg, bg=bg,
             phase=np.zeros(fg.shape[:2], dtype=np.int8), cont=np.ones(fg.shape[:2], dtype=bool))
    source = {"condition": "process", "actions": [[[]] + script for script in formation["actions"]]}
    (tmp_path / "train_process_actions.json").write_text(json.dumps(source))
    loaded = measurement.load_formation(tmp_path)
    assert loaded["actions"] == formation["actions"]
    source["condition"] = "shuffle"
    (tmp_path / "train_process_actions.json").write_text(json.dumps(source))
    with pytest.raises(ValueError, match="Only the shared"):
        measurement.load_formation(tmp_path)


def test_missing_repetitions_and_duplicates_fail_explicitly():
    rows = [{"condition": c, "seed": 0} for c in shared.CONDITIONS]
    with pytest.raises(ValueError, match="Missing="):
        sources.require_pairs(rows, [0, 1, 2])
    sources.require_pairs(rows, [0])
    with pytest.raises(ValueError, match="Duplicate"):
        sources.require_pairs(rows + [rows[0]], [0])


def test_seed_std_is_not_scene_std():
    cube = np.array([[[0., 0.], [0., 0.]], [[1., 1.], [1., 1.]], [[.5, .5], [.5, .5]]])
    draws = np.random.default_rng(0).integers(2, size=(100, 2))
    result = analysis.summarize_cube(cube, [0, 1, 2], draws)
    np.testing.assert_allclose(result["mean"], .5)
    np.testing.assert_allclose(result["std"], .5)
    np.testing.assert_allclose(result["final_scene_95ci"], [.5, .5])
    assert analysis.summarize_cube(cube[:1], [0], draws)["std"] is None


def test_scene_pairing_and_constant_paired_difference():
    raw = {}
    for c in shared.CONDITIONS:
        for seed in (0, 1, 2):
            values = {key: np.zeros((4, 2, 32)) for key in shared.METRICS}
            if c == "process":
                values["exact"][:] = .2
            values.update(case_ids=np.arange(4), split_ids=np.array([0, 0, 1, 1]), ticks=np.arange(1, 33))
            raw[(c, seed)] = values
    oracle = {key: value.copy() for key, value in raw[("state", 0)].items()}
    output = analysis.summarize_task(raw, oracle, [0, 1, 2], ["id", "scale_ood"], 50, 2027)
    paired = output["groups"]["all"]["paired_differences"]["process-state"]["exact"]
    np.testing.assert_allclose(paired["final_scene_95ci"], [.2, .2])
    assert paired["n_scenes"] == 4
    assert output["groups"]["scale_ood"]["n_scenes"] == 2
    raw[("process", 2)]["case_ids"] = np.arange(4)[::-1]
    with pytest.raises(ValueError, match="ordering"):
        analysis.summarize_task(raw, oracle, [0, 1, 2], ["id", "scale_ood"], 50, 2027)


def test_cache_identity_and_atomic_outputs(tmp_path):
    path = tmp_path / "scores.npz"
    identity = {"checkpoint": "frozen"}
    evaluation.save_task(path, identity, {"x": np.ones(2)}, None)
    assert evaluation.cached(path, identity)
    with pytest.raises(ValueError, match="protocol differs"):
        evaluation.cached(path, {"checkpoint": "changed"})
    path.write_bytes(b"not the original result")
    with pytest.raises(ValueError, match="identity changed"):
        evaluation.cached(path, identity)
    assert not list(tmp_path.glob("*.tmp"))


def test_output_never_overwrites_sources(tmp_path):
    with pytest.raises(ValueError, match="overlaps"):
        shared.protect_output(tmp_path / "model/results", [tmp_path / "model"])
    shared.protect_output(tmp_path / "new", [tmp_path / "model"])


def test_one_command_analysis_exports_all_measured_artifacts(tmp_path, bench, formation):
    source = {"version": 1, "seeds": [0], "splits": ["id"],
              "models": [{"condition": c, "seed": 0} for c in shared.CONDITIONS]}
    oracles = {}
    for task in runner.TASKS:
        values, _ = evaluation.evaluate_task(task, bench, formation)
        oracles[task] = values
        evaluation.save_task(tmp_path / "raw/oracle_reseed" / f"{task}.npz",
                             {"protocol": source, "task": task, "model": "oracle_reseed"}, values, None)
        for row in source["models"]:
            evaluation.save_task(tmp_path / "raw" / f"{row['condition']}_seed0" / f"{task}.npz",
                                 {"protocol": source, "task": task, "model": row}, values, None)
    truths, _ = evaluation.audit_targets(bench, formation)
    truths["intervention"]["split_ids"] = bench["split_ids"]
    shared.write_npz(tmp_path / "audit_intervention.npz", truths["intervention"])
    shared.write_json(tmp_path / "audit.json", {"qualified": False, "checks": evaluation.oracle_checks(oracles)})
    report = analysis.analyze(tmp_path, source, runner.TASKS, n_boot=20, make_plots=True)
    assert report["tasks"]["formation"]["groups"]["all"]["conditions"]["process"]["exact"]["final_mean"] == 1
    for name in ("results.json", "complete.json", "tables/endpoints.csv", "tables/curves.csv", "tables/endpoints.tex",
                 "figures/intervention.pdf", "figures/drift.png", "figures/formation.pdf", "figures/fig8.pdf"):
        assert (tmp_path / "analysis" / name).is_file()


def test_existing_03_sources_are_not_counted_as_three_repetitions():
    source = shared.common.RESULTS_ROOT / "03_main/20261111-044057_four_conditions"
    if not source.exists():
        pytest.skip("Frozen local experiment 03 is not installed")
    with pytest.raises(ValueError, match="Missing="):
        sources.load_sources([str(source)], [0, 1, 2])
    audit, rows, _ = sources.load_sources([str(source)], [0])
    assert len(rows) == 4
    assert all(row["seed"] == 0 for row in rows)
    assert audit["protocol"]["episodes"] == 81920
