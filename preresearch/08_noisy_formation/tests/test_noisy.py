from importlib import import_module
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

S = import_module("preresearch.08_noisy_formation.exp.shared")
G = import_module("preresearch.08_noisy_formation.exp.generate")
N = import_module("preresearch.08_noisy_formation.exp.noise")
A = import_module("preresearch.08_noisy_formation.exp.analysis")
T = import_module("preresearch.08_noisy_formation.exp.train")
Q = import_module("preresearch.08_noisy_formation.exp.run_queue")


@pytest.fixture(scope="module")
def episode():
    r = N.v3.make_episode_set(0, 0, p_mistake=0, scaffold=False, profile="simple_to_complex")
    return r["episodes"]["process"]


@pytest.mark.parametrize("p,eligible", [(0, True), (.1, False), (.3, False)])
def test_clean_and_fluid_bitwise(episode, p, eligible):
    f, b, a, meta = N.derive(episode["fg"], episode["bg"], episode["actions"], N.D.int_seed(0, "world", 0), p, eligible=eligible)
    np.testing.assert_array_equal(f, episode["fg"][:32])
    np.testing.assert_array_equal(b, episode["bg"][:32])
    assert a == episode["actions"][:32]
    assert not meta["fallback"] and not meta["effective_episode"]


@pytest.mark.parametrize("p", [.1, .3])
def test_noise_replay_deterministic_final_and_tick_budget(episode, p, monkeypatch):
    args = (episode["fg"], episode["bg"], episode["actions"], N.D.int_seed(0, "world", 0), p)
    original = N.core.FormWorld.tick
    ticks = []

    def count(world):
        ticks.append(world)
        return original(world)

    monkeypatch.setattr(N.core.FormWorld, "tick", count)
    first = N.derive(*args)
    worlds = {}
    for w in ticks:
        worlds[id(w)] = worlds.get(id(w), 0) + 1
    assert list(worlds.values()).count(31) >= 2
    second = N.derive(*args)
    f, b, actions, meta = first
    np.testing.assert_array_equal(f, second[0])
    np.testing.assert_array_equal(b, second[1])
    assert actions == second[2] and meta == second[3]
    G.check_pair(*args[:3], f, b, actions)
    assert not meta["fallback"]
    N.replay(f, b, actions, args[3])
    assert all(not actions[t] for t in range(28, 32))


@pytest.mark.parametrize("p", [-.1, 1.1, float("nan"), float("inf")])
def test_invalid_p(episode, p):
    with pytest.raises(ValueError, match="probability"):
        N.derive(episode["fg"], episode["bg"], episode["actions"], 0, p)


def test_source_replay_remains_strict(episode):
    actions = [list(fr) for fr in episode["actions"]]
    actions[1] = [N.v3._a("remove", 0, 0)]
    with pytest.raises(ValueError):
        N.replay(episode["fg"], episode["bg"], actions, N.D.int_seed(0, "world", 0))


def test_token_conflict_rejected():
    w = N.D.new_world(0)
    actions = [N.v3._a("place", 2, 2, N.core.WOOD), N.v3._a("place", 2, 2, N.core.WALL, "back")]
    with pytest.raises(ValueError, match="same-cell"):
        N.execute(w, actions, strict=False)


def test_repair_fg_bg_conflict_deferred(episode):
    w = N.D.new_world(N.D.int_seed(0, "world", 0))
    f, b = w.grid.copy(), w.background.copy()
    x = 4
    y = int(np.nonzero((w.grid[:-1, x] == 0) & np.isin(w.grid[1:, x], list(N.core.LOAD)))[0][-1])
    f[y, x], b[y, x] = N.core.WOOD, N.core.WALL
    first = N.repair_actions(w, f, b)
    assert len({(a["x"], a["y"]) for a in first}) == len(first)
    assert w.grid[y, x] == N.core.WOOD and w.background[y, x] == 0
    N.finish_tick(w, first)
    second = N.repair_actions(w, f, b)
    N.finish_tick(w, second)
    np.testing.assert_array_equal(w.grid, f)
    np.testing.assert_array_equal(w.background, b)


def test_cached_baseline_does_not_overwrite_noise_metadata(episode, monkeypatch):
    args = (episode["fg"], episode["bg"], episode["actions"], N.D.int_seed(0, "world", 0))
    baseline = N.derive(*args, 0)[3]
    original = N.repair_actions

    def fail(*args):
        raise ValueError("forced repair failure")

    monkeypatch.setattr(N, "repair_actions", fail)
    *_, meta = N.derive(*args, .3, _baseline=baseline)
    assert meta["fallback"] and meta["probability"] == .3
    assert meta["fallback_reason"] == "forced repair failure"
    monkeypatch.setattr(N, "repair_actions", original)


def test_parallel_matches_serial(episode):
    args = [(episode["fg"], episode["bg"], episode["actions"], N.D.int_seed(0, "world", 0), p) for p in (0, .1, .3)]
    serial = list(G.derive_results(args, 1))
    parallel = list(G.derive_results(args, 2))
    for left, right in zip(serial, parallel):
        np.testing.assert_array_equal(left[0], right[0])
        np.testing.assert_array_equal(left[1], right[1])
        assert left[2:] == right[2:]


@pytest.mark.parametrize("fractions,fallback,ok", [((.1, .2, .35), 0, True), ((.1, .2, .29), 0, False),
    ((.2, .1, .35), 0, False), ((.1, .2, .35), .1, False), ((.1, .2, .35), .09, True)])
def test_acceptance(fractions, fallback, ok):
    audits = {v: {"physical_transition_fraction": f, "fallback_rate": fallback} for v, f in zip(S.NOISE, fractions)}
    assert S.qualifies(audits) is ok


@pytest.mark.parametrize("value", [True, 1.2, -1, "1,2", float("nan")])
def test_invalid_integer(value):
    with pytest.raises(ValueError):
        S.integer(value, "seed", 0)


def test_single_task_single_seed_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(S.common, "RESULTS_ROOT", tmp_path)
    p = {"episodes": 12, "steps": 10, "data_paths": {v: str(tmp_path / v) for v in S.VARIANTS}}
    rows = S.jobs("demo", p)
    assert len(rows) == 6 and {r["seed"] for r in rows} == {0}
    assert {r["condition"] for r in rows} == set(S.common.CONDITIONS)
    assert {r["condition"] for r in rows if r["variant"].startswith("process_noisy")} == {"process"}
    with pytest.raises(ValueError, match="single-seed"):
        T.train_task({**rows[0], "seed": 1}, {**p, "training_seeds": [0]})


def test_train_cli_uses_final_uniform_video(tmp_path, monkeypatch):
    monkeypatch.setattr(S.common, "RESULTS_ROOT", tmp_path)
    p = {"episodes": 12, "steps": 10, "batch": 256, "train_defaults": S.DEFAULTS,
         "data_paths": {v: str(tmp_path / v) for v in S.VARIANTS}, "inputs": {"val": str(tmp_path / "val")}}
    a = S.expected_args(S.jobs("demo", p)[-1], p)
    assert a["seed"] == 0 and a["condition"] == "process" and a["protocol"] == "video"
    assert a["steps"] == 10 and a["epochs"] is None and not a["balanced_phases"] and not a["select_best_val"]


def test_full_eval_preflight_stops_before_any_job(monkeypatch):
    rows = [{"out": "/unused/one"}, {"out": "/unused/two"}]
    calls = []

    def check(out, row, p):
        calls.append(out)
        raise ValueError("missing final model")

    monkeypatch.setattr(Q, "validate_done", check)
    with pytest.raises(ValueError, match="missing final"):
        Q.preflight_final(rows, {})
    assert len(calls) == 1


def test_unqualified_data_never_reused_for_training(tmp_path, monkeypatch):
    p = tmp_path / "data"
    p.mkdir()
    audits = {v: {"physical_transition_fraction": .1, "fallback_rate": .2} for v in S.NOISE}
    manifest = {"code": {}, "source": "mock", "source_identity": {}, "episodes": 4,
                "qualified": False, "max_fallback": .1, "audits": audits}
    S.write_json(p / "manifest.json", manifest)
    S.write_json(p / "complete.json", {"manifest_sha1": S.common.sha1_file(p / "manifest.json"), "hashes": S.file_set(p)})
    monkeypatch.setattr(S, "code_identity", lambda: {})
    monkeypatch.setattr(S, "source_bundle", lambda *args: {"manifest": {}})
    assert S.verify_data(p, require_qualified=False)[0] == manifest
    with pytest.raises(ValueError, match="failed noise acceptance"):
        S.verify_data(p)
    with pytest.raises(ValueError, match="failed noise acceptance"):
        S.prepare("demo", p)


def test_extraction_prefix_and_pair_tail(episode, tmp_path):
    np.savez_compressed(tmp_path / "source.npz", **{k: np.stack([episode[k]] * 3) for k in ("fg", "bg", "cont", "phase")})
    work = tmp_path / "work"
    work.mkdir()
    arrays = G.extract_arrays(tmp_path / "source.npz", work, 2)
    for key in arrays:
        assert arrays[key].shape[0] == 2
        np.testing.assert_array_equal(arrays[key][0], episode[key])
    f, b, actions, _ = N.derive(episode["fg"], episode["bg"], episode["actions"], N.D.int_seed(0, "world", 0), .3)
    full_f = np.concatenate((f, episode["fg"][32:]))
    full_b = np.concatenate((b, episode["bg"][32:]))
    np.savez_compressed(tmp_path / "paired.npz", fg=full_f, bg=full_b, cont=episode["cont"], phase=episode["phase"])
    with np.load(tmp_path / "paired.npz") as z:
        for k in ("fg", "bg"):
            np.testing.assert_array_equal(z[k][32:], episode[k][32:])
        for k in ("cont", "phase"):
            np.testing.assert_array_equal(z[k], episode[k])
    assert (actions + episode["actions"][32:])[32:] == episode["actions"][32:]


def test_scene_paired_analysis_and_svg():
    bench = {"split_ids": np.array([0, 0, 1, 1]), "splits": ["id", "scale_ood"],
             "n_changed": np.array([[4, 0], [0, 5], [1, 1], [1, 1]])}
    samples = {v: {m: np.full((4, 2), .2 + .1 * i) if m != "rank_acc" else np.full(4, .5)
                   for m in S.METRICS} for i, v in enumerate(S.VARIANTS)}
    result = A.summarize(samples, bench, n_boot=32)
    delta = result["groups"]["all"]["paired_differences"]["process_noisy_low_minus_state"]["cascade_f1"]
    assert delta["n_cases"] == 4 and delta["mean"] == pytest.approx(.4)
    assert result["groups"]["dependency_proxy_n_changed_ge3"]["estimates"]["state"]["cascade_f1"]["n_cases"] == 2
    assert result["groups"]["all"]["dose_response"][0]["nominal_p"] == 0
    ET.fromstring(A.dose_svg(result))


@pytest.mark.parametrize("name", ["../escape", "", "/absolute", ".", ".."])
def test_run_name_safety(name):
    with pytest.raises(ValueError):
        S.run_path(name)


@pytest.mark.parametrize("kind", ["delay", "offset", "omit"])
def test_each_mistake_kind_has_legal_recovery(episode, kind, monkeypatch):
    original = N.D.rng_for

    class Controlled:
        def __init__(self, rng):
            self.rng = rng

        def random(self):
            return 0.

        def choice(self, values):
            return kind if values == ["delay", "offset", "omit"] else self.rng.choice(values)

        def __getattr__(self, key):
            return getattr(self.rng, key)

    def rng_for(*args):
        rng = original(*args)
        return Controlled(rng) if args[0] == "experiment08" else rng

    monkeypatch.setattr(N.D, "rng_for", rng_for)
    seed = N.D.int_seed(0, "world", 0)
    f, b, actions, meta = N.derive(episode["fg"], episode["bg"], episode["actions"], seed, .3)
    assert meta[kind] > 0 and not meta["fallback"]
    N.replay(f, b, actions, seed)
    np.testing.assert_array_equal(f[31], episode["fg"][31])
    np.testing.assert_array_equal(b[31], episode["bg"][31])


def test_generation_commit_and_cpu_training(episode, tmp_path, monkeypatch):
    import torch
    torch.set_num_threads(2)
    monkeypatch.setattr(import_module(f"{S.PILOT}.train"), "pick_device", lambda _: torch.device("cpu"))
    monkeypatch.setattr(S.common, "RESULTS_ROOT", tmp_path / "results")
    tiny = {**S.DEFAULTS, "dim": 24, "depth": 1, "heads": 3}
    monkeypatch.setattr(S, "DEFAULTS", tiny)
    source = tmp_path / "source"
    val = tmp_path / "val"
    source.mkdir()
    val.mkdir()
    arrays = {k: episode[k][None] for k in ("fg", "bg", "cont", "phase")}
    for condition in S.common.CONDITIONS:
        np.savez_compressed(source / f"train_{condition}.npz", **arrays)
        S.write_json(source / f"train_{condition}_actions.json", {"actions": [episode["actions"]]})
    np.savez_compressed(val / "train_state.npz", **arrays)
    S.write_json(val / "train_state_actions.json", {"actions": [episode["actions"]]})
    summary = {"seed": 0, "layout": {"formation": 32, "clip": 8, "state_clips": 8, "mixed_clips": 4},
               "conditions": {c: {"episodes": 1} for c in S.common.CONDITIONS}}
    S.write_json(source / "train_summary.json", summary)
    S.write_json(val / "train_summary.json", summary)
    S.write_json(source / "train_scenes.json", [{"kinds": ["arch", "tree"]}])
    inputs = {"data": str(source), "val": str(val), "bench": str(tmp_path / "bench.npz")}
    hashes = {"val": S.common.data_hashes(val, "state"), "data": {}, "bench": {}}
    bundle = {"source": source, "inputs": {k: Path(v) for k, v in inputs.items()},
              "manifest": {"inputs": inputs, "hashes": hashes}, "audited": {"inputs": inputs}}
    monkeypatch.setattr(G, "source_bundle", lambda *args: bundle)
    monkeypatch.setattr(S, "source_bundle", lambda *args: bundle)
    monkeypatch.setattr(S.helpers, "forbid_source_output", lambda *args: None)
    out = tmp_path / "results" / S.EXP_NAME / "data_v1"
    try:
        G.generate(out, episodes=1, workers=1)
    except ValueError as exc:
        assert "noise acceptance failed" in str(exc)
    manifest, _ = S.verify_data(out, require_qualified=False)
    assert manifest["pair_audit"]["episodes_checked"] == 1
    assert (out / "pipeline.lock").exists()
    for variant in ("process_noisy_low", "process_noisy_high"):
        with np.load(out / variant / "train_process.npz") as z:
            np.testing.assert_array_equal(z["fg"][0, 32:], episode["fg"][32:])
            np.testing.assert_array_equal(z["bg"][0, 32:], episode["bg"][32:])
            np.testing.assert_array_equal(z["cont"][0], episode["cont"])
            np.testing.assert_array_equal(z["phase"][0], episode["phase"])
    verify = S.verify_data
    monkeypatch.setattr(S, "verify_data", lambda path: verify(path, require_qualified=False))
    run = S.prepare("cpu_smoke", out, episodes=1, steps=1)
    p = S.common.load_json(run / "protocol.json")
    row = S.jobs("cpu_smoke", p)[-1]
    with S.scale.run_lock(S.resolve_file(row["out"])):
        done = T.train_task(row, p)
    assert done["steps"] == 1 and done["samples_seen"] == 256
    assert done["checkpoint_step"] == 1 and done["weights"] == "ema"
    S.validate_done(S.resolve_file(row["out"]), row, p)
    S.verify_run("cpu_smoke")
    altered = {**p, "training_seeds": [0, 1]}
    S.write_json(run / "protocol.json", altered)
    with pytest.raises(ValueError, match="single-seed"):
        S.verify_run("cpu_smoke")


def test_sigterm_waits_for_children_before_unlock(tmp_path, monkeypatch):
    import signal
    monkeypatch.setattr(S.common, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(Q, "verify_run", lambda _: ({}, {}))
    monkeypatch.setattr(Q, "jobs", lambda *args: [{"id": "fake"}])
    events = []
    previous = signal.getsignal(signal.SIGTERM)
    previous_int = signal.getsignal(signal.SIGINT)

    class Pool:
        def __init__(self, **kwargs):
            pass

        def map(self, worker, cards):
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)

        def shutdown(self, **kwargs):
            events.append("waited")
            assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN
            assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN

    monkeypatch.setattr(Q, "ThreadPoolExecutor", Pool)
    with pytest.raises(KeyboardInterrupt, match="terminated"):
        Q.execute_queue("signals", cards="cpu")
    assert events == ["waited"]
    assert signal.getsignal(signal.SIGTERM) == previous
    assert signal.getsignal(signal.SIGINT) == previous_int
    with S.scale.run_lock(S.run_path("signals")):
        pass


def test_reuse_jobs_readonly_and_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(S.common, "RESULTS_ROOT", tmp_path)
    external = tmp_path / "06_data_scale" / "old" / "train" / "state_n12_seed0"
    external.mkdir(parents=True)
    marker = external / "train_complete.json"
    marker.write_text("{}")
    audit_dir = tmp_path / "06_data_scale" / "old"
    S.write_json(audit_dir / "audit.json", {"inputs": {}, "hashes": {}})
    ref_row = {"out": str(external), "condition": "state", "seed": 0, "episodes": 12, "steps": 10}
    p = {"episodes": 12, "steps": 10, "inputs": {}, "hashes": {},
         "data_paths": {v: str(tmp_path / v) for v in S.VARIANTS},
         "baseline_reuse": {"state": {"row": ref_row, "run": "old", "train_complete_sha1": S.common.sha1_file(marker)}}}
    row = S.jobs("reuse", p)[0]
    assert row["reuse"] and row["out"] == str(external)
    assert S.evaluation_path(row) != external
    monkeypatch.setattr(S.scale, "validate_done", lambda *args: {"weights": "ema"})
    assert S.validate_done(external, row, p) == {"weights": "ema"}
    marker.write_text("{\"changed\": true}")
    with pytest.raises(ValueError, match="reused baseline changed"):
        S.validate_done(external, row, p)


def test_source_identity_includes_metric_row_validator():
    assert "preresearch/04_ood_generalization/exp/load.py" in S.code_identity()
