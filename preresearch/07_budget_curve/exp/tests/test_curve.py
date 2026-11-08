"""Preflight, equal-budget batch sampling, and paired crossing analysis."""
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from exp.audit import audit, check_scene_separation
from exp.analysis import format_frames, load_scores, plot, plot_ticks, set_budget_axis, summarize
from exp.generate_data import build, check_prefix, disjoint_scenes, frozen_hard_bench_disjoint, prepare_validation
from datasets_v3 import state_hash
from exp.make_jobs import epoch_steps, jobs
from exp.shared import EXP_NAME, LEVELS, PACKAGE, common


def test_equal_budget_jobs():
    joblist = jobs("test-run", common.DEFAULT_DATA, (0,))
    expected = {"07_03_budget_epochs": 20480, "07_03_simple_to_complex": 40960,
                "07_03_simple_to_complex_81920": 81920}.get(EXP_NAME, 4096)
    assert LEVELS[-1] == expected and len(joblist) == len(LEVELS)*2
    assert len({job["out"] for job in joblist}) == len(joblist)
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
    assert epoch_steps(20480, "state", 256, 5) == 22400
    assert epoch_steps(20480, "process", 256, 5) == 23600
    assert epoch_steps(40960, "state", 256, 5) == 44800
    assert epoch_steps(40960, "process", 256, 5) == 47200
    assert epoch_steps(81920, "state", 256, 5) == 89600
    assert epoch_steps(81920, "process", 256, 5) == 94400


def test_dense_budget_axis_labels_are_sparse_and_diagonal():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    frames = [64*n for n in (16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024,
                              1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 20480,
                              28672, 32768, 40960, 49152, 57344, 65536, 73728, 81920)]
    ticks = plot_ticks(frames, 7)
    assert 2 <= len(ticks) <= 7 and ticks[0] == frames[0] and ticks[-1] == frames[-1]
    assert all(t in frames for t in ticks)
    assert format_frames(frames[-1]).endswith("M")
    fig, ax = plt.subplots()
    ax.set_xscale("log")
    set_budget_axis(ax, frames, limit=7)
    assert len(ax.get_xticks()) == len(ticks)
    assert all(label.get_rotation() == 45 and label.get_ha() == "right" for label in ax.get_xticklabels())
    plt.close(fig)


def test_dense_budget_plots_render_with_rotated_ticks(tmp_path):
    frames = [64 * n for n in (16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512,
                                768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288,
                                16384, 20480, 28672, 32768, 40960, 49152, 57344, 65536, 73728, 81920)]
    rows = {str(f): {"total_frames_per_condition": f, "state": .2, "process": .3,
                     "diff": .1, "diff_ci": [.05, .15]} for f in frames}
    plot({split: {"levels": rows} for split in common.SPLITS}, tmp_path)
    for name in ("budget_curve.png", "budget_curve.pdf", "budget_difference.png", "budget_difference.pdf"):
        assert (tmp_path / name).stat().st_size > 1000


def test_validation_is_at_each_epoch_boundary_and_legacy_schedule_is_unchanged():
    train = import_module(f"{PACKAGE}.train")
    assert [step for step in range(1, 21) if train.validation_due(step, 20, 500, 4)] == [4, 8, 12, 16, 20]
    assert [step for step in range(1, 1201) if train.validation_due(step, 1200, 500, None)] == [500, 1000, 1200]
    assert not train.validation_due(20, 20, 0, None)
    assert train.better_validation({"cascade_acc": .3}, {"cascade_acc": .2})
    assert not train.better_validation({"cascade_acc": .2}, {"cascade_acc": .2})


def test_full_epoch_jobs_keep_original_protocol_isolated():
    rows = jobs("test-run", common.DEFAULT_DATA, (0,), epochs=5)
    assert len(rows) == 2 * len(LEVELS)
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
    if EXP_NAME in ("07_03_simple_to_complex", "07_03_simple_to_complex_81920"):
        pytest.skip("hard-transfer namespace requires its own benchmark")
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


@pytest.mark.parametrize("workers", (1, 2))
@pytest.mark.parametrize("profile", ("v3", "simple_to_complex"))
def test_train_progress_preserves_episode_data(tmp_path, capsys, workers, profile):
    import datasets_v3 as v3

    options = {"F": 32, "profile": profile}
    if profile == "simple_to_complex":
        options.update(p_mistake=0.0, scaffold=False)
    summary = v3.generate_train(tmp_path, 2, 64, 0, workers=workers, **options)
    expected = [v3.make_episode_set(0, i, **options) for i in range(2)]
    assert common.load_json(tmp_path / "train_scenes.json") == v3.D._jsonable([e["meta"] for e in expected])
    for cond in common.CONDITIONS:
        with np.load(tmp_path / f"train_{cond}.npz") as arrays:
            for key in ("fg", "bg", "cont", "phase"):
                np.testing.assert_array_equal(arrays[key], np.stack([e["episodes"][cond][key] for e in expected]))
        actions = common.load_json(tmp_path / f"train_{cond}_actions.json")
        assert actions["actions"] == [e["episodes"][cond]["actions"] for e in expected]
        assert actions["extra"] == v3.D._jsonable([
            {k: e["episodes"][cond][k] for k in ("order", "picks") if k in e["episodes"][cond]}
            for e in expected])
        assert summary["conditions"][cond]["episodes"] == 2
    with np.load(tmp_path / "train_finals.npz") as finals:
        for i, key in enumerate(("fg", "bg")):
            np.testing.assert_array_equal(finals[key], np.stack([e["final"][i] for e in expected]))
    assert common.load_json(tmp_path / "train_summary.json") == summary
    output = capsys.readouterr().out
    assert "2/2 episodes (100.0%)" in output and "rate=" in output and "ETA=" in output
    assert "assemble training arrays: done" in output
    assert all(f"compress/write train_{c}.npz: done" in output for c in common.CONDITIONS)


def test_progress_heartbeat_before_first_result_and_failure(monkeypatch, capsys):
    from threading import Event
    import datasets_v3 as v3

    heartbeat = Event()
    report = v3.GenerationProgress._report

    def observe(self, status="running"):
        report(self, status)
        if status == "running":
            heartbeat.set()

    monkeypatch.setattr(v3.GenerationProgress, "_report", observe)
    with pytest.raises(ValueError, match="worker failed"):
        with v3.GenerationProgress("waiting", 3, "episodes", interval=.01) as progress:
            assert heartbeat.wait(2)
            raise ValueError("worker failed")
    assert not progress._thread.is_alive()
    output = capsys.readouterr().out
    assert "waiting: running 0/3 episodes" in output
    assert "ETA=unknown" in output and "waiting: failed" in output


def test_parallel_progress_counts_completed_jobs_without_reordering(monkeypatch, capsys):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    import datasets_v3 as v3

    release_first = Event()
    advance = v3.GenerationProgress.advance

    def observe(self, n=1):
        advance(self, n)
        release_first.set()

    def worker(i):
        if i == 0:
            assert release_first.wait(2), "progress waited for the first job instead of the completed second job"
        return i

    monkeypatch.setattr(v3, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(v3.GenerationProgress, "advance", observe)
    assert v3._generate_results(worker, [0, 1], 2, "out of order", "episodes") == [0, 1]
    assert "2/2 episodes (100.0%)" in capsys.readouterr().out


def test_input_hash_progress_preserves_checksums(tmp_path, capsys):
    mod = import_module("exp.generate_data")
    names = ["train_summary.json", "train_scenes.json", "train_finals.npz"]
    names += [f"train_{c}{suffix}" for c in common.CONDITIONS for suffix in (".npz", "_actions.json")]
    for name in names:
        (tmp_path / name).write_text(name)
    expected = {name: common.sha1_file(tmp_path / name) for name in names}
    assert mod.input_hashes(tmp_path) == expected
    output = capsys.readouterr().out
    assert "[1/11] train_summary.json: started" in output
    assert "[11/11] train_shuffle_actions.json: done" in output
    (tmp_path / names[0]).write_text("changed")
    assert mod.input_hashes(tmp_path)[names[0]] != expected[names[0]]


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


def test_frozen_hard_bench_only_reused_if_all_new_inputs_disjoint(tmp_path):
    fg = np.zeros((2, 2), dtype=np.uint8)
    bg = np.ones((2, 2), dtype=np.uint8)
    scene = state_hash(fg, bg)
    bench = tmp_path / "bench.npz"
    np.savez_compressed(bench, ctx_fg=np.stack([fg]), ctx_bg=np.stack([bg]))
    common.dump_json({"profile": "simple_to_complex"}, bench.with_suffix(".json"))
    assert frozen_hard_bench_disjoint(bench, {"train"}, {"validation"}, "simple_to_complex")
    assert not frozen_hard_bench_disjoint(bench, {scene}, set(), "simple_to_complex")
    assert not frozen_hard_bench_disjoint(bench, set(), {scene}, "simple_to_complex")
    with pytest.raises(ValueError, match="difficulty profile"):
        frozen_hard_bench_disjoint(bench, set(), set(), "v3")
    check_scene_separation({"a"}, {"b"}, {"c"}, include_validation_bench=True)
    for t, v, b in (({"a"}, {"a"}, {"c"}), ({"a"}, {"b"}, {"a"}),
                    ({"a"}, {"b"}, {"b"})):
        with pytest.raises(ValueError, match="overlap"):
            check_scene_separation(t, v, b, include_validation_bench=True)


def test_extended_prefix_rejects_changed_condition(tmp_path):
    original, extended = tmp_path / "original", tmp_path / "extended"
    original.mkdir()
    extended.mkdir()
    common.dump_json([{"final_hash": "a"}, {"final_hash": "b"}], original / "train_scenes.json")
    common.dump_json([{"final_hash": "a"}, {"final_hash": "b"}, {"final_hash": "c"}],
                     extended / "train_scenes.json")
    for cond in common.CONDITIONS:
        for directory, n in ((original, 2), (extended, 3)):
            values = np.zeros((n, 2, 2), dtype=np.uint8)
            np.savez_compressed(directory / f"train_{cond}.npz",
                                **{key: values for key in ("fg", "bg", "cont", "phase")})
            common.dump_json({"actions": [[] for _ in range(n)], "extra": [{} for _ in range(n)]},
                             directory / f"train_{cond}_actions.json")
    assert check_prefix(original, extended) == 2
    with np.load(extended / "train_process.npz") as z:
        arrays = {key: z[key].copy() for key in z.files}
    arrays["fg"][0, 0, 0] = 1
    np.savez_compressed(extended / "train_process.npz", **arrays)
    with pytest.raises(ValueError, match="regenerated prefix"):
        check_prefix(original, extended)


def test_curve_resume_rejects_missing_checkpoint(tmp_path):
    queue = import_module(f"{PACKAGE}.run_queue")
    job = jobs("test-run", common.DEFAULT_DATA, (0,))[0]
    common.dump_json({"steps": 12000}, tmp_path / "done.json")
    with pytest.raises(FileNotFoundError):
        queue.validate_curve_resume(job, tmp_path, [])
