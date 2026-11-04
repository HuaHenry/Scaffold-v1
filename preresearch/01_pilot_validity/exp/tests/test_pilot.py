import json
import sys
from pathlib import Path

import numpy as np
import pytest

from exp import common, metrics, aggregate as agg, make_jobs
from exp.precheck import decode_token

DATA = common.DEFAULT_DATA
needs_data = pytest.mark.skipif(not (DATA / "bench.npz").exists(), reason="v3 data not generated")


# ---------------------------------------------------------------- encoding
def test_encode_remove_and_interact_ignore_material():
    a = common.encode_actions([{"type": "remove", "x": 3, "y": 4, "material": 10, "layer": "front"}])
    b = common.encode_actions([{"type": "remove", "x": 3, "y": 4, "material": 2, "layer": "front"}])
    assert a[4, 3] == b[4, 3] != 0 and (a != 0).sum() == 1


def test_encode_roundtrip_noop_and_last_wins():
    acts = [{"type": "place", "x": 1, "y": 1, "material": 13, "layer": "front"},
            {"type": "place", "x": 1, "y": 1, "material": 16, "layer": "back"},
            {"type": "noop", "x": 9, "y": 9, "material": 3, "layer": "front"},
            {"type": "interact", "x": 5, "y": 6, "material": 3, "layer": "front"}]
    m = common.encode_actions(acts)
    assert (m != 0).sum() == 2
    assert decode_token(int(m[1, 1])) == {"type": "place", "material": 16, "layer": "back"}
    assert decode_token(int(m[6, 5]))["type"] == "interact" and m.max() < common.N_ACT


def test_compound_intervention_maps():
    iv = {"kind": "seq2", "steps": [{"t": 0, "actions": [{"type": "remove", "x": 2, "y": 3, "material": 2, "layer": "front"}]},
                                    {"t": 2, "actions": [{"type": "place", "x": 5, "y": 5, "material": 2, "layer": "front"}]}]}
    m = common.iv_action_maps(iv)
    assert m.shape == (3, common.H, common.W) and (m[1] == 0).all() and m[0, 3, 2] != 0 and m[2, 5, 5] != 0


# ---------------------------------------------------------------- protocol / data
@needs_data
def test_pair_counts_match_plan():
    s = common.load_summary(DATA)
    if s["conditions"]["state"]["episodes"] != 256:
        pytest.skip("not the full dataset")
    n = {c: (len(common.make_pairs(DATA, c, "video")), len(common.make_pairs(DATA, c, "causal"))) for c in common.CONDITIONS}
    assert n == {"state": (14336, 14336), "partial": (14336, 14336), "process": (15104, 15104), "shuffle": (15104, 7168)}


@needs_data
def test_prefix_subsets_are_prefixes():
    full = common.load_condition(DATA, "state", None, with_actions=False)
    small = common.load_condition(DATA, "state", 64, with_actions=False)
    np.testing.assert_array_equal(full["fg"][:64], small["fg"])
    p_full, p64 = common.make_pairs(DATA, "state"), common.make_pairs(DATA, "state", episodes=64)
    np.testing.assert_array_equal(p_full[:len(p64)], p64)
    assert p64[:, 0].max() == 63 and len(p64) == 64 * 56
    assert len(common.make_pairs(DATA, "state", episodes=16)) == 16 * 56


@needs_data
def test_video_pairs_never_cross_a_clip_boundary():
    for c in ("state", "process", "shuffle"):
        p = common.make_pairs(DATA, c)
        t = p[:, 1]
        starts = [0, 8, 16, 24, 32, 40, 48, 56] if c == "state" else [0, 32, 40, 48, 56]   # first frame of every clip
        assert not np.isin(t, starts).any()
        assert t.min() >= 1


@needs_data
def test_formation_share_is_31_of_59():
    d = np.load(DATA / "train_process.npz")
    p = common.make_pairs(DATA, "process")
    share = float((d["phase"][p[:, 0], p[:, 1]] == 0).mean())
    assert abs(share - 31 / 59) < 1e-9


@needs_data
def test_bench_loader_and_intervention_cells():
    b = common.load_bench()
    assert b["ctx_fg"].shape[1:] == (28, 56) and b["A"].shape[:2] == b["labels"].shape
    assert b["A"].any(axis=(2, 3)).all()
    assert (b["t_last"] >= 0).all() and set(b["splits"]) == set(common.SPLITS)


# ---------------------------------------------------------------- metrics / baselines
def test_rank_acc_constant_and_random_predictions_near_half():
    assert metrics.rank_acc(0.2, 0.2) == 0.5 and metrics.rank_acc(0.3, 0.1) == 1.0 and metrics.rank_acc(0.1, 0.3) == 0.0
    rng = np.random.default_rng(0)
    r = np.mean([metrics.rank_acc(*rng.random(2)) for _ in range(4000)])
    assert abs(r - 0.5) < 0.03


@needs_data
def test_baselines_reproduce_dataset_audit():
    from exp import baselines
    from exp.evaluate import score_bench, summarize
    b = common.select_bench(common.load_bench(), "id", 16)
    audit = json.loads((DATA / "audit.json").read_text())["bench"]["splits"]["id"]["baselines"]
    for name in ("copy_last", "apply_edit"):
        tf, tb = baselines.predict_bench(name, b)
        s = score_bench(b, tf, tb, with_violations=False)
        rows = [i for i in range(len(s["split_ids"]))]
        for key in ("cell_acc_mask", "structure_iou"):
            ours = float(np.nanmean(s[key][:, 0]))
            assert abs(ours - audit[name][key]) < 0.2        # 16 samples only; exact equality is checked in the README run
    tf, tb = baselines.predict_bench("oracle_reseed", b)
    s = score_bench(b, tf, tb, with_violations=True)
    assert np.nanmean(s["cascade_f1"]) > 0.95 and np.nanmean(s["any_violation"]) == 0


def test_fixed_point_step():
    from exp.rollout import fixed_point_step
    N = 1
    ctx = np.zeros((N, 28, 56), np.uint8)
    tr = np.zeros((N, 2, 32, 28, 56), np.uint8)
    tr[0, 0, :3, 0, 0] = [1, 2, 3]
    tr[0, 0, 3:, 0, 0] = 3
    fp = fixed_point_step(tr, tr * 0, ctx, ctx, np.zeros((N, 2), int))
    assert fp[0, 0] == 3 and fp[0, 1] == 1


# ---------------------------------------------------------------- model
def test_model_shapes_and_size():
    import torch
    from exp.model import build_model, n_params
    m = build_model({"dim": 64, "depth": 1, "heads": 2, "emb": 16})
    fg = torch.randint(0, 21, (2, 28, 56)); act = torch.randint(0, 127, (2, 28, 56))
    lf, lb = m(fg, fg, act)
    assert lf.shape == lb.shape == (2, 21, 28, 56)
    assert m.step(fg, fg, act)[0].shape == (2, 28, 56)
    assert 13e6 < n_params(build_model()) < 16e6


# ---------------------------------------------------------------- statistics
def test_aa_partitions_ten_unique():
    parts = agg.aa_partitions([0, 1, 2, 3, 4, 5], 3)
    assert len(parts) == 10
    keys = {frozenset([frozenset(a), frozenset(b)]) for a, b in parts}
    assert len(keys) == 10 and all(set(a) | set(b) == set(range(6)) and not set(a) & set(b) for a, b in parts)


def test_paired_bootstrap_null_and_shift():
    rng = np.random.default_rng(1)
    x = rng.random(300)
    assert not agg.paired_bootstrap(x, x.copy(), 2000)["excludes_0"]
    assert agg.paired_bootstrap(x + 0.1, x, 2000)["excludes_0"]
    assert agg.paired_bootstrap(x + 0.1, x, 2000) == agg.paired_bootstrap(x + 0.1, x, 2000)     # fixed rng


def test_seed_level_bootstrap_is_wider_than_paired_when_seeds_differ():
    rng = np.random.default_rng(2)
    base = rng.random(200)
    A = base[None] + rng.normal(0, .05, (3, 1)) + rng.normal(0, .01, (3, 200))
    B = base[None] + rng.normal(0, .05, (3, 1)) + rng.normal(0, .01, (3, 200))
    assert agg.seed_level_bootstrap(A, B, 500)["se"] > agg.paired_bootstrap(A.mean(0), B.mean(0), 500)["se"]


def test_holm():
    assert agg.holm([0.001, 0.04, 0.03]) == [True, False, False]
    assert agg.holm([0.001, 0.01, 0.02]) == [True, True, True]


# ---------------------------------------------------------------- result folders
def _fake_dir(root, exp, kind, rid, settings, ts, done=("done.json",)):
    d = root / exp / common.result_name(kind, rid, "x", ts)
    d.mkdir(parents=True)
    common.write_meta(d, kind, rid, settings)
    for f in done:
        (d / f).write_text("{}")
    return d


def test_result_name_starts_with_timestamp():
    n = common.result_name("train", "state_s0", "ep256_video_steps6000", "20261104-173012")
    assert n == "20261104-173012_train_state_s0_ep256_video_steps6000"
    assert common.result_name("train", "a b/c", "")[:8].isdigit() and "/" not in common.result_name("train", "a b/c")
    assert sorted([common.result_name("t", "x", "", "20261105-000000"), common.result_name("t", "x", "", "20261104-235959")])[0] \
        .startswith("20261104")


def test_results_root_default_is_experiments_preresearch():
    assert common.RESULTS_ROOT.parts[-2:] == ("experiments", "preresearch")
    assert common.EXP_NAME == "01_pilot_validity"


def test_find_latest_picks_newest_finished_with_matching_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "RESULTS_ROOT", tmp_path)
    old = _fake_dir(tmp_path, "01_pilot_validity", "train", "state_s0", {"steps": 6000}, "20261104-100000")
    new = _fake_dir(tmp_path, "01_pilot_validity", "train", "state_s0", {"steps": 6000}, "20261104-120000")
    _fake_dir(tmp_path, "01_pilot_validity", "train", "state_s0", {"steps": 6000}, "20261104-130000", done=())   # unfinished
    _fake_dir(tmp_path, "01_pilot_validity", "train", "state_s0", {"steps": 30}, "20261104-140000")             # smoke test
    _fake_dir(tmp_path, "03_main", "train", "state_s0", {"steps": 6000}, "20261104-110000")                     # other experiment
    assert common.find_latest("state_s0", "train", "01_pilot_validity", "done.json", {"steps": 6000}) == new
    assert common.find_results("state_s0", "train", "01_pilot_validity", "done.json", {"steps": 6000}) == [old, new]
    assert common.find_latest("state_s0", "train", None, "done.json", {"steps": 6000}).parent.name == "01_pilot_validity"
    assert common.find_latest("nope") is None
    assert common.resolve_run("state_s0", require="done.json", steps=6000) == new   # by id (+ steps filter)
    assert common.resolve_run("state_s0", require="done.json").name.startswith("20261104-140000")
    assert common.resolve_run(str(old)) == old                                   # by absolute path


def test_new_result_dir_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "RESULTS_ROOT", tmp_path)
    a = common.new_result_dir("e", "k", "i", "s", ts="20261104-000000")
    b = common.new_result_dir("e", "k", "i", "s", ts="20261104-000000")
    assert a != b and a.parent == b.parent == tmp_path / "e" and a.name.startswith("20261104-000000_k_i")


# ---------------------------------------------------------------- jobs
def test_pilot_jobs():
    ctx = make_jobs.Ctx(6000, None, "20261104-173012")
    jobs = make_jobs.pilot_validity(ctx)
    assert len(jobs) == 12 and len({j["name"] for j in jobs}) == 12 and len({j["out"] for j in jobs}) == 12
    for j in jobs:                                                  # <experiment>/<timestamp>_train_<id>_<settings>
        exp, name = j["out"].split("/")
        assert exp == "01_pilot_validity" and name.startswith("20261104-173012_train_" + j["name"] + "_ep")
        assert "steps6000" in name and j["cmd"][j["cmd"].index("--id") + 1] == j["name"]
        assert j["cmd"][j["cmd"].index("--out") + 1] == j["out"]
    n64 = [j for j in jobs if "_n64_" in j["name"]]
    assert len(n64) == 3 and all(j["cmd"][j["cmd"].index("--episodes") + 1] == "64" and "_ep64_" in j["out"] for j in n64)
    assert all("_ep256_" in j["out"] for j in jobs if "_n" not in j["name"])
    t = make_jobs.tune(ctx)
    assert len(t) == 18 and all(j["out"].startswith("00_common/") for j in t)


def test_job_settings_match_train_name():
    ctx = make_jobs.Ctx(6000, None, "20261104-173012")
    j = make_jobs.pilot_validity(ctx)[0]
    fz = common.frozen_defaults()
    want = common.train_settings(ctx.n_full, "video", 6000, 256, fz["lr"], fz["change_weight"], fz["depth"], fz["unroll"])
    assert j["out"].endswith(want)


# ---------------------------------------------------------------- analysis end-to-end on synthetic evaluations
def _fake(d: Path, mean: float, rng, N=96):
    split = np.repeat(np.arange(4), N // 4)
    x = {"split_ids": split, "rank_valid": np.ones(N, bool)}
    for k in ("cascade_f1", "change_f1"):
        x[k] = np.clip(mean + rng.normal(0, .2, (N, 2)), 0, 1)
    x["rank_acc"] = np.clip(0.5 + (mean - .1) / 2 + rng.normal(0, .3, N), 0, 1)
    x["any_violation"] = np.zeros((N, 2))
    np.savez(d / "eval_samples.npz", **x)


def test_pilot_analysis_runs_on_synthetic_runs(tmp_path, monkeypatch, capsys):
    from exp.analysis import pilot_validity as pv
    monkeypatch.setattr(common, "RESULTS_ROOT", tmp_path)
    rng = np.random.default_rng(0)
    E = "01_pilot_validity"
    label = common.bench_label(common.DEFAULT_BENCH)
    for i, (name, m) in enumerate((("copy_last", 0.0), ("apply_edit", 0.05), ("random_physics", 0.03), ("oracle_reseed", 0.97))):
        d = _fake_dir(tmp_path, E, "baseline", name, {"bench": label, "splits": "all", "limit": None}, f"20261104-10000{i}", ("eval.json",))
        _fake(d, m, rng)
    # a partial (--limit) baseline evaluation that is newer must be ignored
    _fake(_fake_dir(tmp_path, E, "baseline", "copy_last", {"bench": label, "splits": "all", "limit": 8}, "20261104-110000", ("eval.json",)), 0.9, rng)
    for s in range(6):
        _fake(_fake_dir(tmp_path, E, "train", f"state_s{s}", {"steps": 6000}, "20261104-120000", ("done.json",)), 0.5, rng)
    _fake(_fake_dir(tmp_path, E, "train", "state_s0", {"steps": 30}, "20261104-130000"), 0.99, rng)   # smoke test: ignored
    for n, m in ((64, 0.35), (16, 0.2)):
        for s in range(3):
            _fake(_fake_dir(tmp_path, E, "train", f"state_n{n}_s{s}", {"steps": 6000}, "20261104-120000"), m, rng)
    R = pv.analyze(np.repeat(np.arange(4), 24), boot=300)
    assert not R["missing"] and R["C_aa"]["n_partitions"] == 10
    assert R["checks"]["V1"]["pass"] and R["checks"]["V2"]["pass"]
    assert R["D_noise_mde"]["id"]["cascade_f1"]["mde"] > 0 and "MDE" in R["recommendation"]
    assert R["inputs"]["state_s0"].startswith("20261104-120000_") and R["inputs"]["copy_last"].startswith("20261104-100000_")
    assert "Appendix table A1" in pv.table(R)
    pv.main(["--n-boot", "100"])                                      # writes a new timestamped analysis folder
    out = common.find_latest("pilot_validity", "analysis")
    assert out is not None and (out / "pilot_validity.json").exists() and (out / "pilot_validity.md").exists()
    assert out.name[:8].isdigit() and "_analysis_pilot_validity_steps6000" in out.name


def test_freeze_picks_best_tuning_run(tmp_path, monkeypatch):
    import json
    from exp.analysis import freeze
    monkeypatch.setattr(common, "RESULTS_ROOT", tmp_path)
    for i, (lr, acc) in enumerate(((1e-4, .2), (3e-4, .6), (1e-3, .4))):
        d = _fake_dir(tmp_path, "00_common", "train", f"tune_state_lr{lr:g}_cw8_d8",
                      {"steps": 6000, "lr": lr, "change_weight": 8, "depth": 8}, f"20261104-10000{i}")
        (d / "train_log.jsonl").write_text(json.dumps({"step": 6000, "type": "val", "val_cascade_acc": acc, "val_chg_acc": acc}) + "\n")
    out = tmp_path / "frozen.json"
    freeze.main(["--out", str(out)])
    assert common.load_json(out)["lr"] == 3e-4
    rec = common.find_latest("tune", "freeze")
    assert rec is not None and "_freeze_tune_" in rec.name and (rec / "frozen.json").exists()
