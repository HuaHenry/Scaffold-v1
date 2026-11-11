from importlib import import_module

import numpy as np
import pytest

from exp.analysis import existing_report, markdown, plot
from exp.load import scene_features, validate_samples
from exp.probes import (component_probe, intervention_owners, isolate_component, pack_probe, prepare_probe,
                        single_scene, stable, truth_run, verify_probe)
from exp.shared import CONDITIONS, EXP_NAME, bind_source, common, forbid_source_output, output_path, write_json, write_npz
from exp.statistics import cluster_values, difference_of_differences, estimate, paired
import datasets as D
import datasets_v3 as v3


def synthetic_bundle(n=16):
    ids = np.repeat(np.arange(4), n // 4)
    group = {}
    raw = {}
    for j, c in enumerate(CONDITIONS):
        score = .2 + j * .03 + np.arange(n) * .001
        group[c] = {"cascade_f1": score, "cell_acc_mask": score + .3, "rank_acc": score + .2}
        raw[c] = {"cascade_f1": np.stack((score, score + .01), axis=1)}
    features = [{"index": i, "n_struct": 2 + i % 2, "contains_unseen_kind": bool(i % 2),
                 "composition_seen_in_training": bool(i % 2), "held_pair_present": bool(i % 2),
                 "max_parameter": {"tower": 7 + i % 3, "bridge": 8 + i % 3}} for i in range(n)]
    bench = {"split_ids": ids, "splits": list(common.SPLITS), "cascade": np.tile([2, 8], (n, 1)),
             "n_changed": np.tile([3, 12], (n, 1)), "depth": np.tile([1, 4], (n, 1))}
    ranges = {kind: {"parameter": field, "observed_min": 4, "observed_max": high,
                     "generator_base_range": [4, high]} for kind, field, high in (("tower", "h", 7), ("bridge", "span", 8))}
    return {"group": group, "raw": raw, "bench": bench, "features": features, "ranges": ranges}


def test_output_names_and_paths_are_isolated(tmp_path):
    assert EXP_NAME == "04_ood_generalization"
    assert "04_ood_generalization" in str(output_path("unit"))
    for name in ("", "../bad", "a/b", ".", ".."):
        with pytest.raises(ValueError):
            output_path(name)
    source = tmp_path / "source"
    data = tmp_path / "data"
    audited = {"inputs": {"data": str(data), "bench": str(data / "bench.npz"), "val": str(data / "val")}}
    for path in (source, source / "nested", data / "nested", tmp_path):
        with pytest.raises(ValueError):
            forbid_source_output(path, source, audited)
    forbid_source_output(tmp_path / "results04", source, audited)


def test_portable_source_paths_resolve_without_changing_shared_helpers():
    from exp.shared import resolve_file

    path = common.RESULTS_ROOT / "03_main/example/train/state"
    assert resolve_file(common.rel(path)) == path
    assert resolve_file(str(path)) == path


def test_manifest_reuse_and_change_rejection(tmp_path):
    bind_source(tmp_path, {"models": {"state": "original"}})
    original = (tmp_path / "source.json").read_bytes()
    bind_source(tmp_path, {"models": {"state": "original"}})
    assert (tmp_path / "source.json").read_bytes() == original
    with pytest.raises(ValueError, match="changed"):
        bind_source(tmp_path, {"models": {"state": "different"}})
    assert (tmp_path / "source.json").read_bytes() == original


def test_paired_and_stratified_bootstrap_have_correct_sign_and_pairing():
    left = np.array([.2, .3, .6, .7])
    right = left - np.array([.1, .1, .3, .3])
    ref, other = np.array([1, 1, 0, 0], bool), np.array([0, 0, 1, 1], bool)
    row = difference_of_differences(left, right, ref, other, 100, 7, "test")
    assert row["delta_reference"] == pytest.approx(.1)
    assert row["delta_other"] == pytest.approx(.3)
    assert row["difference"] == pytest.approx(.2)
    np.testing.assert_allclose(row["ci"], [.2, .2])
    assert row == difference_of_differences(left, right, ref, other, 100, 7, "test")
    np.testing.assert_allclose(paired(left, right, ref, 100, 7, "paired")["ci"], [.1, .1])
    with pytest.raises(ValueError, match="different scenes"):
        difference_of_differences(left, right, ref, ref, 100, 7, "bad")


def test_empty_small_missing_and_clustered_groups():
    assert estimate([], 10)["mean"] is None
    assert estimate([.2], 10)["ci"] is None
    assert estimate([np.nan, .2], 10)["n_cases"] == 1
    values = np.array([[1., 3.], [5., 7.], [9., np.nan]])
    member = np.array([[1, 1], [0, 1], [0, 0]], bool)
    clustered = cluster_values(values, member)
    np.testing.assert_allclose(clustered[:2], [2, 7])
    assert np.isnan(clustered[2])
    assert estimate(clustered, 10)["n_cases"] == 2
    with pytest.raises(ValueError):
        cluster_values(values, np.ones(3, bool))


def test_existing_report_preserves_reference_limit_and_counts():
    bundle = synthetic_bundle()
    report = existing_report(bundle, 50, 9)
    assert report["h3_status"] == "not_tested_against_training_id"
    assert report["settings"]["reference_is_training_id"] is False
    assert report["split_scores"]["cascade_f1"]["id"]["n_cases"] == 4
    gap = report["gaps_reference_minus_other"]["cascade_f1"]["combo_ood"]["state"]["difference"]
    assert gap == pytest.approx(-.004)
    relative = report["relative_gains_other_minus_reference"]["cascade_f1"]["combo_ood"]["process-state"]
    assert relative["difference"] == pytest.approx(0)
    bins = report["difficulty"]["cascade"]["id"]
    assert bins[0]["n_cases"] == 0 and bins[0]["conditions"]["state"]["mean"] is None
    assert bins[1]["n_cases"] == 4 and bins[1]["n_interventions"] == 4
    assert bins[2]["n_cases"] == 4 and bins[2]["n_interventions"] == 4
    assert "不宣布真正 ID→OOD" in markdown(report, "source03")


def test_run_analysis_writes_only_new_outputs_and_keeps_previous_analysis(tmp_path, monkeypatch):
    import exp.analysis as mod

    bundle = synthetic_bundle()
    source = tmp_path / "source03"
    source.mkdir()
    (source / "original.json").write_text("unchanged")
    bundle.update(source=source, audited={"inputs": {}}, manifest={"source": str(source), "version": 1})
    monkeypatch.setattr(mod.common, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(mod, "plot", lambda *args: None)
    first = mod.run_analysis(bundle, "unit", 20, 9)
    original = (first / "ood_analysis.json").read_bytes()
    second = mod.run_analysis(bundle, "unit", 20, 9)
    assert first != second and (first / "ood_analysis.json").read_bytes() == original
    assert common.load_json(second / "meta.json")["exp"] == EXP_NAME
    assert common.load_json(second / "ood_analysis.json")["settings"]["training_seeds"] == [0]
    assert (source / "original.json").read_text() == "unchanged"
    assert not (source / "analysis").exists()


def test_reports_render_without_missing_case_imputation(tmp_path):
    report = existing_report(synthetic_bundle(), 20, 1)
    plot(report, tmp_path)
    for name in ("split_scores", "relative_gains", "difficulty_cascade", "difficulty_n_changed", "difficulty_depth", "observed_scales"):
        for suffix in ("png", "pdf"):
            assert (tmp_path / f"{name}.{suffix}").stat().st_size > 1000
    write_json(tmp_path / "report.json", {"missing": np.nan, "values": np.array([.2, np.nan])})
    assert common.load_json(tmp_path / "report.json") == {"missing": None, "values": [.2, None]}


def fixture_items():
    fg, bg = D.terrain(), np.zeros((D.DH, D.DW), np.uint8)
    fg[D.SURF - 4:D.SURF, 10] = v3.BRICK
    scene = {"kinds": ["tower"], "specs": [{"kind": "tower", "h": 4, "width": 1}], "x0s": [10], "mirrored": [False]}
    ivs = [v3.mk("single", (0, [{"type": "remove", "x": 10, "y": D.SURF - 1}])), v3.mk("noop", (0, []))]
    pair = [truth_run(fg, bg, iv, 17) for iv in ivs]
    assert all(r is not None for r in pair)
    return [{"fg": fg, "bg": bg, "pair": pair, "scene": scene,
             "group": {"probe": "scale", "kind": "tower", "value": 4, "within_training_parameter_support": True}}]


def fixture_bench(tmp_path):
    arrays, meta = pack_probe(fixture_items())
    write_npz(tmp_path / "bench.npz", arrays)
    write_json(tmp_path / "bench.json", meta)
    return common.load_bench(tmp_path / "bench.npz")


def test_prediction_metrics_can_be_reproduced_and_alignment_is_checked(tmp_path):
    bench = fixture_bench(tmp_path)
    evaluate = import_module("preresearch.01_pilot_validity.exp.evaluate")
    samples = evaluate.score_bench(bench, bench["fut_fg"], bench["fut_bg"])
    validate_samples(samples, bench)
    bad = {k: v.copy() for k, v in samples.items()}
    bad["cascade_f1"][0, 0] = 0
    with pytest.raises(ValueError, match="cannot be reproduced"):
        validate_samples(bad, bench)
    bad = {k: v.copy() for k, v in samples.items()}
    bad["n_cascade_true"][0, 0] += 1
    with pytest.raises(ValueError, match="not aligned"):
        validate_samples(bad, bench)


def test_scale_metadata_uses_blueprint_not_settling_depth(tmp_path):
    bench = fixture_bench(tmp_path)
    training = [{"kinds": ["tower"], "specs": [{"kind": "tower", "h": 5, "width": 1}]}]
    features, ranges = scene_features(bench, training)
    assert features[0]["max_parameter"] == {"tower": 4}
    assert features[0]["n_struct"] == 1 and not features[0]["contains_unseen_kind"]
    assert ranges["tower"]["observed_max"] == 5
    assert ranges["bridge"]["observed_max"] is None
    assert stable(bench["ctx_fg"][0], bench["ctx_bg"][0], 17)


@pytest.mark.parametrize("kind,values", (("tower", (4, 7, 16)), ("bridge", (4, 8, 11, 14))))
def test_scale_generator_respects_geometry_and_material_limit(kind, values):
    for value in values:
        scene = single_scene(kind, value, D.rng_for(9, kind, value))
        spec = scene["specs"][0]
        assert spec["h" if kind == "tower" else "span"] == value
        assert 2 <= scene["x0s"][0] <= D.DW - 2 - spec["width"]
        if kind == "bridge":
            assert spec["width"] == value
            if value > 10:
                assert spec["beam"] == v3.BRICK
        world = v3.settled_build(scene, 9)
        if world is not None:
            assert stable(world.grid, world.background, 9)


def test_target_component_and_isolation_preserve_position():
    fg, bg = D.terrain(), np.zeros((D.DH, D.DW), np.uint8)
    fg[D.SURF - 2:D.SURF, 10] = v3.BRICK
    fg[D.SURF - 2:D.SURF, 30] = v3.BRICK
    scene = {"x0s": [10, 30], "specs": [{"width": 2}, {"width": 2}]}
    action = {"type": "remove", "x": 10, "y": D.SURF - 1}
    assert intervention_owners(scene, v3.mk("single", (0, [action]))) == 0
    assert intervention_owners(scene, v3.mk("pair", (0, [action, dict(action, x=30)]))) is None
    a, b = isolate_component(fg, bg, scene, 0)
    assert np.array_equal(a[:, 10:12], fg[:, 10:12])
    assert a[D.SURF - 1, 30] == 0 and np.array_equal(b, bg)


def test_matched_component_statistics_only_use_targeted_singletons():
    from exp.probe_analysis import component_family_scores
    samples = {"cascade_f1": np.array([[.5, .6], [.2, 1.], [1., .4]])}
    family = {"combined": 0, "singles": [1, 2], "owners": [0, 1]}
    full, alone = component_family_scores(samples, [family], "cascade_f1")
    np.testing.assert_allclose(full, [.55])
    np.testing.assert_allclose(alone, [.3])
    with pytest.raises(ValueError, match="not comparable"):
        component_family_scores(samples, [family], "rank_acc")


def test_component_probe_builds_same_components_and_reports_missing_pairs():
    fg, bg = D.terrain(), np.zeros((D.DH, D.DW), np.uint8)
    fg[D.SURF - 4:D.SURF, 10] = v3.BRICK
    fg[D.SURF - 4:D.SURF, 30] = v3.BRICK
    scene = {"kinds": ["tower", "tower"], "specs": [{"kind": "tower", "h": 4, "width": 1}] * 2,
             "x0s": [10, 30], "mirrored": [False, False]}
    ivs = [v3.mk("single", (0, [{"type": "remove", "x": x, "y": D.SURF - 1}])) for x in (10, 30)]
    bench = {"ctx_fg": fg[None], "ctx_bg": bg[None], "split_ids": np.array([0]),
             "splits": ["combo_ood"], "meta": {"scenes": [scene]}, "ivs": [ivs]}
    items, coverage = component_probe(bench, {v3.state_hash(fg, bg)}, 1, 17)
    assert len(items) == 3 and coverage["accepted"] == 1
    assert coverage["families"][0]["owners"] == [0, 1]
    assert items[1]["pair"][1]["iv"]["kind"] == "noop"
    assert items[2]["pair"][0]["iv"]["kind"] == "noop"
    assert np.array_equal(items[1]["fg"][:, 10:11], fg[:, 10:11])
    blocked = {v3.state_hash(it["fg"], it["bg"]) for it in items}
    missing, report = component_probe(bench, blocked, 1, 17)
    assert not missing and report["accepted"] == 0
    assert report["rejections"]["single_overlap_or_unstable"] == 1


@pytest.mark.parametrize("condition", ("oracle_reseed", "state"))
def test_chunked_evaluation_reuses_completed_files(tmp_path, monkeypatch, condition):
    import exp.evaluate as mod
    from exp.probe_analysis import summarize_scale
    import torch

    data, model_dir = tmp_path / "data/scale", tmp_path / "source03/train/state"
    data.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    arrays, meta = pack_probe(fixture_items() * 2)
    write_npz(data / "bench.npz", arrays)
    write_json(data / "bench.json", meta)
    write_json(data / "coverage.json", [{"accepted": 2}])
    done = {"settings": {"probe": "scale"}, "status": "ready", "n_samples": 2,
            "hashes": {name: common.sha1_file(data / name) for name in ("bench.npz", "bench.json", "coverage.json")}}
    write_json(data / "complete.json", done)
    source = {"source_run": str(tmp_path / "source03"), "sources": {"state": str(model_dir)}, "models": {"state": {}}}
    if condition == "state":
        model_api = import_module("preresearch.01_pilot_validity.exp.model")
        cfg = {"dim": 16, "depth": 1, "heads": 2}
        model = model_api.build_model(cfg)
        torch.save({"cfg": cfg, "ema": model.state_dict()}, model_dir / "ckpt.pt")
        source["models"]["state"]["ckpt.pt"] = common.sha1_file(model_dir / "ckpt.pt")
        write_json(tmp_path / "source03/audit.json", {})
        monkeypatch.setattr(mod.main_shared, "validate_completed", lambda *args: None)
    write_json(tmp_path / "source.json", source)
    monkeypatch.setattr(mod, "output_path", lambda run: tmp_path)
    result = mod.evaluate_probe("unit", "scale", condition, batch=1, device="cpu")
    assert result == mod.evaluate_probe("unit", "scale", condition, batch=1, device="cpu")
    with np.load(tmp_path / "evaluate/scale" / condition / "eval_samples.npz") as z:
        assert z["cascade_f1"].shape == (2, 2)
        if condition == "oracle_reseed":
            np.testing.assert_array_equal(z["exact"], np.ones((2, 2)))
            bench = common.load_bench(data / "bench.npz")
            samples = {c: {key: z[key].copy() for key in ("cascade_f1", "cell_acc_mask", "rank_acc", "exact")} for c in (*CONDITIONS, "oracle_reseed")}
            summary = summarize_scale(bench, samples, [], 20, 9)
            assert summary["by_kind"]["tower"][0]["n_cases"] == 2
            assert not summary["by_kind"]["bridge"]


def test_complete_probe_reuses_and_changed_settings_refuse(tmp_path, monkeypatch):
    import exp.probes as mod
    bundle = {"manifest": {"source": "same"}, "training": [], "features": [],
              "audited": {"inputs": {"val": str(tmp_path / "val")}}, "bench": {}}
    (tmp_path / "val").mkdir()
    write_json(tmp_path / "val/train_scenes.json", [])
    calls = []

    def create(*args):
        calls.append(1)
        return fixture_items(), [{"accepted": 1}]

    monkeypatch.setattr(mod, "scale_probe", create)
    directory, done = prepare_probe(bundle, tmp_path / "run", "scale", 1, 17, 2)
    assert done["n_samples"] == 1 and len(calls) == 1
    assert prepare_probe(bundle, tmp_path / "run", "scale", 1, 17, 2)[1] == done
    assert len(calls) == 1
    with pytest.raises(ValueError, match="settings"):
        prepare_probe(bundle, tmp_path / "run", "scale", 2, 17, 2)
    (directory / "coverage.json").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        verify_probe(directory, done["settings"])


def test_no_usable_probe_and_partial_probe_never_silently_regenerate(tmp_path, monkeypatch):
    import exp.probes as mod
    (tmp_path / "val").mkdir()
    write_json(tmp_path / "val/train_scenes.json", [])
    bundle = {"manifest": {}, "training": [], "features": [], "audited": {"inputs": {"val": str(tmp_path / "val")}}}
    monkeypatch.setattr(mod, "scale_probe", lambda *args: ([], [{"accepted": 0}]))
    directory, done = prepare_probe(bundle, tmp_path / "run", "scale", 1, 17, 2)
    assert done["status"] == "no_unique_valid_cases" and not (directory / "bench.npz").exists()
    assert prepare_probe(bundle, tmp_path / "run", "scale", 1, 17, 2)[1] == done
    partial = tmp_path / "other/data/scale"
    partial.mkdir(parents=True)
    with pytest.raises(ValueError, match="partial probe"):
        prepare_probe(bundle, tmp_path / "other", "scale", 1, 17, 2)


def test_eval_cache_reuse_and_tampering(tmp_path):
    from exp.evaluate import load_cached
    write_json(tmp_path / "eval.json", {"score": .2})
    write_npz(tmp_path / "eval_samples.npz", {"score": np.array([.2])})
    identity = {"condition": "state", "source_sha": "same"}
    write_json(tmp_path / "complete.json", {"identity": identity, "hashes": {n: common.sha1_file(tmp_path / n) for n in ("eval.json", "eval_samples.npz")}})
    load_cached(tmp_path, identity)
    with pytest.raises(ValueError, match="identity"):
        load_cached(tmp_path, {"condition": "process"})
    (tmp_path / "eval.json").write_text("different")
    with pytest.raises(ValueError, match="modified"):
        load_cached(tmp_path, identity)


def test_oracle_threshold_and_missing_reference_not_a_pass(tmp_path, monkeypatch):
    import exp.evaluate as mod
    data = tmp_path / "data"
    (tmp_path / "evaluate/scale/oracle_reseed").mkdir(parents=True)
    data.mkdir()
    write_json(data / "bench.json", {"groups": [{"kind": "tower", "within_training_parameter_support": True}]})
    monkeypatch.setattr(mod, "output_path", lambda run: tmp_path)
    monkeypatch.setattr(mod, "probe_identity", lambda *args: (data, {}))
    write_npz(tmp_path / "evaluate/scale/oracle_reseed/eval_samples.npz", {"exact": np.array([[1., 1.]])})
    report = mod.check_oracle("unit", "scale")
    assert report["groups"][0]["status"] == "pass" and report["groups"][1]["status"] == "not_available"
    write_npz(tmp_path / "evaluate/scale/oracle_reseed/eval_samples.npz", {"exact": np.array([[0., 0.]])})
    with pytest.raises(ValueError, match="oracle validation failed"):
        mod.check_oracle("unit", "scale")


def test_main_dry_run_and_default_mode_do_not_generate_probes(tmp_path, monkeypatch, capsys):
    import exp.main as mod
    bundle = {"source": tmp_path / "source03", "audited": {"inputs": {}}, "manifest": {"models": "fixed"}}
    monkeypatch.setattr(mod, "load_source", lambda value: bundle)
    monkeypatch.setattr(mod, "output_path", lambda run: tmp_path / "results")
    monkeypatch.setattr(mod, "forbid_source_output", lambda *args: None)

    def forbidden(*args, **kwargs):
        raise AssertionError("must not write in dry-run or create default-mode probes")

    monkeypatch.setattr(mod, "start_run", forbidden)
    monkeypatch.setattr(mod, "bind_source", forbidden)
    monkeypatch.setattr(mod, "run_analysis", forbidden)
    monkeypatch.setattr(mod, "prepare_probe", forbidden)
    mod.main(["--dry-run", "--run", "unit", "--with-probes"])
    assert not (tmp_path / "results").exists()
    assert "no files written" in capsys.readouterr().out
    monkeypatch.setattr(mod, "start_run", lambda label: "unit")
    monkeypatch.setattr(mod.common, "start_run", lambda *args: None)
    monkeypatch.setattr(mod, "bind_source", lambda *args: None)
    calls = []
    monkeypatch.setattr(mod, "run_analysis", lambda *args: calls.append(1) or tmp_path)
    mod.main(["--run", "unit"])
    assert calls == [1]


def test_invalid_cli_counts_and_duplicate_devices_fail_before_source_read(monkeypatch):
    import exp.main as mod
    monkeypatch.setattr(mod, "load_source", lambda *args: pytest.fail("must validate arguments before loading data"))
    for argv in (["--boot", "0"], ["--per-setting", "0"], ["--seed", "-1"], ["--gpus", "0,0"]):
        with pytest.raises(SystemExit):
            mod.main(argv)
