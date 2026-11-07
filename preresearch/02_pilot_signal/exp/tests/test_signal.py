"""Contract tests for the experiment-02 metrics and routing."""
from __future__ import annotations

import numpy as np

from exp.analysis.coverage import changed_keys, summarize, window_key
from exp.analysis.formation_probe import classify
from exp.analysis.pilot_decision import decision, signals


def test_keys_include_padding_actions_and_both_next_layers():
    fg = np.zeros((3, 3), dtype=np.uint8)
    bg = fg.copy()
    act = fg.copy()
    future = fg.copy()
    future[0, 0] = 2
    key = window_key(fg, bg, act, future, bg, 0, 0, 1)
    assert key == window_key(fg, bg, act, future, bg, 0, 0, 1)
    assert key != window_key(fg, bg, act, future, future, 0, 0, 1)
    assert key != window_key(fg, bg, act, future, bg, 1, 1, 1)
    act[0, 0] = 1
    assert key != window_key(fg, bg, act, future, bg, 0, 0, 1)
    act[0, 0] = 0
    rows = list(changed_keys(fg, bg, act, future, bg))
    assert len(rows) == 1 and rows[0][3][0] == key
    assert rows[0][3][1] == window_key(fg, bg, act, future, bg, 0, 0, 2)


def test_formation_classes_disjoint_and_exhaustive():
    action = np.zeros((2, 2), dtype=np.uint8)
    action[0, 0] = 1
    cases = []
    for flag in range(16):
        changed = np.array([(flag >> i) & 1 for i in range(4)], dtype=bool).reshape(2, 2)
        category = classify(0, changed, action)
        assert category in ("formation_placement", "formation_physics")
        assert (category == "formation_physics") == bool(np.any(changed & ~action.astype(bool)))
        cases.append(category)
    assert set(cases) == {"formation_placement", "formation_physics"}
    assert classify(1, np.zeros((2, 2), dtype=bool), action) == "operation"


def test_coverage_counts_occurrences_not_just_distinct_keys():
    rows = {r: {} for r in (1, 2)}
    rows[1][("id", "first", "cascade")] = [b"a", b"a", b"b"]
    train = {r: {c: set() for c in ("state", "process", "partial", "shuffle")} for r in (1, 2)}
    train[1]["process"] = {b"a"}
    phases = {r: {"formation": set(), "operation": set()} for r in (1, 2)}
    phases[1]["formation"] = {b"a"}
    summary = summarize(rows, train, phases)["1"]
    assert summary["cascade_all"]["coverage"]["process"] == 2/3
    assert summary["formation_only_bench_cascade_fraction"] == 2/3
    assert summary["by_split"]["id"]["later"]["cascade"]["coverage"]["state"] is None


def test_decision_precedence_and_uncertain_signal():
    comparisons = {
        "process-state": {"change_acc": {"n": 10, "diff": .2, "lo": .1}},
        "process-shuffle": {"change_acc": {"n": 10, "diff": .1, "lo": .02}},
    }
    probe = {"comparisons": {
        "formation_placement": comparisons, "formation_physics": comparisons,
        "operation": {"process-state": {"change_acc": {"n": 10, "diff": .01}}},
    }}
    coverage = {"by_radius": {"1": {"cascade_all": {"delta_vs_state": {"process": .03}}}}}
    bench = {"by_split": {sp: {"comparisons": {"process-state": {"diff": .02, "lo": -.01}}}
                          for sp in ("id", "combo_ood", "scale_ood", "multi_ood")}}
    pilot01 = {"D_noise_mde": {sp: {"cascade_f1": {"mde": .04}}
                                for sp in bench["by_split"]}}
    threshold = {"G1_formation_gain": .15, "G1_operation_abs_max": .03,
                 "G2_shuffle_gain": .05, "C_cascade_coverage_gain": .02}
    result = signals(coverage, probe, bench, pilot01, threshold)
    assert result["G1"] and result["G2"] and result["S"] == "S0" and result["C"] == "C+"
    bench["by_split"]["id"]["comparisons"]["process-state"] = {"diff": .05, "lo": -.01}
    assert signals(coverage, probe, bench, pilot01, threshold)["S"] == "S?"
    bench["by_split"]["combo_ood"]["comparisons"]["process-state"] = {"diff": -.05, "lo": -.08}
    assert signals(coverage, probe, bench, pilot01, threshold)["S"] == "S-"
    result["G1"] = False
    assert "暂停" in decision(result)
