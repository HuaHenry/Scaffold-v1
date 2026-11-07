"""Pre-registered G1/G2/S/C decisions from Q1/Q2/Q3 and experiment 01's split MDE.

python -m exp.analysis.pilot_decision --run <02-run> --pilot01 <01-pilot_validity.json>
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..shared import EXP_NAME, common

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "pilot_decision.json"


def signals(coverage, probe, bench, pilot01, thresholds):
    mde = {sp: pilot01["D_noise_mde"][sp]["cascade_f1"]["mde"] for sp in common.SPLITS}
    g1_by_class, g2_by_class = {}, {}
    for cl in ("formation_placement", "formation_physics"):
        p = probe["comparisons"][cl]
        ps = p["process-state"]["change_acc"]
        ph = p["process-shuffle"]["change_acc"]
        g1_by_class[cl] = ps["n"] > 0 and ps["diff"] >= thresholds["G1_formation_gain"] and ps["lo"] > 0
        g2_by_class[cl] = ph["n"] > 0 and ph["diff"] >= thresholds["G2_shuffle_gain"] and ph["lo"] > 0
    op = probe["comparisons"]["operation"]["process-state"]["change_acc"]
    g1 = bool(any(g1_by_class.values()) and op["n"] > 0 and abs(op["diff"]) <= thresholds["G1_operation_abs_max"])
    g2 = bool(any(g2_by_class.values()))

    positives, negatives, unresolved = [], [], []
    for sp in common.SPLITS:
        row = bench["by_split"][sp]["comparisons"]["process-state"]
        diff = row["diff"]
        if diff >= mde[sp]:
            (positives if row["lo"] > 0 else unresolved).append(sp)
        elif -diff >= mde[sp]:
            negatives.append(sp)
    if negatives:
        s = "S-" if not positives else "S-mixed"
    elif positives:
        s = "S+"
    elif len(unresolved) == 0:
        s = "S0"
    else:
        s = "S?"
    cascade = coverage["by_radius"]["1"]["cascade_all"]
    delta = cascade["delta_vs_state"]["process"] if cascade else None
    c = "C+" if delta is not None and delta >= thresholds["C_cascade_coverage_gain"] else "C0"
    return {"G1": g1, "G1_by_class": g1_by_class, "G1_operation_diff": op["diff"],
            "G2": g2, "G2_by_class": g2_by_class, "S": s, "S_positive": positives,
            "S_negative": negatives, "S_unresolved": unresolved, "C": c,
            "C_cascade_gain": delta, "mde_by_split": mde}


def decision(flags):
    if not flags["G1"]:
        return ("暂停后续主比较；先核查形成帧的动作—帧对齐、多动作编码、训练步数。"
                "若形成缺口确实不存在，记录负结果并考虑更难的形成过程。")
    if not flags["G2"]:
        return ("先检查 shuffle 的 extra[e]['order'] 和动作对齐；修复后重训 shuffle_s0、partial_s0。"
                "在此之前不解释 shuffle 对照。")
    if flags["S"] == "S+":
        return "可推进后续多种子验证；单种子信号只是优先级信息，不作为效应结论。"
    if flags["S"] == "S0":
        suffix = "；覆盖增量亦不足（C0）" if flags["C"] == "C0" else ""
        return "继续，但形成侧缺口与拆除级联读数应分开叙述，优先补充形成侧评测" + suffix + "。"
    if flags["S"] in ("S-", "S-mixed"):
        return "先检验操作数据量是否混杂（预算对等），再开展后续多种子比较。"
    return "bench 差值超过 MDE 但样本 CI 仍含零；暂不能归类 S+/S0/S−，先补充独立确认读数。"


def markdown(result):
    f = result["signals"]
    lines = ["# Pilot decision — experiment 02", "", "Single seed: paired sample CIs exclude training-seed variance.",
             "", "| signal | value |", "|---|---|"]
    for key in ("G1", "G2", "S", "C", "C_cascade_gain", "G1_operation_diff"):
        lines.append(f"| {key} | {f[key]} |")
    lines += ["", "## FormBench signal versus experiment 01 MDE", "",
              "| split | MDE | process-state | 95% CI |", "|---|---:|---:|---:|"]
    for sp in common.SPLITS:
        row = result["bench"]["by_split"][sp]["comparisons"]["process-state"]
        lines.append(f"| {sp} | {f['mde_by_split'][sp]:.4f} | {row['diff']:+.4f} | "
                     f"[{row['lo']:+.4f}, {row['hi']:+.4f}] |")
    lines += ["", "## Next action", "", result["decision"], "",
              "G1 formation classes: " + str(f["G1_by_class"]),
              "G2 formation classes: " + str(f["G2_by_class"]),
              "", "Inputs: " + str(result["inputs"]), ""]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="02 run containing coverage and probe results")
    ap.add_argument("--pilot01", type=Path, required=True, help="01 pilot_validity.json with per-split MDE")
    ap.add_argument("--coverage", type=Path, default=None, help="exact Q1 result folder (else newest in --run)")
    ap.add_argument("--probe", type=Path, default=None, help="exact Q2/Q3 result folder (else newest in --run)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    cv = common.resolve(a.coverage) if a.coverage else common.find_latest("pilot_coverage", "analysis", EXP_NAME,
                                                                         "pilot_coverage.json", run=a.run)
    pr = common.resolve(a.probe) if a.probe else common.find_latest("pilot_probe", "analysis", EXP_NAME,
                                                                    "pilot_signal.json", run=a.run)
    if cv is None or pr is None:
        raise FileNotFoundError("Q1 coverage and Q2/Q3 probe results are required in this run")
    pilot01 = common.resolve(a.pilot01)
    cover = common.load_json(cv / "pilot_coverage.json")
    probe = common.load_json(pr / "pilot_probe.json")
    signal = common.load_json(pr / "pilot_signal.json")
    one = common.load_json(pilot01)
    settings = common.load_json(CONFIG)
    if one["checks"]["V2"]["pass"] is not True:
        raise ValueError("experiment 01 V2 must pass")
    a1 = one["A_ordering"]["id"]["cascade_f1"]
    gain = a1["state"]["mean"] - a1["apply_edit"]["mean"]
    if gain < .12 or a1["state"]["mean"] >= a1["oracle_reseed"]["mean"] or a1["random_physics"]["mean"] >= a1["apply_edit"]["mean"] + .05:
        raise ValueError("experiment 01 V1 must pass under the current plan (gain >= .12)")
    if cover["inputs"]["bench"] != signal["inputs"]["bench"]:
        raise ValueError("coverage and probe must refer to the same bench")
    if cover["inputs"]["data"] != signal["inputs"]["data"]:
        raise ValueError("coverage and probe must refer to the same training data")
    flags = signals(cover, probe, signal, one, settings)
    inputs = {"coverage": common.rel(cv), "probe": common.rel(pr), "pilot01": common.rel(pilot01),
              "thresholds": common.rel(CONFIG)}
    result = {"signals": flags, "decision": decision(flags), "bench": signal,
              "thresholds": settings, "inputs": inputs,
              "pilot01_V1_gain": gain, "pilot01_V2_pass": True}
    out = common.resolve_out(a.out) if a.out else common.new_result_dir(EXP_NAME, "analysis", "pilot_decision",
                                                                          "cascade_f1", run=a.run)
    out.mkdir(parents=True, exist_ok=True)
    common.write_meta(out, "analysis", "pilot_decision", {"metric": "cascade_f1"}, inputs=inputs)
    common.dump_json(result, out / "pilot_decision.json")
    (out / "pilot_decision.md").write_text(markdown(result), encoding="utf-8")
    print(markdown(result))


if __name__ == "__main__":
    main()
