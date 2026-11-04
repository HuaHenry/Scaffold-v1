"""Job lists for the queue (plan 00 sections 11, plan 01 section 4).

python -m exp.make_jobs pilot_validity     # 12 training tasks of experiment 01
python -m exp.make_jobs pilot_compose      # 2 short runs for check E (formation share per batch)
python -m exp.make_jobs tune               # 18 one-off tuning runs (plan 00 section 13, item 7)

Every job gets its result folder NOW: <results root>/<experiment>/<timestamp>_train_<id>_<settings>/ (the timestamp is
the moment the job list is created, shared by the whole batch). The job file is named after the experiment and the
number of steps (exp/jobs/<experiment>_steps<N>.json). If it already exists it is reused as is, so re-running the queue
resumes the same folders; pass --new to create fresh folders with a new timestamp.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from . import common
from .common import ROOT


class Ctx:
    def __init__(self, steps, data, ts, batch=256):
        self.steps, self.data, self.ts, self.batch = steps, data, ts, batch
        self.dir = Path(data) if data else common.DEFAULT_DATA
        self.fz = common.frozen_defaults()
        try:
            self.n_full = common.load_summary(self.dir)["conditions"]["state"]["episodes"]
        except Exception:
            self.n_full = 256

    def job(self, exp, rid, condition, seed, steps=None, episodes=None, tag="", lr=None, cw=None, depth=None):
        steps = steps or self.steps
        lr = self.fz["lr"] if lr is None else lr
        cw = self.fz["change_weight"] if cw is None else cw
        depth = self.fz["depth"] if depth is None else depth
        sett = common.train_settings(episodes or self.n_full, "video", steps, self.batch, lr, cw, depth,
                                     self.fz["unroll"])
        out = f"{exp}/{common.result_name('train', rid, sett, self.ts)}"
        cmd = ["python", "-m", "exp.train", "--condition", condition, "--seed", str(seed), "--protocol", "video",
               "--steps", str(steps), "--batch", str(self.batch), "--id", rid, "--exp", exp, "--out", out,
               "--lr", str(lr), "--change-weight", str(cw), "--depth", str(depth), "--unroll", str(self.fz["unroll"])]
        if self.data:
            cmd += ["--data", common.rel(self.data)]
        if episodes:
            cmd += ["--episodes", str(episodes)]
        if tag:
            cmd += ["--tag", tag]
        return {"name": rid, "out": out, "cmd": cmd}


def pilot_validity(c: Ctx):
    E = common.EXP_NAME
    jobs = [c.job(E, f"state_s{s}", "state", s) for s in range(6)]        # s0-s2 are shared with experiment 03
    for n in (64, 16):
        jobs += [c.job(E, f"state_n{n}_s{s}", "state", s, episodes=n, tag=f"n{n}") for s in (0, 1, 2)]
    return jobs


def pilot_compose(c: Ctx):
    out = []
    for cond in ("process", "shuffle"):
        j = c.job(common.EXP_NAME, f"compose_{cond}", cond, 0, steps=min(c.steps, 300))
        j["cmd"] += ["--log-batch-composition", "--val-every", "0"]
        out.append(j)
    return out


def tune(c: Ctx):
    jobs = []
    for lr, w, depth in itertools.product((1e-4, 3e-4, 1e-3), (4, 8, 16), (6, 8)):
        rid = f"tune_state_lr{lr:g}_cw{w}_d{depth}"
        jobs.append(c.job(common.COMMON_NAME, rid, "state", 0, lr=lr, cw=w, depth=depth))
    return jobs


EXPS = {"pilot_validity": pilot_validity, "pilot_compose": pilot_compose, "tune": tune}


def job_file(experiment: str, steps: int) -> Path:
    return ROOT / "exp" / "jobs" / f"{experiment}_steps{steps}.json"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=sorted(EXPS))
    ap.add_argument("--steps", type=int, default=6000, help="training steps (6000 = frozen protocol; lower for smoke tests)")
    ap.add_argument("--data", default=None, help="training data dir (default: game_code/output/data_v3)")
    ap.add_argument("--new", action="store_true", help="overwrite an existing job file: new timestamp, new folders")
    ap.add_argument("--out", type=Path, default=None, help="job file (default exp/jobs/<experiment>_steps<N>.json)")
    a = ap.parse_args(argv)
    path = a.out or job_file(a.experiment, a.steps)
    if path.exists() and not a.new:
        print(f"{path} exists - reusing it (resumes the same result folders). Use --new for fresh folders.")
        return
    jobs = EXPS[a.experiment](Ctx(a.steps, a.data, common.now_ts()))
    common.dump_json(jobs, path)
    print(f"{len(jobs)} jobs -> {common.rel(path)}\nresult folders: <results>/{jobs[0]['out'].split('/')[0]}/{common.result_name('train', '<id>', '<settings>', jobs[0]['out'].split('/')[1][:15])}")


if __name__ == "__main__":
    main()
