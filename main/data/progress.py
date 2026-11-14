"""Read-only progress events; logs stay outside the frozen dataset identity."""
from contextlib import contextmanager
import math
import queue
from threading import Thread
import time

_EVENTS = None
_INTERVAL = 10.0


def duration(seconds):
    if seconds is None or not math.isfinite(seconds):
        return "估算中"
    seconds = max(0, int(math.ceil(seconds)))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours}小时{minutes:02d}分" if hours else f"{minutes}分{seconds:02d}秒"


def stage_estimate(event, now):
    elapsed = max(0, now - event["started"])
    done, total = event["done"], event["total"]
    rate = done / elapsed if done > 0 and elapsed > 0 else None
    eta = max(0, total - done) / rate if total is not None and rate else None
    return elapsed, rate, eta


def initialize_worker(events, interval):
    global _EVENTS, _INTERVAL
    _EVENTS, _INTERVAL = events, interval


class Reporter:
    def __init__(self, task, events=None, interval=None):
        self.task = task
        self.events = _EVENTS if events is None else events
        self.interval = _INTERVAL if interval is None else interval
        self.current = None
        self.last = 0

    def start(self, stage, total=None, unit="项", category="stage"):
        self.current = {"kind": "progress", "task": self.task, "stage": stage, "total": total,
                        "done": 0, "unit": unit, "category": category, "started": time.monotonic(),
                        "status": "running"}
        self.update(0, total, force=True)

    def update(self, done, total=None, force=False):
        if self.current is None:
            return
        self.current["done"] = done
        if total is not None:
            self.current["total"] = total
        now = time.monotonic()
        if self.events is not None and (force or now - self.last >= self.interval or done == self.current["total"]):
            self.events.put(dict(self.current))
            self.last = now

    def finish(self):
        if self.current is not None:
            self.current["status"] = "finished"
            self.update(self.current["done"], force=True)

    @contextmanager
    def stage(self, label, total=None, unit="项", category="stage"):
        self.start(label, total, unit, category)
        try:
            yield self
        except BaseException:
            self.current["status"] = "failed"
            self.update(self.current["done"], force=True)
            raise
        else:
            self.finish()

    def committed(self, reused=False):
        if self.events is not None:
            self.events.put({"kind": "committed", "task": self.task, "reused": reused})


class Monitor:
    def __init__(self, tasks, config, workers, interval=10):
        if not 0 < interval < float("inf"):
            raise ValueError("progress interval must be finite and positive")
        import multiprocessing
        self.events = multiprocessing.get_context("spawn").Queue()
        self.tasks, self.workers, self.interval = list(tasks), workers, interval
        self.samples_per_task = config["episodes"] + config["val_episodes"] + config["test_scenes"]
        self.synthetic = [t for t in tasks if t != "T01"]
        self.active, self.sample_stages, self.committed_tasks, self.reused = {}, {}, set(), set()
        self.started = time.monotonic()
        self.thread = Thread(target=self.consume, daemon=True)

    def apply(self, event):
        if event["kind"] == "committed":
            self.committed_tasks.add(event["task"])
            self.active.pop(event["task"], None)
            if event["reused"]:
                self.reused.add(event["task"])
            print(f"[进度] {event['task']} {'复用完成' if event['reused'] else '生成并提交完成'}；"
                  f"任务 {len(self.committed_tasks)}/{len(self.tasks)}", flush=True)
            return
        self.active[event["task"]] = event
        if event["category"] == "samples":
            self.sample_stages[(event["task"], event["stage"])] = event
        if event["done"] == 0 or event["status"] != "running":
            self.print_stage(event)

    def print_stage(self, event, now=None):
        now = time.monotonic() if now is None else now
        elapsed, rate, eta = stage_estimate(event, now if event["status"] == "running" else event.get("finished_at", now))
        total, done = event["total"], event["done"]
        progress = f"{done:,}/{total:,} ({100*done/total:.1f}%)" if total else "处理中"
        speed = f"{rate:.2f}{event['unit']}/秒" if rate else "估算中"
        if event["unit"] == "字节" and rate:
            speed = f"{rate/1024**2:.1f}MiB/秒"
        status = {"running": "进行中", "finished": "阶段完成", "failed": "失败"}[event["status"]]
        suffix = duration(eta) if event["status"] == "running" else "—"
        print(f"[进度] {event['task']} | {event['stage']} | {progress} | {status} | "
              f"速度 {speed} | 已用 {duration(elapsed)} | 阶段剩余 {suffix}", flush=True)

    def sample_summary(self):
        total = len(self.synthetic) * self.samples_per_task
        reused = len(set(self.synthetic) & self.reused) * self.samples_per_task
        done = reused + sum(v["done"] for v in self.sample_stages.values())
        measured = sum(v["done"] for v in self.sample_stages.values())
        elapsed = sum(max(0, v.get("finished_at", time.monotonic()) - v["started"]) for v in self.sample_stages.values())
        rate = measured / elapsed if measured and elapsed else None
        remaining = max(0, total - done)
        unfinished = max(1, len(set(self.synthetic) - self.committed_tasks))
        eta = remaining / (rate * min(self.workers, unfinished)) if rate else None
        return done, total, rate, eta

    def snapshot(self):
        now = time.monotonic()
        for event in self.active.values():
            if event["status"] == "running":
                self.print_stage(event, now)
        done, total, rate, eta = self.sample_summary()
        summary = f"；新任务样本 {done:,}/{total:,}；样本阶段粗估剩余 {duration(eta)}（不含T01导入、写盘和审计）" if total else ""
        print(f"[总进度] 已提交任务 {len(self.committed_tasks)}/{len(self.tasks)}；"
              f"总已用 {duration(now-self.started)}{summary}", flush=True)

    def consume(self):
        last = time.monotonic()
        while True:
            try:
                event = self.events.get(timeout=max(.01, self.interval - (time.monotonic() - last)))
            except queue.Empty:
                event = None
            if event is not None:
                if event["kind"] == "stop":
                    return
                if event.get("status") in ("finished", "failed"):
                    event["finished_at"] = time.monotonic()
                self.apply(event)
            if time.monotonic() - last >= self.interval:
                self.snapshot()
                last = time.monotonic()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.events.put({"kind": "stop"})
        self.thread.join()
        self.events.close()
        self.events.join_thread()
        if exc_type is not None:
            print(f"[总进度] 生成失败：{exc}；未宣告整体完成", flush=True)
