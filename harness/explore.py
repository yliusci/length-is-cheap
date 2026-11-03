"""Massive exploration engine, persistent and resumable.

Usage:
    uv run python harness/explore.py experiments/exp_demo.py --workers 9 --hours 72

An experiment is a Python module that defines:
    CONFIG = {"threshold": float, "replicates": int, "exploit_frac": float}
    sample(rng) -> dict                     # a random point of the space
    run(params: dict, seed: int) -> dict    # must return at least {"score": float}
    perturb(params, rng, scale) -> dict     # optional, otherwise generic perturbation
    control(params, seed) -> dict           # optional, control condition (must NOT show the effect)

Everything is logged by append to runs/<exp>/results.jsonl (crash-resistant).
The state (counters, veins) is checkpointed in runs/<exp>/state.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
_EXP = None


def load_experiment(path: str):
    spec = importlib.util.spec_from_file_location("experiment", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _init_worker(path: str):
    global _EXP
    _EXP = load_experiment(path)


def default_perturb(params: dict, rng: np.random.Generator, scale: float) -> dict:
    out = {}
    for k, v in params.items():
        if isinstance(v, bool):
            out[k] = (not v) if rng.random() < 0.1 * scale else v
        elif isinstance(v, int):
            out[k] = v + int(rng.integers(-1, 2)) if rng.random() < scale else v
        elif isinstance(v, float):
            out[k] = v * math.exp(rng.normal(0, 0.3 * scale)) if v != 0 else rng.normal(0, scale)
        else:
            out[k] = v
    return out


def _task(kind: str, params: dict, seed: int, parent: str | None):
    t0 = time.time()
    try:
        if kind == "control":
            metrics = _EXP.control(params, seed)
        else:
            metrics = _EXP.run(params, seed)
        err = None
    except Exception:
        metrics, err = {"score": float("-inf")}, traceback.format_exc(limit=3)
    return {
        "kind": kind, "params": params, "seed": seed, "parent": parent,
        "metrics": metrics, "error": err, "dt": round(time.time() - t0, 4), "t": time.time(),
    }


def mem_available_pct() -> int:
    """Available memory according to macOS (same indicator as memory_pressure), in %."""
    try:
        return int(subprocess.run(["sysctl", "-n", "kern.memorystatus_level"], capture_output=True,
                                  text=True, timeout=5).stdout.strip())
    except Exception:
        try:
            import psutil  # Windows / Linux (laptop)
            return int(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)
        except Exception:
            return 100


MEM_PAUSE_PCT = int(os.environ.get("MEM_PAUSE_PCT", 20))
MEM_STOP_PCT = int(os.environ.get("MEM_STOP_PCT", 12))


def notify(title: str, msg: str):
    try:
        subprocess.run(["osascript", "-e", f'display notification "{msg}" with title "{title}"'],
                       check=False, timeout=5)
    except Exception:
        pass


class Explorer:
    def __init__(self, exp_path: str, workers: int, hours: float, seed: int, tag: str = ""):
        self.exp_path = exp_path
        self.exp = load_experiment(exp_path)
        self.cfg = {"threshold": 1.0, "replicates": 5, "exploit_frac": 0.7,
                    "max_filons": 8, "control_margin": 0.5, **getattr(self.exp, "CONFIG", {})}
        self.name = Path(exp_path).stem + (f"_{tag}" if tag else "")
        self.dir = ROOT / "runs" / self.name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.results = self.dir / "results.jsonl"
        self.state_path = self.dir / "state.json"
        self.workers = workers
        self.deadline = time.time() + hours * 3600
        self.rng = np.random.default_rng(seed)
        self.state = {"n": 0, "n_err": 0, "best": None, "filons": {}, "confirmed": [], "started": time.time()}
        if self.state_path.exists():
            self.state.update(json.loads(self.state_path.read_text()))
            print(f"[resume] {self.state['n']} evaluations already done", flush=True)
        self.stop = False
        signal.signal(signal.SIGTERM, self._sig)
        signal.signal(signal.SIGINT, self._sig)

    def _sig(self, *_):
        self.stop = True

    def checkpoint(self):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, default=float, indent=1))
        os.replace(tmp, self.state_path)

    def next_task(self):
        """Chooses explore / exploit / replicate / control."""
        filons = self.state["filons"]
        # 1) priority to pending replications and controls: a vein must be confirmed before it is exploited
        for fid, f in filons.items():
            if f["status"] == "pending":
                if f["n_rep"] < self.cfg["replicates"]:
                    f["n_rep"] += 1
                    return "replicate", f["params"], int(self.rng.integers(2**31)), fid
                if hasattr(self.exp, "control") and f["n_ctl"] < self.cfg["replicates"]:
                    f["n_ctl"] += 1
                    return "control", f["params"], int(self.rng.integers(2**31)), fid
        active = [f for f in filons.values() if f["status"] == "confirmed"]
        if active and self.rng.random() < self.cfg["exploit_frac"]:
            f = max(active, key=lambda f: f["best"])
            scale = max(0.05, 1.0 / math.sqrt(1 + f["n_exploit"] / 50))
            f["n_exploit"] += 1
            pert = getattr(self.exp, "perturb", default_perturb)
            return "exploit", pert(f["best_params"], self.rng, scale), int(self.rng.integers(2**31)), f["id"]
        return "explore", self.exp.sample(self.rng), int(self.rng.integers(2**31)), None

    def ingest(self, r: dict):
        with self.results.open("a") as fh:
            fh.write(json.dumps(r, default=float) + "\n")
        self.state["n"] += 1
        if r["error"]:
            self.state["n_err"] += 1
            return
        s = float(r["metrics"].get("score", float("-inf")))
        if self.state["best"] is None or s > self.state["best"]["score"]:
            self.state["best"] = {"score": s, "params": r["params"], "seed": r["seed"], "kind": r["kind"]}
        filons = self.state["filons"]
        thr = self.cfg["threshold"]
        if r["kind"] in ("explore", "exploit") and s >= thr:
            if r["kind"] == "exploit" and r["parent"] in filons:
                f = filons[r["parent"]]
                if s > f["best"]:
                    f["best"], f["best_params"] = s, r["params"]
            elif (same := getattr(self.exp, "same_basin", None)) and any(
                    same(r["params"], f["params"]) for f in filons.values()):
                pass  # same basin as a known vein: not a new vein
            elif len([f for f in filons.values() if f["status"] != "rejected"]) < self.cfg["max_filons"]:
                fid = f"F{len(filons) + 1:03d}"
                filons[fid] = {"id": fid, "params": r["params"], "best_params": r["params"], "best": s,
                               "status": "pending", "trigger": s, "n_rep": 0, "n_ctl": 0,
                               "rep": [], "ctl": [], "n_exploit": 0, "t": time.time()}
                print(f"[vein candidate] {fid} score={s:.4g}", flush=True)
        elif r["kind"] in ("replicate", "control") and r["parent"] in filons:
            f = filons[r["parent"]]
            f["rep" if r["kind"] == "replicate" else "ctl"].append(s)
            self._judge(f)

    def _judge(self, f: dict):
        k = self.cfg["replicates"]
        need_ctl = hasattr(self.exp, "control")
        if len(f["rep"]) < k or (need_ctl and len(f["ctl"]) < k):
            return
        rep = np.array(f["rep"])
        ok = rep.mean() >= self.cfg["threshold"] and rep.min() >= self.cfg["threshold"] * self.cfg["control_margin"]
        if need_ctl:
            ctl = np.array(f["ctl"])
            # the effect must disappear under control: standardized gap (Welch) > 3
            se = math.sqrt(rep.var(ddof=1) / len(rep) + ctl.var(ddof=1) / len(ctl)) + 1e-12
            f["t_stat"] = float((rep.mean() - ctl.mean()) / se)
            ok = ok and f["t_stat"] > 3
        f["status"] = "confirmed" if ok else "rejected"
        f["rep_mean"], f["rep_std"] = float(rep.mean()), float(rep.std(ddof=1))
        print(f"[vein {f['status']}] {f['id']} rep={f['rep_mean']:.4g}+/-{f['rep_std']:.3g}"
              + (f" t={f['t_stat']:.2f}" if need_ctl else ""), flush=True)
        if ok:
            self.state["confirmed"].append(f["id"])
            with (self.dir / "FILONS.md").open("a") as fh:
                fh.write(f"\n## {f['id']} - {time.strftime('%Y-%m-%d %H:%M')}\n"
                         f"- params: `{json.dumps(f['params'], default=float)}`\n"
                         f"- replications ({k} seeds): {f['rep_mean']:.6g} +/- {f['rep_std']:.3g}\n"
                         + (f"- control: {np.mean(f['ctl']):.6g}, t={f['t_stat']:.2f}\n" if need_ctl else ""))
            (self.dir / "ALERT").write_text(f["id"])
            notify("Vein confirmed", f"{self.name} {f['id']} score {f['rep_mean']:.4g}")

    def loop(self):
        if self.workers == 1:
            return self.loop_inline()
        last_ck, last_print = time.time(), time.time()
        with ProcessPoolExecutor(self.workers, initializer=_init_worker, initargs=(self.exp_path,)) as pool:
            inflight = {}
            while not self.stop and time.time() < self.deadline and not (ROOT / "STOP").exists():
                while len(inflight) < 2 * self.workers:
                    kind, params, seed, parent = self.next_task()
                    inflight[pool.submit(_task, kind, params, seed, parent)] = 1
                done, _ = wait(inflight, timeout=5, return_when=FIRST_COMPLETED)
                for fut in done:
                    inflight.pop(fut)
                    self.ingest(fut.result())
                now = time.time()
                if now - last_ck > 30:
                    self.checkpoint(); last_ck = now
                if now - last_print > 60:
                    b = self.state["best"]
                    rate = self.state["n"] / max(1, now - self.state["started"])
                    print(f"[{time.strftime('%H:%M:%S')}] n={self.state['n']} err={self.state['n_err']} "
                          f"best={(b['score'] if b else float('nan')):.4g} veins={len(self.state['filons'])} "
                          f"confirmed={len(self.state['confirmed'])} ~{rate:.1f}/s", flush=True)
                    last_print = now
            for fut in inflight:
                fut.cancel()
        self.checkpoint()
        print("[end] checkpoint written", flush=True)


    def loop_inline(self):
        """workers=1: everything in the main process (MPS/MLX GPU, model kept in memory)."""
        _init_worker(self.exp_path)
        last_ck = last_print = time.time()
        while not self.stop and time.time() < self.deadline and not (ROOT / "STOP").exists():
            m = mem_available_pct()
            if m < MEM_STOP_PCT:
                print(f"[STOP memory] available {m} % < {MEM_STOP_PCT} %", flush=True)
                notify("Memory stop", f"{self.name}: {m} % available")
                break
            if m < MEM_PAUSE_PCT:
                print(f"[memory pause] available {m} % < {MEM_PAUSE_PCT} %", flush=True)
                time.sleep(30)
                continue
            self.ingest(_task(*self.next_task()))
            now = time.time()
            if now - last_ck > 30:
                self.checkpoint(); last_ck = now
            if now - last_print > 60:
                b = self.state["best"]
                print(f"[{time.strftime('%H:%M:%S')}] n={self.state['n']} err={self.state['n_err']} "
                      f"best={(b['score'] if b else float('nan')):.4g} veins={len(self.state['filons'])} "
                      f"confirmed={len(self.state['confirmed'])}", flush=True)
                last_print = now
        self.checkpoint()
        print("[end] checkpoint written", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--hours", type=float, default=24 * 7)
    ap.add_argument("--seed", type=int, default=int(time.time()))
    ap.add_argument("--tag", default="", help="suffix of the runs/ folder (several instances of the same experiment)")
    a = ap.parse_args()
    Explorer(a.experiment, a.workers, a.hours, a.seed, a.tag).loop()
