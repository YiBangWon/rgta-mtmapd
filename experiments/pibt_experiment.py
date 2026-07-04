"""MAPF-integrated validation for RGTA task assignment.

Executes task groups on the real grid with per-timestep agent movement and
PIBT collision resolution (Okumura et al., IJCAI 2019), instead of the
route-cost surrogate. The rgta package is used read-only for maps, task
generation, allocators, and the TSP route oracle.

Usage:
  python3 pibt_experiment.py --map kiva --agents 40 --capacity 6 \
      --methods tp_tsp rgta_eff --seeds 0 1 2 3 4 --output out.csv
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

RGTA_ROOT = str(Path(__file__).resolve().parents[1])
if RGTA_ROOT not in sys.path:
    sys.path.insert(0, RGTA_ROOT)

from rgta.allocators import AllocatorConfig, make_allocator  # noqa: E402
from rgta.benchmarks import build_benchmark  # noqa: E402
from rgta.tsp import RouteCostOracle  # noqa: E402

MAX_STEPS = 500_000


class ExecAgent:
    """Per-agent execution state on the grid."""

    def __init__(self, agent, pos):
        self.agent = agent          # rgta Agent (queue managed by allocator)
        self.pos = pos
        self.route: list[int] = []  # remaining waypoints (targets in TSP order, then home)
        self.group = None
        self.priority = 0.0

    @property
    def goal(self):
        return self.route[0] if self.route else self.pos


def pibt_step(exec_agents, grid, rng):
    """One PIBT timestep; returns dict agent_index -> next node."""
    order = sorted(range(len(exec_agents)), key=lambda i: -exec_agents[i].priority)
    occupied_now = {a.pos: i for i, a in enumerate(exec_agents)}
    nxt: dict[int, int] = {}
    reserved: dict[int, int] = {}

    def pibt(i, parent_pos):
        a = exec_agents[i]
        cands = grid.neighbors(a.pos) + [a.pos]
        cands.sort(key=lambda u: (grid.distance(u, a.goal), rng.random()))
        for u in cands:
            if u in reserved:
                continue
            if parent_pos is not None and u == parent_pos:
                continue
            reserved[u] = i
            j = occupied_now.get(u)
            if j is not None and j != i and j not in nxt:
                if not pibt(j, a.pos):
                    del reserved[u]
                    continue
            nxt[i] = u
            return True
        nxt[i] = a.pos
        return False

    for i in order:
        if i not in nxt:
            pibt(i, None)
    return nxt


def run_one(map_name, num_agents, capacity, seed, method, total_tasks, initial_tasks,
            release_batch, release_interval):
    grid, agents, tasks = build_benchmark(
        map_name, num_agents, capacity, seed,
        total_tasks=total_tasks, initial_tasks=initial_tasks,
        release_batch=release_batch, release_interval=release_interval,
    )
    oracle = RouteCostOracle(grid)
    config = AllocatorConfig(capacity=capacity)
    allocator = make_allocator(method, config, seed=seed)
    rng = random.Random(seed * 7919 + 13)

    homes_seen = set()
    for a in agents:
        if a.home in homes_seen:
            raise RuntimeError("duplicate home cell; reduce agent count")
        homes_seen.add(a.home)
    exec_agents = [ExecAgent(a, a.home) for a in agents]

    pending = sorted(tasks, key=lambda x: (x.release_time, x.task_id))
    backlog = []
    completed: dict[int, int] = {}
    pi = 0
    t = 0
    alloc_seconds = 0.0
    conflicts = 0

    while len(completed) < len(tasks):
        if t > MAX_STEPS:
            raise RuntimeError(f"exceeded {MAX_STEPS} steps (livelock?)")
        while pi < len(pending) and pending[pi].release_time <= t:
            backlog.append(pending[pi])
            pi += 1
        if backlog:
            t0 = time.perf_counter()
            allocator.update_queued_groups(agents, backlog, oracle, float(t))
            allocator.fill_queues(agents, backlog, oracle, float(t))
            alloc_seconds += time.perf_counter() - t0
        for ea in exec_agents:
            if ea.group is None and ea.agent.queue:
                group = ea.agent.queue.pop(0)
                ea.group = group
                ea.route = list(group.sequence) + [group.home]
                ea.agent.active_end_time = float(t)

        nxt = pibt_step(exec_agents, grid, rng)
        positions = list(nxt.values())
        if len(set(positions)) != len(positions):
            conflicts += 1  # should never happen with PIBT
        for i, ea in enumerate(exec_agents):
            ea.pos = nxt[i]
            while ea.route and ea.pos == ea.route[0]:
                ea.route.pop(0)
                if not ea.route and ea.group is not None:
                    for task in ea.group.tasks:
                        completed[task.task_id] = t
                    ea.group = None
            ea.priority = ea.priority + 1.0 if ea.route else 0.0
        t += 1

    service = sum(completed[x.task_id] - x.release_time for x in tasks) / len(tasks)
    makespan = max(completed.values())
    return {
        "method": allocator.name,
        "map_name": map_name,
        "num_agents": num_agents,
        "capacity": capacity,
        "seed": seed,
        "completed_tasks": len(completed),
        "average_service_time": service,
        "makespan": makespan,
        "alloc_seconds": round(alloc_seconds, 3),
        "steps": t,
        "pibt_conflicts": conflicts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="kiva", choices=["kiva", "sorting"])
    ap.add_argument("--agents", type=int, default=40)
    ap.add_argument("--capacity", type=int, default=6)
    ap.add_argument("--methods", nargs="+", default=["tp_tsp", "rgta_eff"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--total-tasks", type=int, default=800)
    ap.add_argument("--initial-tasks", type=int, default=200)
    ap.add_argument("--release-batch", type=int, default=10)
    ap.add_argument("--release-interval", type=int, default=5)
    ap.add_argument("--output", default="pibt_results.csv")
    args = ap.parse_args()

    rows = []
    for seed in args.seeds:
        for method in args.methods:
            r = run_one(args.map, args.agents, args.capacity, seed, method,
                        args.total_tasks, args.initial_tasks,
                        args.release_batch, args.release_interval)
            rows.append(r)
            print(f"{args.map}({args.agents},{args.capacity}) seed={seed} {method}: "
                  f"service={r['average_service_time']:.1f} makespan={r['makespan']} "
                  f"steps={r['steps']} conflicts={r['pibt_conflicts']}", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")

    by = {}
    for r in rows:
        by.setdefault(r["method"], []).append(r)
    print("\nSummary (means)")
    for m, rs in sorted(by.items()):
        svc = sum(x["average_service_time"] for x in rs) / len(rs)
        mk = sum(x["makespan"] for x in rs) / len(rs)
        print(f"  {m}: service={svc:.1f} makespan={mk:.1f}")


if __name__ == "__main__":
    main()
