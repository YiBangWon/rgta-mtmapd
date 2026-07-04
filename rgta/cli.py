"""Command-line runner for RGTA MT-MAPD benchmark experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from .allocators import AllocatorConfig
from .benchmarks import run_benchmark_suite, summarize
from .simulator import SimulationConfig

DEFAULT_METHODS = ["tp", "tsp_mapd", "tp_tsp", "rgta_eff", "rgta_full"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run RGTA MT-MAPD benchmark experiments")
    parser.add_argument("--maps", nargs="+", default=["kiva", "sorting"], choices=["kiva", "sorting"])
    parser.add_argument("--settings", nargs="+", default=["60x9", "100x9"], help="agent/capacity settings, e.g. 100x9")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--task-profile", default="efficient_random", choices=["efficient_random", "rgta_stress"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--total-tasks", type=int, default=2600)
    parser.add_argument("--initial-tasks", type=int, default=600)
    parser.add_argument("--release-batch", type=int, default=10)
    parser.add_argument("--release-interval", type=int, default=5)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--candidate-factor", type=float, default=1.5)
    parser.add_argument("--candidate-limit", type=int, default=None)
    parser.add_argument("--agent-subset-size", type=int, default=None)
    parser.add_argument("--max-queue-groups", type=int, default=10)
    parser.add_argument("--task-update-window", type=int, default=30)
    parser.add_argument("--update-task-limit", type=int, default=60)
    parser.add_argument("--route-cost-exponent", type=float, default=1.75)
    parser.add_argument("--route-cost-normalizer", type=float, default=80.0)
    parser.add_argument("--span-delay-weight", type=float, default=0.0)
    parser.add_argument("--overlap-delay-weight", type=float, default=0.0)
    parser.add_argument("--output", default="outputs/rgta_results.csv")
    args = parser.parse_args(argv)

    settings = [_parse_setting(value) for value in args.settings]
    allocator_config = AllocatorConfig(
        beta=args.beta,
        candidate_task_factor=args.candidate_factor,
        candidate_task_limit=args.candidate_limit,
        agent_subset_size=args.agent_subset_size,
        max_queue_groups=args.max_queue_groups,
        task_update_window=args.task_update_window,
        update_task_limit=args.update_task_limit,
    )
    sim_config = SimulationConfig(
        max_queue_groups=args.max_queue_groups,
        route_cost_exponent=args.route_cost_exponent,
        route_cost_normalizer=args.route_cost_normalizer,
        span_delay_weight=args.span_delay_weight,
        overlap_delay_weight=args.overlap_delay_weight,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results = run_benchmark_suite(
        maps=args.maps,
        settings=settings,
        methods=args.methods,
        seeds=args.seeds,
        total_tasks=args.total_tasks,
        initial_tasks=args.initial_tasks,
        release_batch=args.release_batch,
        release_interval=args.release_interval,
        allocator_config=allocator_config,
        sim_config=sim_config,
        output_csv=args.output,
        task_profile=args.task_profile,
    )
    print("\nSummary")
    print("map,agents,capacity,method,service,makespan,alloc_ms_event,alloc_ms_step")
    for row in summarize(results):
        print(
            f"{row['map']},{row['agents']},{row['capacity']},{row['method']},"
            f"{row['service']:.2f},{row['makespan']:.2f},"
            f"{row['alloc_ms_event']:.2f},{row['alloc_ms_step']:.4f}"
        )
    print(f"\nWrote raw results to {args.output}")
    return 0


def _parse_setting(value: str) -> tuple[int, int]:
    try:
        agents, capacity = value.lower().split("x", 1)
        return int(agents), int(capacity)
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"invalid setting '{value}', expected NxC such as 100x9") from exc


if __name__ == "__main__":
    raise SystemExit(main())

