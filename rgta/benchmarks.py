"""Benchmark generation and suite runner for RGTA experiments."""

from __future__ import annotations

import csv
import random
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from .allocators import AllocatorConfig, make_allocator
from .grid import GridMap, make_kiva_map, make_sorting_map
from .simulator import SimulationConfig, SimulationResult, run_simulation
from .types import Agent, Task


def build_benchmark(
    map_name: str,
    num_agents: int,
    capacity: int,
    seed: int,
    total_tasks: int = 2600,
    initial_tasks: int = 600,
    release_batch: int = 10,
    release_interval: int = 5,
    targets_per_task: int = 3,
    task_profile: str = "efficient_random",
) -> tuple[GridMap, list[Agent], list[Task]]:
    grid = make_kiva_map() if map_name == "kiva" else make_sorting_map()
    agents = generate_agents(grid, num_agents, capacity)
    tasks = generate_tasks(grid, total_tasks, initial_tasks, release_batch, release_interval, targets_per_task, seed, task_profile)
    return grid, agents, tasks


def generate_agents(grid: GridMap, num_agents: int, capacity: int) -> list[Agent]:
    homes = list(grid.home_nodes)
    if len(homes) < num_agents:
        homes.extend(node for node in grid.free_nodes if node not in homes)
    return [Agent(agent_id=i, home=homes[i % len(homes)], capacity=capacity) for i in range(num_agents)]


def generate_tasks(
    grid: GridMap,
    total_tasks: int,
    initial_tasks: int,
    release_batch: int,
    release_interval: int,
    targets_per_task: int,
    seed: int,
    task_profile: str = "efficient_random",
) -> list[Task]:
    rng = random.Random(seed)
    pickups = list(grid.pickup_nodes)
    if task_profile == "efficient_random":
        return _generate_efficient_random_tasks(rng, pickups, total_tasks, initial_tasks, release_batch, release_interval, targets_per_task)
    if task_profile != "rgta_stress":
        raise ValueError(f"unknown task profile: {task_profile}")
    return _generate_clustered_tasks(rng, grid, pickups, total_tasks, initial_tasks, release_batch, release_interval, targets_per_task, seed)


def run_benchmark_suite(
    maps: list[str],
    settings: list[tuple[int, int]],
    methods: list[str],
    seeds: list[int],
    total_tasks: int,
    initial_tasks: int,
    release_batch: int,
    release_interval: int,
    allocator_config: AllocatorConfig,
    sim_config: SimulationConfig | None = None,
    output_csv: str | None = None,
    task_profile: str = "efficient_random",
) -> list[SimulationResult]:
    results: list[SimulationResult] = []
    for map_name in maps:
        for num_agents, capacity in settings:
            for seed in seeds:
                base_grid, _, base_tasks = build_benchmark(
                    map_name,
                    num_agents,
                    capacity,
                    seed,
                    total_tasks,
                    initial_tasks,
                    release_batch,
                    release_interval,
                    task_profile=task_profile,
                )
                for method in methods:
                    agents = generate_agents(base_grid, num_agents, capacity)
                    method_config = AllocatorConfig(**{**allocator_config.__dict__, "capacity": capacity})
                    allocator = make_allocator(method, method_config, seed=seed)
                    result = run_simulation(base_grid, agents, list(base_tasks), allocator, seed, sim_config)
                    results.append(result)
                    print(
                        f"{map_name}({num_agents},{capacity}) seed={seed} {method}: "
                        f"service={result.average_service_time:.1f} makespan={result.makespan:.1f} "
                        f"alloc_ms/event={result.allocation_runtime_ms_per_event:.2f}",
                        flush=True,
                    )
    if output_csv and results:
        write_results_csv(results, output_csv)
    return results


def write_results_csv(results: list[SimulationResult], output_csv: str) -> None:
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def summarize(results: list[SimulationResult]) -> list[dict[str, float | str | int]]:
    groups: dict[tuple[str, int, int, str], list[SimulationResult]] = {}
    for result in results:
        key = (result.map_name, result.num_agents, result.capacity, result.method)
        groups.setdefault(key, []).append(result)
    rows: list[dict[str, float | str | int]] = []
    for (map_name, num_agents, capacity, method), values in sorted(groups.items()):
        rows.append(
            {
                "map": map_name,
                "agents": num_agents,
                "capacity": capacity,
                "method": method,
                "service": mean(value.average_service_time for value in values),
                "makespan": mean(value.makespan for value in values),
                "alloc_ms_event": mean(value.allocation_runtime_ms_per_event for value in values),
                "alloc_ms_step": mean(value.allocation_runtime_ms_per_step for value in values),
            }
        )
    return rows


def _generate_efficient_random_tasks(
    rng: random.Random,
    pickups: list[int],
    total_tasks: int,
    initial_tasks: int,
    release_batch: int,
    release_interval: int,
    targets_per_task: int,
) -> list[Task]:
    tasks: list[Task] = []
    for task_id in range(total_tasks):
        if len(pickups) >= targets_per_task:
            targets = tuple(rng.sample(pickups, targets_per_task))
        else:
            targets = tuple(rng.choice(pickups) for _ in range(targets_per_task))
        tasks.append(Task(task_id=task_id, targets=targets, release_time=_release_time(task_id, initial_tasks, release_batch, release_interval)))
    return tasks


def _generate_clustered_tasks(
    rng: random.Random,
    grid: GridMap,
    pickups: list[int],
    total_tasks: int,
    initial_tasks: int,
    release_batch: int,
    release_interval: int,
    targets_per_task: int,
    seed: int,
) -> list[Task]:
    centers = sorted(pickups, key=lambda node: (grid.xy(node)[0], grid.xy(node)[1]))
    centers = centers[:: max(1, len(centers) // max(16, int(total_tasks**0.5)))] or pickups
    tasks: list[Task] = []
    for task_id in range(total_tasks):
        center = centers[(task_id + seed) % len(centers)]
        neighborhood = sorted(pickups, key=lambda node: grid.manhattan(center, node))[: max(12, targets_per_task * 4)]
        targets = tuple(rng.sample(neighborhood, targets_per_task))
        tasks.append(Task(task_id=task_id, targets=targets, release_time=_release_time(task_id, initial_tasks, release_batch, release_interval)))
    return tasks


def _release_time(task_id: int, initial_tasks: int, release_batch: int, release_interval: int) -> int:
    if task_id < initial_tasks:
        return 0
    return ((task_id - initial_tasks) // release_batch + 1) * release_interval

