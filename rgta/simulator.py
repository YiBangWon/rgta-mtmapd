"""Event-based online MT-MAPD simulator for task-assignment experiments."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from math import ceil, inf

from .allocators import BaseAllocator
from .grid import GridMap
from .tsp import RouteCostOracle
from .types import Agent, Task


@dataclass
class SimulationConfig:
    target_service_time: float = 0.0
    max_queue_groups: int = 10
    route_cost_exponent: float = 1.75
    route_cost_normalizer: float = 80.0
    span_delay_weight: float = 0.0
    overlap_delay_weight: float = 0.0


@dataclass
class SimulationResult:
    method: str
    map_name: str
    num_agents: int
    capacity: int
    seed: int
    completed_tasks: int
    average_service_time: float
    makespan: float
    allocation_runtime_ms: float
    allocation_runtime_ms_per_event: float
    allocation_runtime_ms_per_step: float
    generated_groups: int
    assigned_tasks: int


def run_simulation(
    grid: GridMap,
    agents: list[Agent],
    tasks: list[Task],
    allocator: BaseAllocator,
    seed: int,
    config: SimulationConfig | None = None,
) -> SimulationResult:
    config = config or SimulationConfig(max_queue_groups=allocator.config.max_queue_groups)
    oracle = RouteCostOracle(grid)
    pending = deque(sorted(tasks, key=lambda task: (task.release_time, task.task_id)))
    backlog: list[Task] = []
    completed: dict[int, float] = {}
    now = 0.0
    allocation_seconds = 0.0
    allocation_events = 0

    def release_ready() -> None:
        while pending and pending[0].release_time <= now:
            backlog.append(pending.popleft())

    def complete_ready() -> None:
        for agent in agents:
            if agent.active_group is not None and agent.active_end_time <= now + 1e-9:
                for task in agent.active_group.tasks:
                    completed[task.task_id] = agent.active_end_time
                agent.active_group = None
                agent.completed_groups += 1

    def start_ready() -> None:
        active_signatures = [
            _route_signature(grid, agent.active_group)
            for agent in agents
            if agent.active_group is not None
        ]
        for agent in agents:
            if agent.is_free(now) and agent.queue:
                group = agent.queue.pop(0)
                signature = _route_signature(grid, group)
                duration = _execution_duration(grid, group, signature, active_signatures, config)
                active_signatures.append(signature)
                agent.active_group = group
                agent.active_end_time = now + duration

    release_ready()
    while len(completed) < len(tasks):
        complete_ready()
        start_ready()
        if backlog:
            start_time = time.perf_counter()
            allocator.update_queued_groups(agents, backlog, oracle, now)
            allocator.fill_queues(agents, backlog, oracle, now)
            allocation_seconds += time.perf_counter() - start_time
            allocation_events += 1
            start_ready()
        if len(completed) >= len(tasks):
            break
        next_release = pending[0].release_time if pending else inf
        next_finish = min((agent.active_end_time for agent in agents if agent.active_group is not None), default=inf)
        if next_release == inf and next_finish == inf:
            raise RuntimeError("simulation stalled")
        now = min(next_release, next_finish)
        release_ready()

    service_times = [completed[task.task_id] - task.release_time for task in tasks]
    makespan = max(completed.values(), default=0.0)
    runtime_ms = allocation_seconds * 1000.0
    return SimulationResult(
        method=allocator.name,
        map_name=grid.name,
        num_agents=len(agents),
        capacity=agents[0].capacity if agents else 0,
        seed=seed,
        completed_tasks=len(completed),
        average_service_time=sum(service_times) / max(1, len(service_times)),
        makespan=makespan,
        allocation_runtime_ms=runtime_ms,
        allocation_runtime_ms_per_event=runtime_ms / max(1, allocation_events),
        allocation_runtime_ms_per_step=runtime_ms / max(1, ceil(makespan) + 1),
        generated_groups=allocator.generated_groups,
        assigned_tasks=allocator.assigned_tasks,
    )


def _execution_duration(grid: GridMap, group, signature: set[tuple[int, int]], active_signatures: list[set[tuple[int, int]]], config: SimulationConfig) -> float:
    base = group.cost + group.load * config.target_service_time
    if base <= 0:
        return 0.0
    if config.route_cost_exponent != 1.0:
        normalized = max(base / max(1e-9, config.route_cost_normalizer), 1e-9)
        base = base * (normalized ** (config.route_cost_exponent - 1.0))
    if config.span_delay_weight > 0 and group.sequence:
        xs = [grid.xy(node)[0] for node in (group.start,) + group.sequence + (group.home,)]
        ys = [grid.xy(node)[1] for node in (group.start,) + group.sequence + (group.home,)]
        span = ((max(xs) - min(xs)) / max(1, grid.width - 1)) + ((max(ys) - min(ys)) / max(1, grid.height - 1))
        base += config.span_delay_weight * span * group.cost
    if config.overlap_delay_weight > 0 and signature:
        for active in active_signatures:
            if active:
                overlap = len(signature & active) / max(1, min(len(signature), len(active)))
                base += config.overlap_delay_weight * overlap * group.cost
    return base


def _route_signature(grid: GridMap, group) -> set[tuple[int, int]]:
    if group is None:
        return set()
    nodes = (group.start,) + group.sequence + (group.home,)
    signature: set[tuple[int, int]] = set()
    for left, right in zip(nodes, nodes[1:]):
        lx, ly = grid.xy(left)
        rx, ry = grid.xy(right)
        signature.add((min(lx, rx) // 4, min(ly, ry) // 4))
        signature.add((max(lx, rx) // 4, max(ly, ry) // 4))
        if lx == rx:
            y0, y1 = sorted((ly, ry))
            for y in range(y0, y1 + 1, 4):
                signature.add((lx // 4, y // 4))
        elif ly == ry:
            x0, x1 = sorted((lx, rx))
            for x in range(x0, x1 + 1, 4):
                signature.add((x // 4, ly // 4))
        else:
            x0, x1 = sorted((lx, rx))
            y0, y1 = sorted((ly, ry))
            for x in range(x0, x1 + 1, 4):
                signature.add((x // 4, ly // 4))
            for y in range(y0, y1 + 1, 4):
                signature.add((rx // 4, y // 4))
    return signature

