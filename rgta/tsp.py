"""TSP route-cost oracle and online insertion estimators."""

from __future__ import annotations

from itertools import permutations
from math import inf

from .grid import GridMap, INF_DISTANCE
from .types import Task, TaskGroup


class RouteCostOracle:
    def __init__(self, grid: GridMap):
        self.grid = grid
        self._cache: dict[tuple[int, int, tuple[int, ...]], tuple[tuple[int, ...], float]] = {}

    def route_cost(self, sequence: tuple[int, ...], start: int, home: int) -> float:
        if not sequence:
            return 0.0
        cost = self.grid.distance(start, sequence[0])
        if cost >= INF_DISTANCE:
            return inf
        for prev, curr in zip(sequence, sequence[1:]):
            step = self.grid.distance(prev, curr)
            if step >= INF_DISTANCE:
                return inf
            cost += step
        back = self.grid.distance(sequence[-1], home)
        if back >= INF_DISTANCE:
            return inf
        return float(cost + back)

    def solve_full_tsp(self, targets: tuple[int, ...], start: int, home: int) -> tuple[tuple[int, ...], float]:
        unique_targets = tuple(dict.fromkeys(targets))
        if not unique_targets:
            return (), 0.0
        key = (start, home, tuple(sorted(unique_targets)))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if len(unique_targets) <= 10:
            answer = self._held_karp(unique_targets, start, home)
        else:
            answer = self._nearest_neighbor_2opt(unique_targets, start, home)
        self._cache[key] = answer
        return answer

    def estimate_insertion(self, group: TaskGroup, task: Task, mode: str = "eff") -> tuple[float, tuple[int, ...], float]:
        if mode == "full":
            sequence, new_cost = self.solve_full_tsp(group.all_targets() + task.targets, group.start, group.home)
            return new_cost - group.cost, sequence, new_cost
        return self._online_block_insertion(group, task)

    def merge_task(self, group: TaskGroup, task: Task, mode: str = "eff", refine_after_insert: bool = True) -> None:
        _, sequence, new_cost = self.estimate_insertion(group, task, mode)
        group.tasks.append(task)
        group.sequence = sequence
        group.cost = new_cost
        if refine_after_insert:
            group.sequence, group.cost = self.solve_full_tsp(group.all_targets(), group.start, group.home)

    def rebuild_group(self, group: TaskGroup) -> None:
        group.sequence, group.cost = self.solve_full_tsp(group.all_targets(), group.start, group.home)

    def estimate_replacement(self, group: TaskGroup, old_task: Task, new_task: Task) -> tuple[float, tuple[int, ...], float]:
        remaining_tasks = [task for task in group.tasks if task.task_id != old_task.task_id]
        required_targets: set[int] = set()
        ordered_required: list[int] = []
        for task in remaining_tasks:
            for target in task.targets:
                if target not in required_targets:
                    required_targets.add(target)
                    ordered_required.append(target)
        filtered = [target for target in group.sequence if target in required_targets]
        seen = set(filtered)
        filtered.extend(target for target in ordered_required if target not in seen)
        base_sequence = tuple(filtered)
        base_cost = self.route_cost(base_sequence, group.start, group.home)
        if base_cost == inf:
            base_sequence, base_cost = self._nearest_neighbor_2opt(tuple(ordered_required), group.start, group.home)
        base_group = TaskGroup(
            group.agent_id,
            group.start,
            group.home,
            group.capacity,
            remaining_tasks,
            base_sequence,
            base_cost,
        )
        _, sequence, cost = self._online_block_insertion(base_group, new_task)
        if cost == inf:
            targets = tuple(dict.fromkeys(base_group.all_targets() + new_task.targets))
            sequence, cost = self._nearest_neighbor_2opt(targets, group.start, group.home)
        return cost - group.cost, sequence, cost

    def _online_block_insertion(self, group: TaskGroup, task: Task) -> tuple[float, tuple[int, ...], float]:
        current = group.sequence
        current_set = set(current)
        new_targets = tuple(target for target in task.targets if target not in current_set)
        if not new_targets:
            return 0.0, current, group.cost
        if not current:
            sequence, cost = self.solve_full_tsp(new_targets, group.start, group.home)
            return cost, sequence, cost

        best_delta = inf
        best_sequence: tuple[int, ...] | None = None
        endpoints = (group.start,) + current + (group.home,)
        orderings = permutations(new_targets) if len(new_targets) <= 7 else (new_targets,)
        for ordering in orderings:
            block_cost = 0
            feasible = True
            for left, right in zip(ordering, ordering[1:]):
                step = self.grid.distance(left, right)
                if step >= INF_DISTANCE:
                    feasible = False
                    break
                block_cost += step
            if not feasible:
                continue
            for pos in range(len(endpoints) - 1):
                left, right = endpoints[pos], endpoints[pos + 1]
                old_edge = self.grid.distance(left, right)
                first = self.grid.distance(left, ordering[0])
                last = self.grid.distance(ordering[-1], right)
                if min(old_edge, first, last) >= INF_DISTANCE:
                    continue
                delta = first + block_cost + last - old_edge
                if delta < best_delta:
                    best_delta = float(delta)
                    best_sequence = current[:pos] + tuple(ordering) + current[pos:]
        if best_sequence is None:
            return inf, current, inf
        return best_delta, best_sequence, group.cost + best_delta

    def _held_karp(self, targets: tuple[int, ...], start: int, home: int) -> tuple[tuple[int, ...], float]:
        n = len(targets)
        dp: dict[tuple[int, int], float] = {}
        parent: dict[tuple[int, int], int] = {}
        for j, target in enumerate(targets):
            dp[(1 << j, j)] = float(self.grid.distance(start, target))
        for mask in range(1, 1 << n):
            for j in range(n):
                if not (mask & (1 << j)):
                    continue
                base = dp.get((mask, j), inf)
                if base == inf:
                    continue
                for k in range(n):
                    if mask & (1 << k):
                        continue
                    step = self.grid.distance(targets[j], targets[k])
                    if step >= INF_DISTANCE:
                        continue
                    next_mask = mask | (1 << k)
                    value = base + step
                    key = (next_mask, k)
                    if value < dp.get(key, inf):
                        dp[key] = value
                        parent[key] = j
        full = (1 << n) - 1
        best_cost = inf
        best_last = -1
        for j, target in enumerate(targets):
            value = dp.get((full, j), inf)
            back = self.grid.distance(target, home)
            if value + back < best_cost:
                best_cost = float(value + back)
                best_last = j
        if best_last < 0:
            return (), inf
        order = [best_last]
        mask = full
        last = best_last
        while mask != (1 << last):
            prev = parent[(mask, last)]
            order.append(prev)
            mask ^= 1 << last
            last = prev
        order.reverse()
        return tuple(targets[i] for i in order), best_cost

    def _nearest_neighbor_2opt(self, targets: tuple[int, ...], start: int, home: int) -> tuple[tuple[int, ...], float]:
        remaining = list(targets)
        current = start
        sequence: list[int] = []
        while remaining:
            nxt = min(remaining, key=lambda node: self.grid.distance(current, node))
            sequence.append(nxt)
            remaining.remove(nxt)
            current = nxt
        improved = True
        while improved:
            improved = False
            base_cost = self.route_cost(tuple(sequence), start, home)
            for i in range(len(sequence) - 1):
                for j in range(i + 2, len(sequence) + 1):
                    candidate = sequence[:i] + list(reversed(sequence[i:j])) + sequence[j:]
                    if self.route_cost(tuple(candidate), start, home) + 1e-9 < base_cost:
                        sequence = candidate
                        improved = True
                        break
                if improved:
                    break
        return tuple(sequence), self.route_cost(tuple(sequence), start, home)

