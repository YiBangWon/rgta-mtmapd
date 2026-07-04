"""Task allocation policies, including regret-guided task assignment."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from math import inf

from .tsp import RouteCostOracle
from .types import Agent, Task, TaskGroup


@dataclass
class AllocatorConfig:
    capacity: int = 9
    beta: float = 0.05
    candidate_task_factor: float = 1.5
    candidate_task_limit: int | None = None
    agent_subset_size: int | None = None
    min_agent_subset: int = 2
    max_queue_groups: int = 10
    refine_after_insert: bool = True
    task_update_window: int = 30
    update_task_limit: int | None = 60


class BaseAllocator:
    name = "base"

    def __init__(self, config: AllocatorConfig, rng: random.Random):
        self.config = config
        self.rng = rng
        self.generated_groups = 0
        self.assigned_tasks = 0

    def fill_queues(self, agents: list[Agent], backlog: list[Task], oracle: RouteCostOracle, now: float) -> int:
        assigned = 0
        while backlog:
            selected_agents = self._select_agents(agents)
            if not selected_agents:
                break
            groups, selected_ids = self.generate_groups(selected_agents, agents, backlog, oracle)
            groups = [group for group in groups if group.tasks]
            if not groups or not selected_ids:
                break
            by_id = {agent.agent_id: agent for agent in agents}
            for group in groups:
                by_id[group.agent_id].queue.append(group)
                self.generated_groups += 1
            backlog[:] = [task for task in backlog if task.task_id not in selected_ids]
            assigned += len(selected_ids)
            self.assigned_tasks += len(selected_ids)
        return assigned

    def generate_groups(
        self,
        selected_agents: list[Agent],
        all_agents: list[Agent],
        backlog: list[Task],
        oracle: RouteCostOracle,
    ) -> tuple[list[TaskGroup], set[int]]:
        raise NotImplementedError

    def update_queued_groups(self, agents: list[Agent], backlog: list[Task], oracle: RouteCostOracle, now: float) -> int:
        return 0

    def _select_agents(self, agents: list[Agent]) -> list[Agent]:
        eligible = [agent for agent in agents if len(agent.queue) < self.config.max_queue_groups]
        eligible.sort(key=lambda agent: (len(agent.queue), agent.active_end_time, agent.agent_id))
        if not eligible:
            return []
        if self.config.agent_subset_size is None:
            count = len(eligible)
        else:
            count = min(len(eligible), max(self.config.min_agent_subset, self.config.agent_subset_size))
        return eligible[:count]

    def _candidate_tasks(self, all_agents: list[Agent], backlog: list[Task]) -> list[Task]:
        if self.config.candidate_task_limit is not None:
            limit = self.config.candidate_task_limit
        else:
            min_load = max(1, min((task.load for task in backlog), default=1))
            max_tasks_per_group = max(1, self.config.capacity // min_load)
            limit = math.ceil(self.config.candidate_task_factor * len(all_agents) * max_tasks_per_group)
        ordered = sorted(backlog, key=lambda task: (task.release_time, task.priority, task.task_id))
        return ordered[: min(len(ordered), max(1, limit))]

    def _new_group(self, agent: Agent) -> TaskGroup:
        return TaskGroup(agent_id=agent.agent_id, start=agent.home, home=agent.home, capacity=agent.capacity)


class RegretGuidedAllocator(BaseAllocator):
    def __init__(self, config: AllocatorConfig, rng: random.Random, insertion_mode: str = "eff", name: str = "rgta_eff"):
        super().__init__(config, rng)
        self.insertion_mode = insertion_mode
        self.name = name

    def update_queued_groups(self, agents: list[Agent], backlog: list[Task], oracle: RouteCostOracle, now: float) -> int:
        if not backlog or self.config.task_update_window <= 0:
            return 0
        new_tasks = [task for task in backlog if task.release_time == int(now)]
        if not new_tasks:
            return 0
        old_candidates: list[tuple[TaskGroup, Task]] = []
        for agent in agents:
            for group in agent.queue:
                for old_task in group.tasks:
                    if old_task.release_time >= now - self.config.task_update_window:
                        old_candidates.append((group, old_task))
        if not old_candidates:
            return 0
        limit = self.config.update_task_limit
        if limit is None:
            min_load = max(1, min((task.load for task in backlog), default=1))
            max_tasks_per_group = max(1, self.config.capacity // min_load)
            limit = math.ceil(self.config.candidate_task_factor * len(agents) * max_tasks_per_group)
        old_candidates.sort(key=lambda item: (-item[1].release_time, item[0].agent_id, item[1].task_id))
        old_candidates = old_candidates[:limit]

        replaced = 0
        for new_task in list(new_tasks):
            best_delta = 0.0
            best_group: TaskGroup | None = None
            best_old_task: Task | None = None
            best_sequence: tuple[int, ...] = ()
            best_cost = 0.0
            for group, old_task in old_candidates:
                if old_task not in group.tasks:
                    continue
                if group.load - old_task.load + new_task.load > group.capacity:
                    continue
                delta, sequence, cost = oracle.estimate_replacement(group, old_task, new_task)
                if delta < best_delta:
                    best_delta = delta
                    best_group = group
                    best_old_task = old_task
                    best_sequence = sequence
                    best_cost = cost
            if best_group is not None and best_old_task is not None:
                best_group.tasks = [new_task if task.task_id == best_old_task.task_id else task for task in best_group.tasks]
                best_group.sequence = best_sequence
                best_group.cost = best_cost
                backlog.remove(new_task)
                backlog.append(best_old_task)
                replaced += 1
        return replaced

    def generate_groups(
        self,
        selected_agents: list[Agent],
        all_agents: list[Agent],
        backlog: list[Task],
        oracle: RouteCostOracle,
    ) -> tuple[list[TaskGroup], set[int]]:
        tasks = self._candidate_tasks(all_agents, backlog)
        groups = [self._new_group(agent) for agent in selected_agents]
        selected_ids: set[int] = set()
        if not tasks or not groups:
            return groups, selected_ids

        costs = self._build_cost_matrix(groups, tasks, oracle)
        active = [True] * len(tasks)
        while True:
            best = self._best_second(costs, active)
            critical = [(idx, item[0]) for idx, item in enumerate(best) if active[idx] and item[2] == 1]
            if critical:
                task_index = min(critical, key=lambda item: (item[1], tasks[item[0]].task_id))[0]
            else:
                scored: list[tuple[float, float, int, int]] = []
                for idx, (first, second, feasible_count, _) in enumerate(best):
                    if not active[idx] or feasible_count < 2:
                        continue
                    regret = second - first
                    scored.append((first - self.config.beta * regret, first, tasks[idx].task_id, idx))
                if not scored:
                    break
                task_index = min(scored)[3]
            group_index = best[task_index][3]
            if group_index < 0 or costs[group_index][task_index] == inf:
                break
            task = tasks[task_index]
            oracle.merge_task(
                groups[group_index],
                task,
                mode=self.insertion_mode,
                refine_after_insert=self.config.refine_after_insert,
            )
            selected_ids.add(task.task_id)
            active[task_index] = False
            for col, candidate in enumerate(tasks):
                if not active[col]:
                    costs[group_index][col] = inf
                elif groups[group_index].can_fit(candidate):
                    costs[group_index][col] = oracle.estimate_insertion(groups[group_index], candidate, self.insertion_mode)[0]
                else:
                    costs[group_index][col] = inf
        return groups, selected_ids

    def _build_cost_matrix(self, groups: list[TaskGroup], tasks: list[Task], oracle: RouteCostOracle) -> list[list[float]]:
        rows: list[list[float]] = []
        for group in groups:
            row: list[float] = []
            for task in tasks:
                if group.can_fit(task):
                    row.append(oracle.estimate_insertion(group, task, self.insertion_mode)[0])
                else:
                    row.append(inf)
            rows.append(row)
        return rows

    @staticmethod
    def _best_second(costs: list[list[float]], active: list[bool]) -> list[tuple[float, float, int, int]]:
        if not costs:
            return []
        result: list[tuple[float, float, int, int]] = []
        for col in range(len(costs[0])):
            if not active[col]:
                result.append((inf, inf, 0, -1))
                continue
            first = second = inf
            count = 0
            first_group = -1
            for row, values in enumerate(costs):
                value = values[col]
                if value == inf:
                    continue
                count += 1
                if value < first:
                    second = first
                    first = value
                    first_group = row
                elif value < second:
                    second = value
            result.append((first, second, count, first_group))
        return result


class SequentialGreedyAllocator(BaseAllocator):
    name = "sequential"

    def _select_agents(self, agents: list[Agent]) -> list[Agent]:
        eligible = [agent for agent in agents if len(agent.queue) < self.config.max_queue_groups]
        eligible.sort(key=lambda agent: (agent.active_end_time, agent.agent_id))
        return eligible[:1]

    def generate_groups(self, selected_agents: list[Agent], all_agents: list[Agent], backlog: list[Task], oracle: RouteCostOracle):
        available = self._candidate_tasks(all_agents, backlog)
        groups = [self._new_group(agent) for agent in selected_agents]
        selected_ids: set[int] = set()
        for group in groups:
            while True:
                candidates = [task for task in available if task.task_id not in selected_ids and group.can_fit(task)]
                if not candidates:
                    break
                task = min(candidates, key=lambda item: (oracle.estimate_insertion(group, item, "eff")[0], item.task_id))
                oracle.merge_task(group, task, mode="eff", refine_after_insert=self.config.refine_after_insert)
                selected_ids.add(task.task_id)
        return groups, selected_ids


class NearestGroupAllocator(BaseAllocator):
    name = "tp_tsp"

    def _select_agents(self, agents: list[Agent]) -> list[Agent]:
        eligible = [agent for agent in agents if len(agent.queue) < self.config.max_queue_groups]
        eligible.sort(key=lambda agent: (agent.active_end_time, agent.agent_id))
        return eligible[:1]

    def generate_groups(self, selected_agents: list[Agent], all_agents: list[Agent], backlog: list[Task], oracle: RouteCostOracle):
        available = self._candidate_tasks(all_agents, backlog)
        groups = [self._new_group(agent) for agent in selected_agents]
        selected_ids: set[int] = set()
        for group in groups:
            while True:
                candidates = [task for task in available if task.task_id not in selected_ids and group.can_fit(task)]
                if not candidates:
                    break
                task = min(candidates, key=lambda item: (self._avg_home_distance(group, item, oracle), item.task_id))
                group.tasks.append(task)
                oracle.rebuild_group(group)
                selected_ids.add(task.task_id)
        return groups, selected_ids

    @staticmethod
    def _avg_home_distance(group: TaskGroup, task: Task, oracle: RouteCostOracle) -> float:
        return sum(oracle.grid.distance(group.home, target) for target in task.targets) / max(1, task.load)


class TokenPassingAllocator(BaseAllocator):
    name = "tp"

    def _select_agents(self, agents: list[Agent]) -> list[Agent]:
        eligible = [agent for agent in agents if len(agent.queue) < self.config.max_queue_groups]
        eligible.sort(key=lambda agent: (agent.active_end_time, agent.agent_id))
        return eligible[:1]

    def generate_groups(self, selected_agents: list[Agent], all_agents: list[Agent], backlog: list[Task], oracle: RouteCostOracle):
        available = self._candidate_tasks(all_agents, backlog)
        groups = [self._new_group(agent) for agent in selected_agents]
        selected_ids: set[int] = set()
        for group in groups:
            candidates = [task for task in available if task.task_id not in selected_ids and group.can_fit(task)]
            if not candidates:
                continue
            task = min(candidates, key=lambda item: (NearestGroupAllocator._avg_home_distance(group, item, oracle), item.task_id))
            group.tasks.append(task)
            oracle.rebuild_group(group)
            selected_ids.add(task.task_id)
        return groups, selected_ids


class RandomTSPAllocator(BaseAllocator):
    name = "tsp_mapd"

    def _select_agents(self, agents: list[Agent]) -> list[Agent]:
        eligible = [agent for agent in agents if len(agent.queue) < self.config.max_queue_groups]
        eligible.sort(key=lambda agent: (agent.active_end_time, agent.agent_id))
        return eligible[:1]

    def generate_groups(self, selected_agents: list[Agent], all_agents: list[Agent], backlog: list[Task], oracle: RouteCostOracle):
        available = self._candidate_tasks(all_agents, backlog)
        self.rng.shuffle(available)
        groups = [self._new_group(agent) for agent in selected_agents]
        selected_ids: set[int] = set()
        cursor = 0
        for group in groups:
            while cursor < len(available):
                task = available[cursor]
                cursor += 1
                if task.task_id in selected_ids or not group.can_fit(task):
                    continue
                group.tasks.append(task)
                selected_ids.add(task.task_id)
                if group.load >= group.capacity:
                    break
            oracle.rebuild_group(group)
        return groups, selected_ids


def make_allocator(method: str, config: AllocatorConfig, seed: int = 0) -> BaseAllocator:
    rng = random.Random(seed)
    method = method.lower()
    if method == "rgta_eff":
        return RegretGuidedAllocator(config, rng, insertion_mode="eff", name="rgta_eff")
    if method == "rgta_full":
        full = AllocatorConfig(**{**config.__dict__, "refine_after_insert": True})
        return RegretGuidedAllocator(full, rng, insertion_mode="full", name="rgta_full")
    if method == "marginal":
        marginal = AllocatorConfig(**{**config.__dict__, "beta": 0.0})
        return RegretGuidedAllocator(marginal, rng, insertion_mode="eff", name="marginal")
    if method == "sequential":
        return SequentialGreedyAllocator(config, rng)
    if method == "tp_tsp":
        return NearestGroupAllocator(config, rng)
    if method == "tp":
        return TokenPassingAllocator(config, rng)
    if method == "tsp_mapd":
        return RandomTSPAllocator(config, rng)
    raise ValueError(f"unknown allocator method: {method}")

