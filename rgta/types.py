"""Shared data objects for MT-MAPD task assignment experiments."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Task:
    task_id: int
    targets: tuple[int, ...]
    release_time: int = 0
    priority: int = 0

    @property
    def load(self) -> int:
        return len(self.targets)


@dataclass
class TaskGroup:
    agent_id: int
    start: int
    home: int
    capacity: int
    tasks: list[Task] = field(default_factory=list)
    sequence: tuple[int, ...] = ()
    cost: float = 0.0

    @property
    def load(self) -> int:
        return sum(task.load for task in self.tasks)

    @property
    def task_ids(self) -> tuple[int, ...]:
        return tuple(task.task_id for task in self.tasks)

    def can_fit(self, task: Task) -> bool:
        return self.load + task.load <= self.capacity

    def all_targets(self) -> tuple[int, ...]:
        targets: list[int] = []
        for task in self.tasks:
            targets.extend(task.targets)
        return tuple(targets)


@dataclass
class Agent:
    agent_id: int
    home: int
    capacity: int
    queue: list[TaskGroup] = field(default_factory=list)
    active_group: TaskGroup | None = None
    active_end_time: float = 0.0
    completed_groups: int = 0

    def is_free(self, now: float) -> bool:
        return self.active_group is None and self.active_end_time <= now

    def queued_work(self) -> int:
        return len(self.queue) + (1 if self.active_group is not None else 0)

