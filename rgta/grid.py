"""Procedural Kiva and sorting-center grid maps with cached shortest paths."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

INF_DISTANCE = 10**9


@dataclass
class GridMap:
    name: str
    width: int
    height: int
    obstacles: set[int] = field(default_factory=set)
    pickup_nodes: list[int] = field(default_factory=list)
    home_nodes: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.obstacles = set(self.obstacles)
        self.free_nodes = [
            self.to_id(x, y)
            for y in range(self.height)
            for x in range(self.width)
            if self.to_id(x, y) not in self.obstacles
        ]
        if not self.pickup_nodes:
            self.pickup_nodes = _adjacent_free_cells(self)
        if not self.home_nodes:
            self.home_nodes = _perimeter_homes(self)
        self.pickup_nodes = [node for node in self.pickup_nodes if node not in self.obstacles]
        self.home_nodes = [node for node in self.home_nodes if node not in self.obstacles]
        if not self.pickup_nodes:
            self.pickup_nodes = list(self.free_nodes)
        if not self.home_nodes:
            self.home_nodes = list(self.free_nodes)
        self._distance_cache: dict[int, list[int]] = {}

    def to_id(self, x: int, y: int) -> int:
        return y * self.width + x

    def xy(self, node: int) -> tuple[int, int]:
        return node % self.width, node // self.width

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_free_xy(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and self.to_id(x, y) not in self.obstacles

    def neighbors(self, node: int) -> list[int]:
        x, y = self.xy(node)
        result: list[int] = []
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if self.is_free_xy(nx, ny):
                result.append(self.to_id(nx, ny))
        return result

    def distance(self, start: int, goal: int) -> int:
        if start == goal:
            return 0
        if start not in self._distance_cache:
            self._distance_cache[start] = self._bfs(start)
        return self._distance_cache[start][goal]

    def manhattan(self, a: int, b: int) -> int:
        ax, ay = self.xy(a)
        bx, by = self.xy(b)
        return abs(ax - bx) + abs(ay - by)

    def _bfs(self, start: int) -> list[int]:
        dist = [INF_DISTANCE] * (self.width * self.height)
        if start in self.obstacles:
            return dist
        dist[start] = 0
        queue: deque[int] = deque([start])
        while queue:
            node = queue.popleft()
            next_dist = dist[node] + 1
            for nbr in self.neighbors(node):
                if dist[nbr] == INF_DISTANCE:
                    dist[nbr] = next_dist
                    queue.append(nbr)
        return dist


def make_kiva_map() -> GridMap:
    width, height = 48, 36
    obstacles: set[int] = set()
    for x0 in range(6, width - 7, 6):
        for y0 in range(4, height - 7, 6):
            for dx in range(3):
                for dy in range(4):
                    obstacles.add(y0 * width + x0 + dx)
    return GridMap("kiva", width, height, obstacles)


def make_sorting_map() -> GridMap:
    width, height = 77, 37
    obstacles: set[int] = set()
    horizontal_aisles = {8, 18, 28}
    for x0 in range(10, width - 10, 8):
        for y in range(4, height - 4):
            if any(abs(y - aisle) <= 1 for aisle in horizontal_aisles):
                continue
            obstacles.add(y * width + x0)
            obstacles.add(y * width + x0 + 1)
    for x in range(30, 47):
        for y in range(15, 22):
            if y not in horizontal_aisles:
                obstacles.add(y * width + x)
    return GridMap("sorting", width, height, obstacles)


def _adjacent_free_cells(grid: GridMap) -> list[int]:
    pickups: set[int] = set()
    for obs in grid.obstacles:
        x, y = grid.xy(obs)
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if grid.is_free_xy(nx, ny):
                pickups.add(grid.to_id(nx, ny))
    return sorted(pickups)


def _perimeter_homes(grid: GridMap) -> list[int]:
    homes: list[int] = []
    for y in range(1, grid.height - 1):
        for x in (1, grid.width - 2):
            if grid.is_free_xy(x, y):
                homes.append(grid.to_id(x, y))
    for x in range(1, grid.width - 1):
        for y in (1, grid.height - 2):
            node = grid.to_id(x, y)
            if grid.is_free_xy(x, y) and node not in homes:
                homes.append(node)
    return homes

