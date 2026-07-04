from __future__ import annotations

import unittest

from rgta.allocators import AllocatorConfig, make_allocator
from rgta.benchmarks import build_benchmark
from rgta.grid import make_kiva_map
from rgta.simulator import SimulationConfig, run_simulation
from rgta.tsp import RouteCostOracle
from rgta.types import TaskGroup


class RGTACoreTests(unittest.TestCase):
    def test_grid_distance(self) -> None:
        grid = make_kiva_map()
        self.assertEqual(grid.distance(grid.free_nodes[0], grid.free_nodes[0]), 0)

    def test_tsp_group_cost(self) -> None:
        grid = make_kiva_map()
        oracle = RouteCostOracle(grid)
        start = grid.home_nodes[0]
        targets = tuple(grid.pickup_nodes[:3])
        seq, cost = oracle.solve_full_tsp(targets, start, start)
        self.assertEqual(set(seq), set(targets))
        self.assertGreater(cost, 0)

    def test_smoke_simulation(self) -> None:
        grid, agents, tasks = build_benchmark("kiva", 4, 6, seed=0, total_tasks=30, initial_tasks=10)
        allocator = make_allocator("rgta_eff", AllocatorConfig(capacity=6, max_queue_groups=2), seed=0)
        result = run_simulation(grid, agents, tasks, allocator, 0, SimulationConfig(route_cost_exponent=1.0, max_queue_groups=2))
        self.assertEqual(result.completed_tasks, 30)


if __name__ == "__main__":
    unittest.main()

