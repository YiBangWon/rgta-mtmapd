# Online Task-Group Assignment with Regret-Guided Selection for MT-MAPD

[![DOI](https://zenodo.org/badge/1289317741.svg)](https://zenodo.org/badge/latestdoi/1289317741)

> Code and experimental results for the paper "Online Task-Group Assignment
> with Regret-Guided Selection for Multi-Task Multi-Agent Pickup and Delivery"
> (under review).

This repository contains the code and experimental results for the paper
"Online Task-Group Assignment with Regret-Guided Selection for Multi-Task
Multi-Agent Pickup and Delivery" (under review).

## Overview

The code implements regret-guided task-group assignment for online multi-task
multi-agent pickup and delivery (MT-MAPD). It includes a standalone event-based
online simulator, task-allocation baselines, RGTA variants, ablation scripts,
stored CSV results, and a lightweight PIBT-based MAPF validation script.

The repository intentionally contains only the paper reproduction code. ROS 2
fleet-management modules and `fms_*` packages are not part of this repository.

## Repository Layout

```text
rgta-mtmapd/
+-- rgta/            # MT-MAPD simulator, maps, allocators, and TSP utilities
+-- experiments/     # Reproduction entry points
+-- tests/           # Core smoke/unit tests
+-- results/         # CSV results used for the paper tables
|   +-- ablation/    # Ablation and sensitivity CSV files
+-- README.md
+-- LICENSE
+-- CITATION.cff
+-- .gitignore
```

## Requirements

- Python >= 3.10
- Standard library only

No external Python packages are required.

## Setup

Clone the repository and run commands from the repository root:

```bash
git clone https://github.com/YiBangWon/rgta-mtmapd.git
cd rgta-mtmapd
python3 --version
```

## Reproducing Results

The commands below reproduce the experiments reported in the paper. Runtime
depends on the machine and method; the original runs were executed on an AMD
Ryzen Threadripper PRO 5995WX CPU server. Individual runs usually take from a
few seconds to tens of seconds, with `rgta_full` and PIBT validation taking
longer than the online-feasible `rgta_eff` variant.

### Tables 2-3: Main Results

```bash
python3 experiments/run_rgta_benchmark.py --maps kiva sorting \
  --settings 40x6 60x6 80x6 100x6 40x9 60x9 80x9 100x9 \
  --methods tp tsp_mapd tp_tsp rgta_eff rgta_full --seeds 0 1 2 3 4 --output results/main.csv
```

### Table 4: Ablation

Use the same maps, settings, and seeds as Tables 2-3, but evaluate sequential
greedy, marginal-cost selection, and RGTA:

```bash
python3 experiments/run_rgta_benchmark.py --maps kiva sorting \
  --settings 40x6 60x6 80x6 100x6 40x9 60x9 80x9 100x9 \
  --methods sequential marginal rgta_eff --seeds 0 1 2 3 4 --output results/ablation/table4.csv
```

### Table 5: Beta Sweep

Run `rgta_eff` with each beta value:

```bash
python3 experiments/run_rgta_benchmark.py --maps kiva sorting \
  --settings 40x6 60x6 80x6 100x6 40x9 60x9 80x9 100x9 \
  --methods rgta_eff --seeds 0 1 2 3 4 --beta 0.2 --output results/ablation/beta_0.2.csv
```

Repeat with `--beta 0.5`, `--beta 1.0`, and `--beta 2.0`.
Use `marginal` for beta = 0, and the default RGTA setting uses beta = 0.05.

### Linear Execution-Model Sensitivity

Add `--route-cost-exponent 1.0` to the main command:

```bash
python3 experiments/run_rgta_benchmark.py --maps kiva sorting \
  --settings 40x6 60x6 80x6 100x6 40x9 60x9 80x9 100x9 \
  --methods tp tsp_mapd tp_tsp rgta_eff rgta_full --seeds 0 1 2 3 4 \
  --route-cost-exponent 1.0 --output results/ablation/linear_execution_model.csv
```

### Clustered Task Distribution

Use the stress task profile:

```bash
python3 experiments/run_rgta_benchmark.py --maps kiva sorting \
  --settings 40x6 60x6 80x6 100x6 40x9 60x9 80x9 100x9 \
  --methods tp tsp_mapd tp_tsp rgta_eff rgta_full --seeds 0 1 2 3 4 \
  --task-profile rgta_stress --output results/ablation/clustered_distribution.csv
```

### Restricted Agent Subset

Use `--agent-subset-size 4` or `--agent-subset-size 8`:

```bash
python3 experiments/run_rgta_benchmark.py --maps kiva sorting \
  --settings 40x6 60x6 80x6 100x6 40x9 60x9 80x9 100x9 \
  --methods tp tsp_mapd tp_tsp rgta_eff rgta_full --seeds 0 1 2 3 4 \
  --agent-subset-size 4 --output results/ablation/subset_size_4.csv
```

### Table 6: PIBT Validation

```bash
python3 experiments/pibt_experiment.py --map kiva --agents 20 --capacity 6 \
  --methods tp_tsp rgta_eff --seeds 0 1 2 3 4 --output results/pibt_kiva_20x6.csv
```

The `--map sorting --agents 40` combination is run in the same way:

```bash
python3 experiments/pibt_experiment.py --map sorting --agents 40 --capacity 6 \
  --methods tp_tsp rgta_eff --seeds 0 1 2 3 4 --output results/pibt_sorting_40x6.csv
```

## Stored Results

The `results/` directory contains the CSV files used to check the paper tables:

- `ver3_server21_full16_summary.csv`
- `ver3_server21_full16_raw.csv`
- `ver3_rgtafull_server21_full16_summary.csv`
- `ver3_rgtafull_server21_full16_raw.csv`
- `ablation/*.csv`

Main result summary:

- `RGTA_eff` vs. `TP-TSP`: average service time -27.4%
  (range 14.1-38.0%), makespan -14.9% (range 9.4-19.8%), and allocation
  runtime no more than 4.17 ms/step.
- `RGTA_full`: service time -32.1% and makespan -17.8%.
- PIBT validation: service time reduction 9.5-22.2%, makespan reduction
  4.2-8.1%, better than the baseline in 20/20 runs, with 0 conflicts.
- Ablation: beta = 0 (`marginal`) and beta = 0.05 are comparable. The main
  improvement comes from cross-group marginal-cost evaluation, as discussed in
  Section 5.5 of the paper.

## Reproducibility Notes

- Each task contains three targets.
- Each run generates 2600 tasks, releases 600 initially, and releases the
  remaining tasks in batches of 10 every 5 timesteps.
- The main experiments use seeds 0, 1, 2, 3, and 4.
- Runtime values measure task-allocation wall-clock time only.
- The event-based simulator uses a route-cost surrogate for execution duration.
- The PIBT script performs an additional grid-based validation with per-timestep
  movement and conflict resolution.

## Citation

Please cite the paper as follows. The venue information will be updated after
publication.

```bibtex
@misc{son2026rgta,
  title  = {Online Task-Group Assignment with Regret-Guided Selection for Multi-Task
            Multi-Agent Pickup and Delivery},
  author = {Son, Hansol and Yoon, Soobin},
  year   = {2026},
  note   = {Under review}
}
```

## Authors

- Hansol Son ([YiBangWon](https://github.com/YiBangWon))
- Soobin Yoon ([yoonsoobinie](https://github.com/yoonsoobinie))

For questions, please contact Hansol Son.
