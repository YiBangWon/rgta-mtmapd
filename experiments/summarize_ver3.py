from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

PAPER_REDUCTION = {
    ("kiva", 60, 9): 27.5,
    ("kiva", 100, 9): 35.2,
    ("sorting", 60, 9): 20.3,
    ("sorting", 100, 9): 24.9,
}


def main() -> int:
    raw = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/ver3_full_raw.csv")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else raw.with_name(raw.stem.replace("_raw", "") + "_summary.csv")
    rows = list(csv.DictReader(raw.open()))
    groups: dict[tuple[str, int, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["map_name"], int(row["num_agents"]), int(row["capacity"]), row["method"])].append(row)

    summary: list[dict[str, str]] = []
    methods = sorted({key[3] for key in groups})
    for map_name in ["kiva", "sorting"]:
        for agents in [40, 60, 80, 100]:
            for capacity in [6, 9]:
                base = groups.get((map_name, agents, capacity, "tp_tsp"), [])
                if not base:
                    continue
                base_service = mean(float(row["average_service_time"]) for row in base)
                base_makespan = mean(float(row["makespan"]) for row in base)
                for method in methods:
                    values = groups.get((map_name, agents, capacity, method), [])
                    if not values:
                        continue
                    service = mean(float(row["average_service_time"]) for row in values)
                    makespan = mean(float(row["makespan"]) for row in values)
                    service_reduction = (base_service - service) / base_service * 100.0
                    makespan_reduction = (base_makespan - makespan) / base_makespan * 100.0
                    key = (map_name, agents, capacity)
                    paper = PAPER_REDUCTION.get(key)
                    threshold = None if paper is None else paper * 0.93
                    summary.append(
                        {
                            "map": map_name,
                            "agents": str(agents),
                            "capacity": str(capacity),
                            "method": method,
                            "service": f"{service:.6f}",
                            "makespan": f"{makespan:.6f}",
                            "tp_tsp_service": f"{base_service:.6f}",
                            "tp_tsp_makespan": f"{base_makespan:.6f}",
                            "service_reduction_pct": f"{service_reduction:.6f}",
                            "makespan_reduction_pct": f"{makespan_reduction:.6f}",
                            "paper_reduction_pct": "" if paper is None else f"{paper:.6f}",
                            "threshold_93pct": "" if threshold is None else f"{threshold:.6f}",
                            "pass_93pct": "" if threshold is None else str(service_reduction + 1e-9 >= threshold),
                            "makespan_improved": str(makespan < base_makespan),
                        }
                    )

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        fields = list(summary[0].keys()) if summary else []
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)

    print(f"wrote {out}")
    print("RGTA paper 93% checkpoints:")
    ok = True
    for row in summary:
        if row["method"] != "rgta_eff" or not row["threshold_93pct"]:
            continue
        passed = row["pass_93pct"] == "True" and row["makespan_improved"] == "True"
        ok = ok and passed
        print(
            f"{row['map']}({row['agents']},{row['capacity']}): "
            f"service_red={float(row['service_reduction_pct']):.3f}% "
            f"threshold={float(row['threshold_93pct']):.3f}% "
            f"makespan_red={float(row['makespan_reduction_pct']):.3f}% "
            f"pass={passed}"
        )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

