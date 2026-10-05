"""One-command entry point for the complete T1 mini-lab."""

from __future__ import annotations

import json
from pathlib import Path

from src.benchmark import run_benchmark


def main() -> None:
    root = Path(__file__).resolve().parent
    summary = run_benchmark(
        input_dir=root / "data" / "real",
        generated_dir=root / "data" / "generated",
        output_dir=root / "outputs",
        project_root=root,
    )
    print("\nCamera degradation benchmark completed.")
    print(f"Samples evaluated: {summary['sample_count']}")
    for kind, result in summary["degradations"].items():
        scores = " -> ".join(f"{score:.1f}" for score in result["mean_health_by_severity"])
        print(f"  {kind:14s}: {scores} (corr={result['severity_health_correlation']:+.3f})")
    print("\nArtifacts: outputs/benchmark.csv, outputs/benchmark_summary.json,")
    print("           outputs/health_vs_severity.png, outputs/comparison_grid.png,")
    print("           outputs/failure_case_local_glare.png")
    print("\nFull summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
