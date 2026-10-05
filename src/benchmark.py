"""Run the health-score benchmark and produce report artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .generate_degradation import apply_degradation, generate_dataset, standardize_frame
from .health_score import evaluate_frame


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _resolve_image_path(raw_path: str, project_root: Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else project_root / path


def _write_benchmark_csv(rows: list[dict], path: Path) -> None:
    fields = [
        "source", "scene_mode", "degradation", "severity", "image_path",
        "health_score", "status", "reasons", "sharpness", "exposure", "contrast", "noise",
        "laplacian_variance", "median_luminance", "contrast_span", "estimated_noise_sigma", "entropy_bits",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict]) -> dict:
    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["degradation"]][int(row["severity"])].append(float(row["health_score"]))
    summary: dict[str, dict] = {}
    for kind, levels in sorted(grouped.items()):
        means = [round(float(np.mean(levels[level])), 2) for level in sorted(levels)]
        severity = np.array(sorted(levels), dtype=np.float64)
        correlation = float(np.corrcoef(severity, np.array(means))[0, 1])
        violations = sum(means[index + 1] > means[index] + 1.0 for index in range(len(means) - 1))
        summary[kind] = {
            "mean_health_by_severity": means,
            "severity_health_correlation": round(correlation, 4),
            "monotonic_violations": violations,
            "expected_trend_pass": correlation <= -0.70 and violations <= 1,
        }
    return {
        "sample_count": len(rows),
        "thresholds": {"healthy_min": 70, "degraded_min": 40},
        "degradations": summary,
        "note": "Negative correlation means health decreases as corruption severity increases.",
    }


def _plot_trends(rows: list[dict], output_path: Path) -> None:
    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["degradation"]][int(row["severity"])].append(float(row["health_score"]))
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for kind, levels in sorted(grouped.items()):
        x = sorted(levels)
        y = [np.mean(levels[level]) for level in x]
        ax.plot(x, y, marker="o", linewidth=2.2, label=kind.replace("_", " ").title())
    ax.axhspan(70, 100, color="#55a868", alpha=0.08)
    ax.axhspan(40, 70, color="#ddaa33", alpha=0.08)
    ax.axhspan(0, 40, color="#c44e52", alpha=0.08)
    ax.axhline(70, color="#55a868", linestyle="--", linewidth=1)
    ax.axhline(40, color="#c44e52", linestyle="--", linewidth=1)
    ax.set(title="Camera health response to degradation severity", xlabel="Severity level", ylabel="Mean health score")
    ax.set_xticks(range(5))
    ax.set_ylim(0, 103)
    ax.grid(alpha=0.25)
    ax.legend(ncols=3, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _plot_comparison(rows: list[dict], project_root: Path, output_path: Path) -> None:
    source = next(row["source"] for row in rows if row["scene_mode"] == "day")
    kinds = ["blur", "underexposure", "overexposure", "noise", "fog"]
    levels = [0, 2, 4]
    fig, axes = plt.subplots(len(kinds), len(levels), figsize=(11, 12))
    for row_index, kind in enumerate(kinds):
        for col_index, level in enumerate(levels):
            row = next(
                item for item in rows
                if item["source"] == source and item["degradation"] == kind and int(item["severity"]) == level
            )
            image = cv2.imread(str(_resolve_image_path(row["image_path"], project_root)))
            axes[row_index, col_index].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[row_index, col_index].set_title(f"{kind} L{level} | health {float(row['health_score']):.1f}")
            axes[row_index, col_index].axis("off")
    fig.suptitle("Controlled camera degradation: clean → severe", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_local_glare_failure(input_dir: Path, output_path: Path) -> None:
    day_paths = sorted(
        path for path in input_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"} and "night" not in path.stem.lower()
    )
    if not day_paths:
        return
    source = cv2.imread(str(day_paths[0]))
    clean = standardize_frame(source)
    glare = apply_degradation(clean, "glare", 4, np.random.default_rng(404))
    clean_report = evaluate_frame(clean, "day")
    glare_report = evaluate_frame(glare, "day")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for axis, image, report, label in [
        (axes[0], clean, clean_report, "Clean real camera frame"),
        (axes[1], glare, glare_report, "Severe local glare"),
    ]:
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axis.set_title(f"{label}\nHealth {report.health_score:.1f} — {report.status}")
        axis.text(
            0.02, 0.03, f"Exposure: {report.components['exposure']:.2f} | Sharpness: {report.components['sharpness']:.2f}",
            transform=axis.transAxes, color="white", fontsize=10,
            bbox={"facecolor": "black", "alpha": 0.65, "pad": 4},
        )
        axis.axis("off")
    fig.suptitle("Failure case: global metrics under-react to spatially local glare")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def run_benchmark(
    input_dir: Path,
    generated_dir: Path,
    output_dir: Path,
    project_root: Path | None = None,
    synthetic_fallback: bool = False,
) -> dict:
    project_root = project_root or Path.cwd()
    manifest_path = generate_dataset(input_dir, generated_dir, synthetic_fallback)
    manifest = _read_manifest(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for item in manifest:
        image_path = _resolve_image_path(item["image_path"], project_root)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot read generated image: {image_path}")
        report = evaluate_frame(image, item["scene_mode"])
        results.append(
            {
                **item,
                "severity": int(item["severity"]),
                "health_score": report.health_score,
                "status": report.status,
                "reasons": "|".join(report.reasons),
                **report.components,
                **{key: report.raw_metrics[key] for key in [
                    "laplacian_variance", "median_luminance", "contrast_span", "estimated_noise_sigma", "entropy_bits"
                ]},
            }
        )
    _write_benchmark_csv(results, output_dir / "benchmark.csv")
    summary = _summarize(results)
    with (output_dir / "benchmark_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    _plot_trends(results, output_dir / "health_vs_severity.png")
    _plot_comparison(results, project_root, output_dir / "comparison_grid.png")
    _plot_local_glare_failure(input_dir, output_dir / "failure_case_local_glare.png")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the camera health score")
    parser.add_argument("--input-dir", type=Path, default=Path("data/real"))
    parser.add_argument("--generated-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--synthetic-fallback", action="store_true")
    args = parser.parse_args()
    summary = run_benchmark(
        args.input_dir, args.generated_dir, args.output_dir,
        synthetic_fallback=args.synthetic_fallback,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
