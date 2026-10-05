# ADAS Camera Degradation Health Score (Topic T1)

Repository for **Track 2 - Day 4 Minilab: Camera Degradation Health Score Benchmark** on branch `feat/gun`.

## Quick Start

### 1. Requirements
Ensure Python 3.10+ is installed with OpenCV, NumPy, Matplotlib, PyTorch, and Ultralytics:
```bash
pip install opencv-python numpy matplotlib scipy scikit-image torch ultralytics
```

### 2. Run Complete Benchmark
```bash
python -m src.benchmark
```
This script will:
- Run 5 corruption types (Blur, Glare, Darkness, Rain, Soiling) across 5 severity levels (1 to 5).
- Evaluate real-time Camera Health Scores (NR-IQA).
- Measure downstream object detection impact (YOLOv8n confidence drop and retention).
- Save all plots, comparison HUDs, and markdown tables to `outputs/`.

## Directory Structure
```
track4_day4_minilab/
├── data/
│   └── samples/                 # Real-world driving test scenes
├── src/
│   ├── degradation.py           # nuScenes-C / KITTI-C corruption generator
│   ├── health_score.py          # Real-time Camera Health Scorer (< 2.5ms)
│   ├── detector_eval.py         # YOLOv8n object detection evaluator + HUD overlay
│   └── benchmark.py             # Complete benchmark pipeline and visual generator
├── outputs/
│   ├── benchmark_table.md       # Full numerical results table
│   ├── benchmark_results.json   # Raw experiment metrics
│   ├── degradation_grid.png     # Visual grid of 10 degradation states
│   ├── hud_comparison.png       # Side-by-side ADAS HUD comparison (Clean vs Glare vs Soiling)
│   └── health_vs_confidence.png # 4-panel analytical plots
├── REPORT.md                    # Official 1-Page / 1-Slide Technical Report & Pitch Script
└── README.md
```

## Generated Artifacts
- **Detailed Report & Pitch Script:** [`REPORT.md`](REPORT.md)
- **Numerical Summary Table:** [`outputs/benchmark_table.md`](outputs/benchmark_table.md)
- **Multi-panel Plots:** `outputs/health_vs_confidence.png`
- **ADAS HUD Comparison:** `outputs/hud_comparison.png`