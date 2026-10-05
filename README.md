# track4_day4_minilab

## Generate the sample dataset and benchmark outputs

Run the benchmark from the repository root:

```powershell
python src\benchmark.py
```

The command creates a photo-style synthetic dashcam scene and degradation examples. These are generated images, not photographs captured by a real camera.

- `data\input\`: clean synthetic road-scene and night-scene images.
- `data\generated\`: images synthesized with blur, exposure, sensor-noise, and fog degradations.
- `outputs\`: benchmark results (`benchmark_results.csv`), charts, and the night-scene failure-case report.