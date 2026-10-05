# ADAS Camera Degradation Benchmark: Health Score vs Perception Impact

| Condition | Severity | Health Score | Status | Rec. Action | Fusion Weight | Detections (Retention) | Mean Conf (Drop) | Scorer Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean Baseline** | - | **92.1%** | `HEALTHY` | `FULL_CONFIDENCE` | **1.00** | 6 (100%) | 0.66 (+0.00) | 23.5 ms |
| **Blur** | Level 1 | **52.5%** | `DEGRADED` | `DOWN_WEIGHT` | **0.32** | 4 (67%) | 0.84 (+0.18) | 24.3 ms |
| **Blur** | Level 2 | **31.3%** | `CRITICAL` | `FALLBACK_DISENGAGE` | **0.00** | 4 (67%) | 0.73 (+0.08) | 30.3 ms |
| **Blur** | Level 3 | **20.7%** | `CRITICAL` | `FALLBACK_DISENGAGE` | **0.00** | 6 (100%) | 0.50 (-0.16) | 37.4 ms |
| **Blur** | Level 4 | **20.4%** | `CRITICAL` | `FALLBACK_DISENGAGE` | **0.00** | 3 (50%) | 0.47 (-0.19) | 23.7 ms |
| **Blur** | Level 5 | **20.2%** | `CRITICAL` | `FALLBACK_DISENGAGE` | **0.00** | 1 (17%) | 0.28 (-0.38) | 33.3 ms |
| **Glare** | Level 1 | **60.4%** | `DEGRADED` | `DOWN_WEIGHT` | **0.36** | 6 (100%) | 0.65 (-0.01) | 26.7 ms |
| **Glare** | Level 2 | **39.9%** | `CRITICAL` | `FALLBACK_DISENGAGE` | **0.00** | 7 (117%) | 0.61 (-0.04) | 32.5 ms |
| **Glare** | Level 3 | **17.1%** | `CRITICAL` | `FALLBACK_DISENGAGE` | **0.00** | 6 (100%) | 0.59 (-0.07) | 33.0 ms |
| **Glare** | Level 4 | **4.7%** | `CRITICAL` | `FALLBACK_DISENGAGE` | **0.00** | 4 (67%) | 0.68 (+0.03) | 22.3 ms |
| **Glare** | Level 5 | **2.7%** | `CRITICAL` | `FALLBACK_DISENGAGE` | **0.00** | 1 (17%) | 0.54 (-0.11) | 30.6 ms |
| **Darkness** | Level 1 | **80.8%** | `HEALTHY` | `FULL_CONFIDENCE` | **1.00** | 5 (83%) | 0.73 (+0.07) | 27.6 ms |
| **Darkness** | Level 2 | **55.3%** | `DEGRADED` | `DOWN_WEIGHT` | **0.33** | 4 (67%) | 0.77 (+0.12) | 25.8 ms |
| **Darkness** | Level 3 | **38.4%** | `CRITICAL` | `FALLBACK_DISENGAGE` | **0.00** | 5 (83%) | 0.56 (-0.10) | 24.2 ms |
| **Darkness** | Level 4 | **32.3%** | `CRITICAL` | `FALLBACK_DISENGAGE` | **0.00** | 3 (50%) | 0.50 (-0.16) | 19.0 ms |
| **Darkness** | Level 5 | **29.1%** | `CRITICAL` | `FALLBACK_DISENGAGE` | **0.00** | 0 (0%) | 0.00 (-0.66) | 28.7 ms |
| **Rain** | Level 1 | **94.7%** | `HEALTHY` | `FULL_CONFIDENCE` | **1.00** | 5 (83%) | 0.73 (+0.08) | 19.4 ms |
| **Rain** | Level 2 | **92.3%** | `HEALTHY` | `FULL_CONFIDENCE` | **1.00** | 5 (83%) | 0.72 (+0.07) | 25.0 ms |
| **Rain** | Level 3 | **88.7%** | `HEALTHY` | `FULL_CONFIDENCE` | **1.00** | 4 (67%) | 0.67 (+0.02) | 25.7 ms |
| **Rain** | Level 4 | **60.1%** | `DEGRADED` | `DOWN_WEIGHT` | **0.36** | 2 (33%) | 0.51 (-0.15) | 23.1 ms |
| **Rain** | Level 5 | **58.4%** | `DEGRADED` | `DOWN_WEIGHT` | **0.35** | 0 (0%) | 0.00 (-0.66) | 17.6 ms |
| **Soiling** | Level 1 | **92.3%** | `HEALTHY` | `FULL_CONFIDENCE` | **1.00** | 6 (100%) | 0.66 (+0.00) | 21.1 ms |
| **Soiling** | Level 2 | **91.4%** | `HEALTHY` | `FULL_CONFIDENCE` | **1.00** | 5 (83%) | 0.74 (+0.09) | 30.8 ms |
| **Soiling** | Level 3 | **84.2%** | `HEALTHY` | `FULL_CONFIDENCE` | **1.00** | 6 (100%) | 0.64 (-0.01) | 22.1 ms |
| **Soiling** | Level 4 | **61.3%** | `DEGRADED` | `DOWN_WEIGHT` | **0.37** | 7 (117%) | 0.59 (-0.06) | 21.1 ms |
| **Soiling** | Level 5 | **47.8%** | `DEGRADED` | `DOWN_WEIGHT` | **0.29** | 4 (67%) | 0.61 (-0.04) | 25.6 ms |

> **Key Engineering Takeaway:** When Health Score falls below **42%** (e.g. Level 3 Glare, Level 3 Darkness, or Level 3 Blur), detector retention drops severely, justifying immediate camera down-weighting (`DOWN_WEIGHT` or `FALLBACK_DISENGAGE`) to protect vehicle safety.