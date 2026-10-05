"""
Benchmark runner:
1. Load or generate base images.
2. Sweep severities for Blur, Underexposure, Overexposure, Noise, Fog.
3. Record measurements, status, action, and health scores to CSV.
4. Produce visualization charts and before/after comparison grids.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# Local imports
from health_score import evaluate_camera_health
from generate_degradation import (
    apply_blur,
    apply_exposure,
    apply_noise,
    apply_fog_low_contrast,
    create_synthetic_road_scene
)


def run_benchmark(output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    input_dir = os.path.join("data", "input")
    generated_dir = os.path.join("data", "generated")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(generated_dir, exist_ok=True)

    # 1. Base clean image
    base_img = create_synthetic_road_scene()
    save_image(os.path.join(input_dir, "base_clean_image.png"), base_img)
    save_image(os.path.join(output_dir, "base_clean_image.png"), base_img)

    records = []

    # Nominal check
    res_clean = evaluate_camera_health(base_img)
    records.append({
        "degradation_type": "nominal",
        "severity_param": 0.0,
        "severity_label": "clean",
        "health_score": res_clean["health_score"],
        "status": res_clean["status"],
        "action": res_clean["action"],
        "q_sharpness": res_clean["components"]["q_sharpness"],
        "q_exposure": res_clean["components"]["q_exposure"],
        "q_contrast": res_clean["components"]["q_contrast"],
        "q_noise": res_clean["components"]["q_noise"],
        "reasons": ";".join(res_clean["reasons"])
    })

    # Sweep 1: Blur (sigma 0 to 6)
    blur_sigmas = [0.5, 1.0, 2.0, 3.5, 5.5]
    for s in blur_sigmas:
        corrupted = apply_blur(base_img, s)
        severity_label = f"sigma_{s}"
        save_image(os.path.join(generated_dir, f"blur_{severity_label}.png"), corrupted)
        eval_res = evaluate_camera_health(corrupted)
        records.append({
            "degradation_type": "blur",
            "severity_param": s,
            "severity_label": severity_label,
            "health_score": eval_res["health_score"],
            "status": eval_res["status"],
            "action": eval_res["action"],
            "q_sharpness": eval_res["components"]["q_sharpness"],
            "q_exposure": eval_res["components"]["q_exposure"],
            "q_contrast": eval_res["components"]["q_contrast"],
            "q_noise": eval_res["components"]["q_noise"],
            "reasons": ";".join(eval_res["reasons"])
        })
        if s in [1.0, 3.5, 5.5]:
            cv2.imwrite(os.path.join(output_dir, f"degraded_blur_{s}.png"), corrupted)

    # Sweep 2: Underexposure (factor 0.85 to 0.15)
    under_factors = [0.85, 0.65, 0.45, 0.25, 0.12]
    for f in under_factors:
        corrupted = apply_exposure(base_img, f)
        severity_label = f"gain_{f}"
        save_image(os.path.join(generated_dir, f"underexposure_{severity_label}.png"), corrupted)
        eval_res = evaluate_camera_health(corrupted)
        records.append({
            "degradation_type": "underexposure",
            "severity_param": f,
            "severity_label": severity_label,
            "health_score": eval_res["health_score"],
            "status": eval_res["status"],
            "action": eval_res["action"],
            "q_sharpness": eval_res["components"]["q_sharpness"],
            "q_exposure": eval_res["components"]["q_exposure"],
            "q_contrast": eval_res["components"]["q_contrast"],
            "q_noise": eval_res["components"]["q_noise"],
            "reasons": ";".join(eval_res["reasons"])
        })
        if f in [0.65, 0.45, 0.12]:
            cv2.imwrite(os.path.join(output_dir, f"degraded_underexp_{f}.png"), corrupted)

    # Sweep 3: Overexposure / Glare (factor 1.2 to 2.5)
    over_factors = [1.2, 1.5, 1.8, 2.2, 2.8]
    for f in over_factors:
        corrupted = apply_exposure(base_img, f)
        severity_label = f"gain_{f}"
        save_image(os.path.join(generated_dir, f"overexposure_{severity_label}.png"), corrupted)
        eval_res = evaluate_camera_health(corrupted)
        records.append({
            "degradation_type": "overexposure",
            "severity_param": f,
            "severity_label": severity_label,
            "health_score": eval_res["health_score"],
            "status": eval_res["status"],
            "action": eval_res["action"],
            "q_sharpness": eval_res["components"]["q_sharpness"],
            "q_exposure": eval_res["components"]["q_exposure"],
            "q_contrast": eval_res["components"]["q_contrast"],
            "q_noise": eval_res["components"]["q_noise"],
            "reasons": ";".join(eval_res["reasons"])
        })
        if f in [1.5, 2.2]:
            cv2.imwrite(os.path.join(output_dir, f"degraded_overexp_{f}.png"), corrupted)

    # Sweep 4: Sensor Noise (sigma 5 to 40)
    noise_sigmas = [5, 12, 22, 35, 50]
    for n in noise_sigmas:
        corrupted = apply_noise(base_img, n)
        severity_label = f"noise_{n}"
        save_image(os.path.join(generated_dir, f"sensor_noise_{severity_label}.png"), corrupted)
        eval_res = evaluate_camera_health(corrupted)
        records.append({
            "degradation_type": "sensor_noise",
            "severity_param": n,
            "severity_label": severity_label,
            "health_score": eval_res["health_score"],
            "status": eval_res["status"],
            "action": eval_res["action"],
            "q_sharpness": eval_res["components"]["q_sharpness"],
            "q_exposure": eval_res["components"]["q_exposure"],
            "q_contrast": eval_res["components"]["q_contrast"],
            "q_noise": eval_res["components"]["q_noise"],
            "reasons": ";".join(eval_res["reasons"])
        })
        if n in [12, 35]:
            cv2.imwrite(os.path.join(output_dir, f"degraded_noise_{n}.png"), corrupted)

    # Sweep 5: Fog / low contrast
    fog_levels = [0.2, 0.4, 0.6, 0.8]
    for fg in fog_levels:
        corrupted = apply_fog_low_contrast(base_img, fg)
        severity_label = f"fog_{fg}"
        save_image(os.path.join(generated_dir, f"fog_{severity_label}.png"), corrupted)
        eval_res = evaluate_camera_health(corrupted)
        records.append({
            "degradation_type": "fog",
            "severity_param": fg,
            "severity_label": severity_label,
            "health_score": eval_res["health_score"],
            "status": eval_res["status"],
            "action": eval_res["action"],
            "q_sharpness": eval_res["components"]["q_sharpness"],
            "q_exposure": eval_res["components"]["q_exposure"],
            "q_contrast": eval_res["components"]["q_contrast"],
            "q_noise": eval_res["components"]["q_noise"],
            "reasons": ";".join(eval_res["reasons"])
        })

    # Save DataFrame
    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"[OK] Saved benchmark records to {csv_path}")

    # Generate Visualization Plot
    plot_benchmark_results(df, output_dir)
    generate_failure_case_demo(output_dir)

    return df


def save_image(path, image):
    """Write an image and fail explicitly if OpenCV cannot save it."""
    if not cv2.imwrite(path, image):
        raise OSError(f"Could not write image to {path}")


def plot_benchmark_results(df, output_dir):
    """Plot Health Score vs Severity for the major degradations."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.patch.set_facecolor('#ffffff')

    # Color palette matching clean presentation guidelines
    col_blur = '#2563eb'
    col_under = '#d97706'
    col_noise = '#dc2626'
    col_over = '#9333ea'

    # 1. Blur
    df_blur = df[df["degradation_type"] == "blur"]
    ax1 = axes[0, 0]
    ax1.plot(df_blur["severity_param"], df_blur["health_score"], marker='o', color=col_blur, linewidth=2, label="Total Health")
    ax1.plot(df_blur["severity_param"], df_blur["q_sharpness"] * 100, marker='s', linestyle='--', color='#60a5fa', label="Sharpness Comp.")
    ax1.axhline(70, color='#16a34a', linestyle=':', label='Healthy (70)')
    ax1.axhline(40, color='#ef4444', linestyle=':', label='Critical (40)')
    ax1.set_title("Gaussian Blur Severity vs Health", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Blur Kernel Sigma (px)")
    ax1.set_ylabel("Score (0-100)")
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower left', fontsize=8)

    # 2. Underexposure
    df_under = df[df["degradation_type"] == "underexposure"]
    ax2 = axes[0, 1]
    ax2.plot(df_under["severity_param"], df_under["health_score"], marker='o', color=col_under, linewidth=2, label="Total Health")
    ax2.plot(df_under["severity_param"], df_under["q_exposure"] * 100, marker='s', linestyle='--', color='#fbbf24', label="Exposure Comp.")
    ax2.axhline(70, color='#16a34a', linestyle=':')
    ax2.axhline(40, color='#ef4444', linestyle=':')
    ax2.set_title("Underexposure vs Health", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Brightness Gain Factor (1.0 = normal)")
    ax2.set_ylabel("Score (0-100)")
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right', fontsize=8)

    # 3. Sensor Noise
    df_noise = df[df["degradation_type"] == "sensor_noise"]
    ax3 = axes[1, 0]
    ax3.plot(df_noise["severity_param"], df_noise["health_score"], marker='o', color=col_noise, linewidth=2, label="Total Health")
    ax3.plot(df_noise["severity_param"], df_noise["q_noise"] * 100, marker='^', linestyle='--', color='#f87171', label="Noise Comp.")
    ax3.axhline(70, color='#16a34a', linestyle=':')
    ax3.axhline(40, color='#ef4444', linestyle=':')
    ax3.set_title("Sensor Noise Severity vs Health", fontsize=11, fontweight='bold')
    ax3.set_xlabel("Noise StdDev (Gaussian sigma)")
    ax3.set_ylabel("Score (0-100)")
    ax3.set_ylim(0, 105)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='lower left', fontsize=8)

    # 4. Overexposure / Glare
    df_over = df[df["degradation_type"] == "overexposure"]
    ax4 = axes[1, 1]
    ax4.plot(df_over["severity_param"], df_over["health_score"], marker='o', color=col_over, linewidth=2, label="Total Health")
    ax4.plot(df_over["severity_param"], df_over["q_exposure"] * 100, marker='s', linestyle='--', color='#c084fc', label="Exposure Comp.")
    ax4.axhline(70, color='#16a34a', linestyle=':')
    ax4.axhline(40, color='#ef4444', linestyle=':')
    ax4.set_title("Overexposure / Glare vs Health", fontsize=11, fontweight='bold')
    ax4.set_xlabel("Exposure Gain Multiplier")
    ax4.set_ylabel("Score (0-100)")
    ax4.set_ylim(0, 105)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='lower left', fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "health_vs_severity.png")
    plt.savefig(plot_path, dpi=160)
    plt.close()
    print(f"[OK] Saved plot to {plot_path}")


def generate_failure_case_demo(output_dir):
    """
    Demonstrate classic real-world failure case:
    Legitimate Night-time road scene is falsely penalized as 'severe_underexposure' / low health
    because global histogram is dark, even though road headlights and taillights are clearly visible!
    """
    h, w = 480, 640
    night_img = np.zeros((h, w, 3), dtype=np.uint8)
    # Pitch dark sky and surroundings
    night_img[0:220, :] = [10, 10, 15]
    night_img[220:, :] = [15, 15, 20]

    # Headlight illumination cone on road
    pts_cone = np.array([[240, 480], [400, 480], [340, 310], [300, 310]], np.int32)
    cv2.fillPoly(night_img, [pts_cone], (120, 120, 140))

    # Reflector lane dashes illuminated
    for y in range(320, 470, 50):
        cv2.line(night_img, (320, y), (320, y + 20), (240, 240, 255), 4)

    # Lead car tail lights glowing bright red
    cv2.circle(night_img, (300, 320), 8, (0, 0, 255), -1)
    cv2.circle(night_img, (340, 320), 8, (0, 0, 255), -1)

    eval_night = evaluate_camera_health(night_img)

    input_dir = os.path.join("data", "input")
    os.makedirs(input_dir, exist_ok=True)
    save_image(os.path.join(input_dir, "failure_case_night_scene.png"), night_img)
    save_image(os.path.join(output_dir, "failure_case_night_scene.png"), night_img)

    fail_report = (
        f"--- FAILURE CASE DEMO REPORT ---\n"
        f"Scene: Valid Night Driving (Lit Road & Taillights)\n"
        f"Computed Health Score: {eval_night['health_score']}/100\n"
        f"Status Assigned: {eval_night['status']}\n"
        f"Reasons: {eval_night['reasons']}\n"
        f"Underexposed Pixel Ratio: {eval_night['raw_measurements']['underexposed_ratio'] * 100:.1f}%\n"
        f"Root Cause: Global histogram assumes daylight baseline. 80%+ pixels are legitimately dark at night.\n"
        f"Recommended Fix: Adaptive Day/Night thresholding via camera ambient illuminance or exposure metadata.\n"
    )

    with open(os.path.join(output_dir, "failure_case_report.txt"), "w", encoding="utf-8") as f:
        f.write(fail_report)
    print(f"[OK] Failure case generated and saved.")


if __name__ == "__main__":
    run_benchmark()
