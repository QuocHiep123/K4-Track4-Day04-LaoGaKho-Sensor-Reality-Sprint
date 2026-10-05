"""Interactive Streamlit UI for the camera degradation health-score demo."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.generate_degradation import apply_degradation, standardize_frame
from src.health_score import HealthReport, evaluate_frame


ROOT = Path(__file__).resolve().parent
REAL_DATA_DIR = ROOT / "data" / "real"
OUTPUT_DIR = ROOT / "outputs"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

DEGRADATIONS = {
    "Không biến đổi": None,
    "Blur": "blur",
    "Thiếu sáng": "underexposure",
    "Cháy sáng": "overexposure",
    "Sensor noise": "noise",
    "Sương mù / low contrast": "fog",
    "Glare cục bộ (failure case)": "glare",
}

MODE_LABELS = {
    "Tự động": "auto",
    "Ban ngày": "day",
    "Ban đêm": "night",
}

REASON_LABELS = {
    "blur_or_low_texture": "Mờ hoặc cảnh quá ít texture",
    "underexposure": "Thiếu sáng",
    "overexposure": "Cháy sáng",
    "low_contrast": "Độ tương phản thấp",
    "high_noise": "Nhiễu cảm biến cao",
}

ACTION_LABELS = {
    "use_camera_normally": "Dùng camera bình thường",
    "down_weight_camera": "Giảm trọng số camera, ưu tiên LiDAR/radar",
    "fallback_or_safe_mode": "Fallback hoặc chuyển sang chế độ an toàn",
}


def _sample_paths() -> list[Path]:
    if not REAL_DATA_DIR.exists():
        return []
    return sorted(path for path in REAL_DATA_DIR.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def _read_upload(uploaded_file) -> np.ndarray | None:
    if uploaded_file is None:
        return None
    buffer = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def _rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _status_color(status: str) -> str:
    return {"HEALTHY": "#22c55e", "DEGRADED": "#f59e0b", "CRITICAL": "#ef4444"}[status]


def _report_card(report: HealthReport, title: str) -> None:
    color = _status_color(report.status)
    reasons = ", ".join(REASON_LABELS.get(item, item) for item in report.reasons) or "Không phát hiện lỗi rõ ràng"
    st.markdown(
        f"""
        <div class="health-card" style="border-top: 5px solid {color}">
          <div class="card-label">{title}</div>
          <div class="score" style="color:{color}">{report.health_score:.1f}</div>
          <div class="score-unit">/ 100 · {report.status}</div>
          <div class="card-detail"><b>Mode:</b> {report.scene_mode.upper()}</div>
          <div class="card-detail"><b>Nhận định:</b> {reasons}</div>
          <div class="card-detail"><b>Hành động:</b> {ACTION_LABELS[report.action]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _component_table(clean: HealthReport, degraded: HealthReport) -> pd.DataFrame:
    labels = {
        "sharpness": "Sharpness",
        "exposure": "Exposure",
        "contrast": "Contrast",
        "noise": "Noise quality",
    }
    return pd.DataFrame(
        {
            "Component": [labels[key] for key in labels],
            "Ảnh gốc": [clean.components[key] for key in labels],
            "Sau degradation": [degraded.components[key] for key in labels],
        }
    ).set_index("Component")


def _encode_png(image: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Không thể encode ảnh kết quả")
    return encoded.tobytes()


def _live_demo() -> None:
    st.subheader("Live camera health demo")
    st.caption("Chọn ảnh camera thật hoặc tải ảnh riêng, sau đó điều chỉnh degradation để quan sát health score.")

    samples = _sample_paths()
    source_col, settings_col = st.columns([1.15, 1], gap="large")
    with source_col:
        source_type = st.radio("Nguồn ảnh", ["Ảnh thật có sẵn", "Tải ảnh của bạn"], horizontal=True)
        uploaded = None
        selected_path: Path | None = None
        if source_type == "Ảnh thật có sẵn":
            if not samples:
                st.error("Chưa có ảnh trong data/real. Chạy: python -m src.fetch_real_data")
                return
            selected_name = st.selectbox("Camera frame", [path.name for path in samples])
            selected_path = next(path for path in samples if path.name == selected_name)
        else:
            uploaded = st.file_uploader("PNG/JPG", type=["png", "jpg", "jpeg", "bmp"])
            if uploaded is None:
                st.info("Hãy tải một ảnh camera để bắt đầu.")
                return

    with settings_col:
        mode_name = st.selectbox("Ngữ cảnh", list(MODE_LABELS), index=0)
        degradation_name = st.selectbox("Loại degradation", list(DEGRADATIONS), index=1)
        severity = st.slider("Mức độ", min_value=0, max_value=4, value=2, step=1)
        st.caption("L0 = clean, L4 = nghiêm trọng. Underexposure có thêm noise để mô phỏng low-light camera thực tế.")

    image = _read_upload(uploaded) if uploaded is not None else cv2.imread(str(selected_path), cv2.IMREAD_COLOR)
    if image is None:
        st.error("Không thể đọc ảnh đã chọn.")
        return

    clean = standardize_frame(image)
    degradation = DEGRADATIONS[degradation_name]
    if degradation is None or severity == 0:
        degraded = clean.copy()
    else:
        degraded = apply_degradation(clean, degradation, severity, np.random.default_rng(2026))
    mode = MODE_LABELS[mode_name]
    clean_report = evaluate_frame(clean, mode)
    degraded_report = evaluate_frame(degraded, mode)

    st.divider()
    image_left, image_right = st.columns(2, gap="large")
    with image_left:
        st.image(_rgb(clean), caption="Ảnh camera gốc · 640×360", width="stretch")
        _report_card(clean_report, "BASELINE")
    with image_right:
        st.image(_rgb(degraded), caption=f"{degradation_name} · severity L{severity}", width="stretch")
        _report_card(degraded_report, "CURRENT FRAME")

    delta = degraded_report.health_score - clean_report.health_score
    metric_cols = st.columns(4)
    metric_cols[0].metric("Health hiện tại", f"{degraded_report.health_score:.1f}", f"{delta:+.1f} vs clean")
    metric_cols[1].metric("Sharpness", f"{degraded_report.components['sharpness']:.2f}")
    metric_cols[2].metric("Exposure", f"{degraded_report.components['exposure']:.2f}")
    metric_cols[3].metric("Noise quality", f"{degraded_report.components['noise']:.2f}")

    st.markdown("#### So sánh quality components")
    component_data = _component_table(clean_report, degraded_report)
    st.bar_chart(component_data, horizontal=True, height=300)

    if degraded_report.status == "HEALTHY":
        st.success(f"Engineering decision: {ACTION_LABELS[degraded_report.action]}")
    elif degraded_report.status == "DEGRADED":
        st.warning(f"Engineering decision: {ACTION_LABELS[degraded_report.action]}")
    else:
        st.error(f"Engineering decision: {ACTION_LABELS[degraded_report.action]}")

    download_left, download_right = st.columns(2)
    report_payload = {
        "source": uploaded.name if uploaded is not None else selected_path.name,
        "degradation": degradation,
        "severity": severity,
        "baseline": clean_report.to_dict(),
        "result": degraded_report.to_dict(),
    }
    download_left.download_button(
        "Tải report JSON",
        json.dumps(report_payload, indent=2, ensure_ascii=False).encode("utf-8"),
        file_name="camera_health_report.json",
        mime="application/json",
        width="stretch",
    )
    download_right.download_button(
        "Tải ảnh degraded",
        _encode_png(degraded),
        file_name=f"{degradation or 'clean'}_level_{severity}.png",
        mime="image/png",
        width="stretch",
    )

    with st.expander("Raw metrics và JSON chi tiết"):
        raw_left, raw_right = st.columns(2)
        raw_left.markdown("**Ảnh gốc**")
        raw_left.json(clean_report.to_dict())
        raw_right.markdown("**Sau degradation**")
        raw_right.json(degraded_report.to_dict())


def _benchmark_tab() -> None:
    st.subheader("Benchmark trên ảnh camera thật")
    summary_path = OUTPUT_DIR / "benchmark_summary.json"
    chart_path = OUTPUT_DIR / "health_vs_severity.png"
    grid_path = OUTPUT_DIR / "comparison_grid.png"
    if not summary_path.exists():
        st.error("Chưa có benchmark. Chạy `python run_demo.py` trước.")
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    st.metric("Số mẫu đã đánh giá", summary["sample_count"])
    if chart_path.exists():
        st.image(str(chart_path), caption="Health score theo severity trên bốn cảnh thật", width="stretch")

    table_rows = []
    for name, values in summary["degradations"].items():
        table_rows.append(
            {
                "Corruption": name,
                "Health L0": values["mean_health_by_severity"][0],
                "Health L4": values["mean_health_by_severity"][-1],
                "Correlation": values["severity_health_correlation"],
                "Monotonic violations": values["monotonic_violations"],
                "Pass": values["expected_trend_pass"],
            }
        )
    st.dataframe(pd.DataFrame(table_rows), hide_index=True, width="stretch")
    if grid_path.exists():
        with st.expander("Xem toàn bộ ảnh clean → severe", expanded=False):
            st.image(str(grid_path), width="stretch")


def _failure_tab() -> None:
    st.subheader("Failure case cần trình bày")
    failure_path = OUTPUT_DIR / "failure_case_local_glare.png"
    if failure_path.exists():
        st.image(str(failure_path), width="stretch")
    st.warning(
        "Glare chỉ che một vùng nên global score vẫn có thể báo HEALTHY. "
        "Production system nên chia frame thành ROI/grid, dùng worst-region score và kết hợp detector confidence."
    )
    st.markdown(
        """
        **Trade-off kỹ thuật**

        - Global metric rất nhanh và ổn định nhưng bỏ sót lỗi cục bộ.
        - ROI metric nhạy hơn với lens dirt/glare nhưng cần biết vùng nào quan trọng.
        - Day/night inference từ histogram có thể nhầm cảnh đêm hợp lệ với underexposure.
        """
    )


def _method_tab() -> None:
    st.subheader("Pipeline và quyết định")
    st.markdown(
        """
        1. Chuẩn hóa frame về **640×360**.
        2. Tính **Variance of Laplacian**, exposure/clipping, percentile contrast và robust noise estimate.
        3. Chuẩn hóa bốn metric về `[0, 1]`.
        4. Gộp bằng weighted geometric mean thành health score `[0, 100]`.
        5. Ánh xạ sang quyết định sensor fusion.

        | Health | State | Decision |
        |---:|---|---|
        | 70–100 | HEALTHY | Dùng camera bình thường |
        | 40–69 | DEGRADED | Down-weight camera |
        | 0–39 | CRITICAL | Fallback / safe mode |

        Score này là baseline giải thích được cho mini-lab, chưa phải safety-certified production monitor.
        """
    )


def main() -> None:
    st.set_page_config(
        page_title="Camera Health Monitor",
        page_icon="📷",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        """
        <style>
          .block-container {max-width: 1250px; padding-top: 2rem; padding-bottom: 3rem;}
          .hero-kicker {color:#38bdf8; font-size:.82rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase;}
          .hero-title {font-size:2.35rem; font-weight:800; margin:.15rem 0 .2rem 0; line-height:1.1;}
          .hero-subtitle {color:#94a3b8; font-size:1.05rem; margin-bottom:1.3rem;}
          .health-card {background:linear-gradient(145deg,rgba(30,41,59,.92),rgba(15,23,42,.94)); padding:1.1rem 1.2rem; border-radius:14px; min-height:225px; box-shadow:0 8px 24px rgba(0,0,0,.16);}
          .card-label {color:#94a3b8; font-size:.76rem; font-weight:800; letter-spacing:.12em;}
          .score {display:inline-block; font-size:3.2rem; font-weight:850; line-height:1.15; margin-top:.25rem;}
          .score-unit {display:inline-block; color:#cbd5e1; margin-left:.4rem; font-weight:650;}
          .card-detail {color:#e2e8f0; margin-top:.48rem; font-size:.92rem;}
          [data-testid="stMetric"] {background:rgba(30,41,59,.45); border:1px solid rgba(148,163,184,.16); padding:.8rem 1rem; border-radius:12px;}
        </style>
        <div class="hero-kicker">Sensor Reality Sprint · T1</div>
        <div class="hero-title">Camera Degradation Health Monitor</div>
        <div class="hero-subtitle">Reference-free camera quality scoring for ADAS and robotic perception.</div>
        """,
        unsafe_allow_html=True,
    )

    live_tab, benchmark_tab, failure_tab, method_tab = st.tabs(
        ["🎛️ Live demo", "📊 Benchmark", "⚠️ Failure case", "🧠 Method"]
    )
    with live_tab:
        _live_demo()
    with benchmark_tab:
        _benchmark_tab()
    with failure_tab:
        _failure_tab()
    with method_tab:
        _method_tab()


if __name__ == "__main__":
    main()
