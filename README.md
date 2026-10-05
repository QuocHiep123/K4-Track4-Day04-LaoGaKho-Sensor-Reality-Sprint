# T1 — Camera Degradation Health Score

Mini-lab đánh giá **độ tin cậy của ảnh camera đối với perception** trong ADAS/robot. Benchmark mặc định dùng bốn ảnh đường phố thật: ba frame camera xe từ KITTI và một cảnh đường ban đêm CC0. Trên các frame thật này, pipeline sinh nhiều mức blur/underexposure/overexposure/noise/fog, chấm health score và xuất báo cáo trực quan.

## Kết quả đầu ra

Mỗi ảnh nhận được:

- `health_score` từ 0–100;
- trạng thái `HEALTHY`, `DEGRADED` hoặc `CRITICAL`;
- điểm thành phần cho sharpness, exposure, contrast và noise;
- nguyên nhân suy giảm và hành động gợi ý.

Ví dụ:

```json
{
  "health_score": 62.4,
  "status": "DEGRADED",
  "reasons": ["blur", "underexposure"],
  "action": "down_weight_camera"
}
```

## Cài đặt và chạy

Yêu cầu Python 3.10+.

```powershell
python -m pip install -r requirements.txt
python -m src.fetch_real_data
python run_demo.py
python -m unittest discover -s tests -v
```

Khởi động UI demo:

```powershell
python -m streamlit run app.py
```

Mở địa chỉ hiển thị trong terminal, mặc định là `http://localhost:8501`. UI hỗ trợ chọn bốn ảnh thật có sẵn hoặc upload ảnh riêng, tạo degradation trực tiếp, so sánh health score trước/sau, tải JSON kết quả và xem toàn bộ benchmark/failure case.

Chấm một ảnh riêng:

```powershell
python -m src.health_score path\to\image.jpg --mode auto
```

`--mode` nhận `auto`, `day` hoặc `night`. Nếu biết ngữ cảnh từ camera/xe, nên truyền trực tiếp `day` hoặc `night`; chế độ `auto` chỉ dùng histogram nên vẫn có thể nhầm ảnh ban ngày bị thiếu sáng thành cảnh đêm.

## Cấu trúc

```text
.
├── data/
│   ├── real/                  # 4 ảnh đường phố thật + thông tin nguồn/license
│   ├── input/                 # dữ liệu mô phỏng cũ, không dùng trong benchmark chính
│   └── generated/             # ảnh corruption và manifest.csv
├── outputs/
│   ├── benchmark.csv
│   ├── benchmark_summary.json
│   ├── health_vs_severity.png
│   ├── comparison_grid.png
│   └── failure_case_local_glare.png
├── src/
│   ├── health_score.py        # metric và quyết định
│   ├── fetch_real_data.py     # tải bộ ảnh thật có attribution
│   ├── generate_degradation.py
│   └── benchmark.py
├── tests/
├── app.py                     # Streamlit demo UI
└── run_demo.py
```

Các ảnh thật đã được đặt trong `data/real/`. Lệnh `fetch_real_data` cho phép tải lại chúng khi cần; chi tiết nguồn, giấy phép và citation nằm tại `data/real/SOURCES.md`.

## Phương pháp

Ảnh được đổi sang grayscale và tính bốn quality component trong khoảng `[0, 1]`:

| Component | Metric | Vai trò |
|---|---|---|
| Sharpness | Variance of Laplacian sau denoise nhẹ | Phát hiện defocus/motion blur |
| Exposure | Median luminance và tỷ lệ pixel bị clip | Phát hiện quá tối/cháy sáng |
| Contrast | Khoảng percentile `P95 - P5` | Phát hiện ảnh phẳng, sương mờ |
| Noise | Robust noise estimate bằng high-pass kernel | Phát hiện sensor noise |

Entropy được báo cáo để phân tích nhưng không đưa vào health score vì noise có thể làm entropy tăng dù chất lượng perception giảm.

Health score dùng weighted geometric mean:

```text
health = 100 × exp(
    0.40 log(q_sharpness)
  + 0.30 log(q_exposure)
  + 0.15 log(q_contrast)
  + 0.15 log(q_noise)
)
```

Geometric mean phạt mạnh một thành phần rất xấu, tránh trường hợp ảnh cực mờ vẫn được các metric khác “bù điểm”. Các ngưỡng hiện tại được hiệu chỉnh cho demo 640×360; camera thật cần calibration lại theo sensor, resolution và pipeline ISP.

Quyết định mặc định:

| Score | Trạng thái | Hành động |
|---:|---|---|
| 70–100 | `HEALTHY` | Dùng camera bình thường |
| 40–69 | `DEGRADED` | Cảnh báo hoặc giảm trọng số camera |
| 0–39 | `CRITICAL` | Fallback sang cảm biến khác/chế độ an toàn |

Trong video nên làm mượt score qua nhiều frame bằng `TemporalHealthSmoother` để tránh trạng thái nhấp nháy.

## Benchmark

Demo dùng bốn cảnh thật (`kitti_um_000032`, `kitti_umm_000005`, `kitti_uu_000010`, `night_wikimedia_cc0`) và năm mức cho từng corruption:

- Gaussian blur;
- underexposure;
- overexposure;
- Gaussian noise;
- fog/low contrast.

`benchmark_summary.json` ghi mean score theo severity, tương quan severity–score và số vi phạm xu hướng đơn điệu. `health_vs_severity.png` là biểu đồ chính để đưa lên slide; `comparison_grid.png` cung cấp ảnh trước/sau.

Mỗi ảnh nguồn được resize-to-cover và center-crop về 640×360 trước khi tạo degradation để metric giữa các camera có thể so sánh được. Tổng cộng benchmark tạo `4 × 5 × 5 = 100` mẫu. Glare vẫn được ghi nhận là limitation/failure case vì metric toàn ảnh khó phân biệt glare cục bộ với vùng trời sáng hợp lệ.

Kết quả hiện tại trên ảnh thật:

| Corruption | Mean health L0 → L4 | Tương quan severity–health |
|---|---|---:|
| Blur | 94.0 → 20.9 | -0.962 |
| Fog | 94.0 → 22.1 | -0.888 |
| Noise | 94.0 → 43.2 | -0.910 |
| Overexposure | 94.0 → 35.3 | -0.984 |
| Underexposure | 94.0 → 64.3 | -0.970 |

Cả năm corruption đều giảm đơn điệu qua các mức severity và vượt điều kiện kiểm tra xu hướng của benchmark.

Bạn có thể thay hoặc bổ sung ảnh thật vào `data/real/` rồi chạy lại. Tên ảnh chứa `night` sẽ dùng ngưỡng ban đêm, các ảnh còn lại dùng ngưỡng ban ngày. Nếu dùng toàn bộ KITTI hoặc một dataset khác, hãy giữ nguyên attribution/license tương ứng.

## Failure case và limitation

Failure case chính nằm trong `failure_case_local_glare.png`: glare mạnh che một vùng quan trọng nhưng các metric toàn ảnh chỉ thay đổi ít, vì phần lớn frame vẫn sắc nét và có exposure hợp lệ. Cải tiến phù hợp là chia ảnh thành grid/ROI, lấy worst-region score hoặc ưu tiên vùng detector/đường phía trước.

Day/night vẫn là một limitation: histogram không thể phân biệt chắc chắn “cảnh đêm hợp lệ” với “camera ban ngày bị tối”. Trong hệ thống thật nên dùng ambient light, exposure time và sensor gain thay cho suy đoán chỉ từ ảnh.

Các giới hạn khác:

- cảnh ít texture có thể bị nhầm thành blur;
- edge/noise mạnh có thể làm sharpness metric tăng;
- metric toàn ảnh có thể bỏ sót lens bẩn hoặc glare chỉ che vùng quan trọng;
- threshold phụ thuộc camera và độ phân giải;
- health score không thay thế kiểm chứng bằng detector downstream.

## Nội dung pitch 3–5 phút

1. **Problem:** camera vẫn trả frame nhưng blur/exposure/noise làm perception mất tin cậy.
2. **Method:** bốn metric reference-free, giải thích được, chạy CPU; kết hợp thành score 0–100.
3. **Benchmark:** năm corruption × năm severity × bốn cảnh thật; trình bày `health_vs_severity.png` và `comparison_grid.png`.
4. **Failure:** global score bỏ sót glare cục bộ; trình bày `failure_case_local_glare.png` và đề xuất ROI/grid score.
5. **Decision:** dưới 70 down-weight, dưới 40 fallback; làm mượt theo thời gian và calibration theo camera.

## Tuyên bố kỹ thuật

Giải pháp phù hợp làm health monitor nhẹ trên edge device vì không cần ground-truth/reference frame và mọi quyết định đều truy vết được về metric thành phần. Đây là baseline để benchmark và thảo luận trade-off, chưa phải safety-certified production monitor.
