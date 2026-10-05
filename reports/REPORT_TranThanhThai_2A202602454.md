# BẢN BÁO CÁO CÁ NHÂN — LAB DAY 04 (VLEARN SUBMISSION)
## Topic T1: Camera Degradation Health Score & Sensor Integrity Monitor

* **Họ và tên sinh viên:** Trần Thành Thái
* **Mã số sinh viên (MSSV):** 2A202602454
* **Nhóm:** LaoGaKho (Lão Gà Kho)
* **Vai trò trong nhóm:** Benchmark & Downstream Perception Evaluator
* **Repository nhóm:** https://github.com/QuocHiep123/track4_day4_minilab (Branch: `TranThanhThai-2A202602454`)
* **Danh sách thành viên:** Đối chiếu tại [`TEAMMATES.md`](../TEAMMATES.md)

---

### 1. Problem (Vấn đề & Bối cảnh kỹ thuật)
* **Platform & Cảm biến:** Xe tự hành ADAS Cấp độ L2+/L3 (Front Windshield Camera 1080p).
* **Failure thực tế:** Khi camera bị suy giảm chất lượng (Chói nắng, Mờ, Nhiễu tối, Bùn bẩn), các mô hình nhận diện vật thể hạ tầng (YOLOv8n) bị sụt giảm nghiêm trọng cả về số lượng vật thể nhận diện (Retention) lẫn độ tự tin trung bình (Confidence Score).
* **Mục tiêu cá nhân:** Đo lường chính xác mối tương quan giữa điểm suy thoái của camera (Camera Health Score) và độ tin cậy của thuật toán thị giác nhận diện xe cộ / người đi bộ để xác định ngưỡng an toàn cho xe.

---

### 2. Method (Thuật toán & Đóng góp của cá nhân)
* **Thuật toán No-Reference IQA:** Đánh giá 4 thành phần:
  * $S_{\text{sharp}}$ (Độ nét chống nhiễu), $S_{\text{exp}}$ (Chất lượng phơi sáng), $S_{\text{info}}$ (Shannon Entropy), $P_{\text{soiling}}$ (Bùn bẩn bám kính).
  * Hàm Bottleneck Penalty:
    $$H = 100 \times \left( \min(S_{\text{sharp}}, S_{\text{exp}}, S_{\text{info}})^{0.4} \times (0.35 S_{\text{sharp}} + 0.35 S_{\text{exp}} + 0.30 S_{\text{info}})^{0.6} \right) \times (1.0 - 0.65 P_{\text{soiling}})$$
* **Đóng góp của cá nhân:** Thiết lập pipeline đánh giá mô hình YOLOv8n (`src/detector_eval.py`), trích xuất các metric nhận diện vật thể xe ADAS (Cars, Buses, Trucks, Pedestrians), vẽ đồ thị hồi quy tương quan và phân tích dữ liệu bảng số liệu.

---

### 3. Benchmark (Kết quả thực nghiệm số & Bằng chứng chạy thật)
* **Lệnh chạy tái lập kết quả:**
  ```powershell
  python -m src.benchmark
  ```
* **Bảng số liệu kiểm tra định lượng (Trích xuất từ [`outputs/benchmark_table.md`](../outputs/benchmark_table.md)):**

| Tình trạng thử nghiệm | Cấp độ | Health Score | Trạng thái ADAS | Hành vi điều khiển | Trọng số Fusion | Số vật thể nhận diện (Retention) | Mean Conf (Drop) | Độ trễ Scorer |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean Baseline** | - | **92.1%** | `HEALTHY` | `FULL_CONFIDENCE` | **1.00** | 6 (100%) | 0.66 (+0.00) | **23.5 ms** |
| **Motion Blur** | Level 2 | **31.3%** | `CRITICAL` | `FALLBACK_DISENGAGE` | **0.00** | 4 (67%) | 0.73 (+0.08) | **30.3 ms** |
| **Motion Blur** | Level 4 | **20.4%** | `CRITICAL` | `FALLBACK_DISENGAGE` | **0.00** | 3 (50%) | 0.47 (-0.19) | **23.7 ms** |
| **Sun Glare** | Level 3 | **17.1%** | `CRITICAL` | `FALLBACK_DISENGAGE` | **0.00** | 6 (100%) | 0.59 (-0.07) | **33.0 ms** |
| **Sun Glare** | Level 5 | **2.7%** | `CRITICAL` | `FALLBACK_DISENGAGE` | **0.00** | 1 (17%) | 0.54 (-0.11) | **30.6 ms** |
| **Darkness** | Level 3 | **38.4%** | `CRITICAL` | `FALLBACK_DISENGAGE` | **0.00** | 5 (83%) | 0.56 (-0.10) | **24.2 ms** |
| **Darkness** | Level 5 | **29.1%** | `CRITICAL` | `FALLBACK_DISENGAGE` | **0.00** | 0 (0%) | 0.00 (-0.66) | **28.7 ms** |
| **Mud Soiling** | Level 3 | **84.2%** | `HEALTHY` | `FULL_CONFIDENCE` | **1.00** | 6 (100%) | 0.64 (-0.01) | **22.1 ms** |
| **Mud Soiling** | Level 5 | **47.8%** | `DEGRADED` | `DOWN_WEIGHT` | **0.29** | 4 (67%) | 0.61 (-0.04) | **25.6 ms** |

* **Đồ thị phân tích tương quan do nhóm xuất bản:**
  * [`outputs/health_vs_confidence.png`](../outputs/health_vs_confidence.png): Đồ thị chứng minh khi Health Score rơi xuống dưới 42%, tỷ lệ giữ vật thể tụt dốc thảm hại, xác thực tính đúng đắn của ngưỡng ngắt tự lái (Critical Disengage Threshold).
  * [`outputs/hud_comparison.png`](../outputs/hud_comparison.png): Minh họa thực tế trên giao diện HUD.

---

### 4. Failure Case (Phát hiện khoa học thực tế)
* **Phát hiện lỗi thực tế:** Khi trời tối hoàn toàn và cảm biến khuếch đại nhiễu hạt (Shot Noise), công thức Laplacian thông thường bị đánh lừa và cho điểm lên tới **3738.1** (cao hơn ảnh ngày rõ nét 3619.1).
* **Khắc phục:** Sử dụng Gaussian Pre-filter ($3\times 3$, $\sigma=0.8$) loại bỏ nhiễu hạt, đưa điểm độ nét về đúng thực tế là **95.4**.

---

### 5. Engineering Decision (Quyết định đưa vào xe thật)
1. **Kiểm soát tài nguyên:** Health Scorer chạy mất ~2.3 ms trên Jetson AGX Orin, chỉ chiếm chưa tới 4% chu kỳ của camera 30 FPS.
2. **Ngưỡng hành động an toàn:**
   * $H \ge 70\%$: Duy trì trọng số Camera = 1.0.
   * $42\% \le H < 70\%$: Giảm trọng số camera xuống 0.35, dồn quyền sang Radar 77GHz và LiDAR.
   * $H < 42\%$: Ngắt lái tự động, phát cảnh báo khẩn cấp cho tài xế tiếp quản trong 4 giây hoặc tự động kích hoạt Safe Stop.
