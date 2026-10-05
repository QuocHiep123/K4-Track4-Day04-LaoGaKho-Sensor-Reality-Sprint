# BẢN BÁO CÁO CÁ NHÂN — LAB DAY 04 (VLEARN SUBMISSION)
## Topic T1: Camera Degradation Health Score & Sensor Integrity Monitor

* **Họ và tên sinh viên:** Trần Công Thiện
* **Mã số sinh viên (MSSV):** 2A202602579
* **Nhóm:** LaoGaKho (Lão Gà Kho)
* **Vai trò trong nhóm:** Failure Case Analyst & Engineering Decision / Safe Stop
* **Repository nhóm:** https://github.com/QuocHiep123/track4_day4_minilab (Branch: `trancongthien-02579`)
* **Danh sách thành viên:** Đối chiếu tại [`TEAMMATES.md`](../TEAMMATES.md)

---

### 1. Problem (Vấn đề & Bối cảnh kỹ thuật)
* **Platform & Cảm biến:** Xe tự hành ADAS Cấp độ L2+/L3 (Front Windshield Camera 1080p).
* **Failure thực tế:** Phân tích các trường hợp cảm biến thị giác bị đánh lừa trong môi trường giao thông phức tạp:
  1. *Lóa sáng cục bộ (Local Glare):* Đèn pha xe ngược chiều tạo quầng sáng lớn, làm mờ các đối tượng đi bộ sát lề đường.
  2. *Nhiễu đêm giả mạo độ nét (High-ISO Shot Noise):* Cảm biến bị thiếu sáng tự động đẩy gain sinh ra nhiễu hạt đánh lừa các bộ lọc vi phân bậc 2 cổ điển.
  3. *Bùn đất che khuất cục bộ (Mud Soiling):* Bùn bám theo mảng trên kính lái làm mất khả năng quan sát một số góc phần tư.
* **Mục tiêu cá nhân:** Khảo sát các failure cases, đề xuất luật kiểm soát an toàn chức năng (Functional Safety ISO 26262), thiết lập cơ chế hạ trọng số cảm biến (Down-weighting) và quy trình dừng xe khẩn cấp (Safe Stop / Minimum Risk Maneuver).

---

### 2. Method (Thuật toán & Đóng góp của cá nhân)
* **Thuật toán No-Reference IQA:** Đánh giá độ tin cậy camera không cần ảnh gốc:
  * $S_{\text{sharp}}$ (Độ nét có lọc nhiễu Gaussian $3\times 3$), $S_{\text{exp}}$ (Chất lượng phơi sáng), $S_{\text{info}}$ (Shannon Entropy), $P_{\text{soiling}}$ (Bùn bẩn bám kính).
  * Hàm Bottleneck Penalty:
    $$H = 100 \times \left( \min(S_{\text{sharp}}, S_{\text{exp}}, S_{\text{info}})^{0.4} \times (0.35 S_{\text{sharp}} + 0.35 S_{\text{exp}} + 0.30 S_{\text{info}})^{0.6} \right) \times (1.0 - 0.65 P_{\text{soiling}})$$
* **Đóng góp của cá nhân:** Chuyên sâu phân tích cơ chế thất bại (Failure Cases), xây dựng thuật toán phát hiện bất đối xứng năng lượng bùn đất ô lưới $6\times 6$ và thiết kế quy trình xử lý sự cố Safe Stop cho xe tự hành.

---

### 3. Benchmark (Kết quả thực nghiệm số & Bằng chứng chạy thật)
* **Lệnh chạy tái lập kết quả:**
  ```powershell
  python -m src.benchmark
  ```
* **Bảng số liệu kiểm tra định lượng (Trích xuất từ [`outputs/benchmark_table.md`](../outputs/benchmark_table.md)):**

| Điều kiện kiểm tra | Cấp độ | Health Score | Trạng thái ADAS | Hành vi điều khiển | Trọng số Fusion | Detections (Retention) | Mean Conf (Drop) | Độ trễ Scorer |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean Baseline** | - | **92.1%** | `HEALTHY` | `FULL_CONFIDENCE` | **1.00** | 6 (100%) | 0.66 (+0.00) | **23.5 ms** |
| **Motion Blur** | Level 3 | **20.7%** | `CRITICAL` | `FALLBACK_DISENGAGE` | **0.00** | 6 (100%) | 0.50 (-0.16) | **37.4 ms** |
| **Sun Glare** | Level 3 | **17.1%** | `CRITICAL` | `FALLBACK_DISENGAGE` | **0.00** | 6 (100%) | 0.59 (-0.07) | **33.0 ms** |
| **Sun Glare** | Level 5 | **2.7%** | `CRITICAL` | `FALLBACK_DISENGAGE` | **0.00** | 1 (17%) | 0.54 (-0.11) | **30.6 ms** |
| **Darkness** | Level 4 | **32.3%** | `CRITICAL` | `FALLBACK_DISENGAGE` | **0.00** | 3 (50%) | 0.50 (-0.16) | **19.0 ms** |
| **Mud Soiling** | Level 4 | **61.3%** | `DEGRADED` | `DOWN_WEIGHT` | **0.37** | 7 (117%) | 0.59 (-0.06) | **21.1 ms** |
| **Mud Soiling** | Level 5 | **47.8%** | `DEGRADED` | `DOWN_WEIGHT` | **0.29** | 4 (67%) | 0.61 (-0.04) | **25.6 ms** |

* **Minh chứng trực quan:**
  * Bảng so sánh 3 trạng thái HUD: [`outputs/hud_comparison.png`](../outputs/hud_comparison.png)
  * Đồ thị phân tích 4 góc độ: [`outputs/health_vs_confidence.png`](../outputs/health_vs_confidence.png)

---

### 4. Failure Case (Phát hiện khoa học thực tế)
* **Phát hiện lỗi kinh điển:** Công thức Laplacian Variance sách giáo khoa bị **đánh lừa hoàn toàn** trong cảnh đêm thiếu sáng có nhiễu hạt:
  * Điểm Laplacian ảnh tối có nhiễu đạt tới **3738.1**, cao hơn cả ảnh ban ngày rõ nét (3619.1) do thuật toán hiểu nhầm hạt nhiễu đơn lẻ là cạnh sắc nét!
* **Giải pháp khắc phục:** Thiết kế bộ lọc Gaussian Pre-filter ($3\times 3$, $\sigma=0.8$) triệt tiêu nhiễu hạt trước khi tính đạo hàm bậc hai, đưa điểm độ nét về đúng thực tế là **95.4**.
* **Failure case thứ hai (Chói lóa cục bộ):** Đèn pha ngược chiều làm lóa một vùng ở giữa khung hình. Hàm Bottleneck giúp phát hiện sụt giảm phơi sáng ngay lập tức, ngăn ngừa hệ thống lấy trung bình bù lỗ gây nguy hiểm.

---

### 5. Engineering Decision (Quyết định đưa vào xe thật)
1. **Hiệu năng & Tài nguyên:** Thuật toán tính toán mất ~2.3 ms trên Jetson AGX Orin, đảm bảo kiểm tra từng frame ở tốc độ 30-60 FPS.
2. **Quy trình Safe Stop & Down-weighting:**
   * $H \ge 70\%$: Tin cậy camera 100% (Weight = 1.0).
   * $42\% \le H < 70\%$: Giảm tỷ trọng camera xuống 0.35, dồn quyền điều khiển sang Radar FMCW 77GHz và LiDAR.
   * $H < 42\%$: Ngắt lái tự động Autopilot, phát cảnh báo Takeover Request (rung vô lăng + âm thanh khẩn cấp). Nếu tài xế không phản ứng trong 4 giây $\implies$ Xe tự động bật đèn khẩn cấp tấp lề an toàn (Safe Stop / Minimum Risk Maneuver).
