# BẢN BÁO CÁO CÁ NHÂN — LAB DAY 04 (VLEARN SUBMISSION)
## Topic T1: Camera Degradation Health Score & Sensor Integrity Monitor

* **Họ và tên sinh viên:** Cao Đức Hiếu
* **Mã số sinh viên (MSSV):** 2A202602701
* **Nhóm:** LaoGaKho (Lão Gà Kho)
* **Vai trò trong nhóm:** Research Lead & Real-world Dataset Pipeline
* **Repository nhóm:** https://github.com/QuocHiep123/track4_day4_minilab (Branch: `CaoDucHieu-2A202602701`)
* **Danh sách thành viên:** Đối chiếu tại [`TEAMMATES.md`](../TEAMMATES.md)

---

### 1. Problem (Vấn đề & Bối cảnh kỹ thuật)
* **Platform & Cảm biến:** Hệ thống tự hành ADAS L2+/L3 (Front-facing RGB Camera).
* **Failure thực tế:** Phân tích các dạng suy thoái chất lượng hình ảnh xe tự hành dựa trên tiêu chuẩn benchmark thế giới **nuScenes-C / KITTI-C** (*Dong et al., CVPR 2023*):
  1. *Optical Corruptions:* Motion blur và Defocus blur do rung chấn mặt đường.
  2. *Lighting Corruptions:* Sun glare (hoàng hôn) và Severe underexposure (đêm tối).
  3. *Weather Corruptions:* Mưa lớn, sương mù làm suy thoái tương phản không khí.
  4. *Surface Corruptions:* Bùn đất và nước bẩn bắn che khuất cục bộ mặt kính thấu kính.
* **Rủi ro:** Mô hình thị giác máy tính giả định đầu vào luôn hoàn hảo. Khi gặp suy giảm, mạng nơ-ron sinh ra lỗi ảo giác hoặc bỏ sót đối tượng nghiêm trọng.

---

### 2. Method (Thuật toán & Đóng góp của cá nhân)
* **Thuật toán & Chuẩn nghiên cứu:**
  * Xây dựng pipeline tạo 5 loại suy thoái chuẩn hóa theo 5 mức độ nghiêm trọng (Level 1 đến Level 5) mô phỏng chính xác hiện tượng vật lý quang học (`src/degradation.py`).
  * Tham chiếu công thức No-Reference IQA để đo độ suy giảm gradient Sobel, tỷ lệ bão hòa pixel $r_{\text{over}}, r_{\text{under}}$, và hàm phạt bùn đất qua phân tích ma trận năng lượng ô lưới $6\times 6$.
  * Thiết kế hàm Bottleneck Penalty:
    $$H = 100 \times \left( \min(S_{\text{sharp}}, S_{\text{exp}}, S_{\text{info}})^{0.4} \times (0.35 S_{\text{sharp}} + 0.35 S_{\text{exp}} + 0.30 S_{\text{info}})^{0.6} \right) \times (1.0 - 0.65 P_{\text{soiling}})$$
* **Đóng góp của cá nhân:** Khảo cứu chuẩn nuScenes-C, thu thập tập dữ liệu ảnh lái xe thực tế (KITTI, Wikimedia CC0, Urban Bus scenes), thiết lập hàm suy thoái quang học và viết tài liệu nghiên cứu.

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
| **Sun Glare** | Level 1 | **60.4%** | `DEGRADED` | `DOWN_WEIGHT` | **0.36** | 6 (100%) | 0.65 (-0.01) | **26.7 ms** |
| **Sun Glare** | Level 3 | **17.1%** | `CRITICAL` | `FALLBACK_DISENGAGE` | **0.00** | 6 (100%) | 0.59 (-0.07) | **33.0 ms** |
| **Darkness** | Level 2 | **55.3%** | `DEGRADED` | `DOWN_WEIGHT` | **0.33** | 4 (67%) | 0.77 (+0.12) | **25.8 ms** |
| **Darkness** | Level 5 | **29.1%** | `CRITICAL` | `FALLBACK_DISENGAGE` | **0.00** | 0 (0%) | 0.00 (-0.66) | **28.7 ms** |
| **Rain/Fog** | Level 4 | **60.1%** | `DEGRADED` | `DOWN_WEIGHT` | **0.36** | 2 (33%) | 0.51 (-0.15) | **23.1 ms** |
| **Mud Soiling** | Level 4 | **61.3%** | `DEGRADED` | `DOWN_WEIGHT` | **0.37** | 7 (117%) | 0.59 (-0.06) | **21.1 ms** |

* **Minh chứng trực quan:**
  * Lưới 10 trạng thái hư hại cảm biến: [`outputs/degradation_grid.png`](../outputs/degradation_grid.png)
  * Đồ thị phân tích tương quan 4 chiều: [`outputs/health_vs_confidence.png`](../outputs/health_vs_confidence.png)

---

### 4. Failure Case (Phát hiện khoa học thực tế)
* **Khác biệt giữa Paper và Nhóm tự Benchmark:** Trong paper nuScenes-C, tác giả chỉ đánh giá độ sụt mAP tĩnh ở đầu ra mạng nơ-ron. Nhóm tự thực nghiệm và phát hiện lỗi toán học ở tầng trích xuất đặc trưng:
  * Trong cảnh đêm tối (Darkness), cảm biến khuếch đại hạt nhiễu (High-ISO Shot Noise) khiến toán tử Laplacian thông thường tưởng ảnh rất nét, cho điểm vọt lên **3738.1** (lớn hơn ban ngày 3619.1)!
  * Nhóm khắc phục bằng bộ lọc Gaussian tiền xử lý ($3\times 3$, $\sigma=0.8$) lọc sạch nhiễu hạt, đưa độ nét sụp về đúng giá trị thực **95.4**.

---

### 5. Engineering Decision (Quyết định đưa vào xe thật)
1. **Kiểm soát độ trễ:** Health Scorer thực thi trong ~2.3 ms trên Jetson AGX Orin, không gây nghẽn luồng so với mô hình AI nhận diện (67 ms).
2. **Chiến lược Down-weight đa cảm biến:** Khi $H < 70\%$, giảm tỷ trọng camera xuống 0.35, dồn quyền định vị và ước lượng khoảng cách sang Radar 77GHz và LiDAR.
3. **Chiến lược Fallback an toàn:** Khi $H < 42\%$, lập tức ngắt tự lái, yêu cầu tài xế tiếp quản trong 4 giây hoặc kích hoạt dừng đỗ khẩn cấp an toàn (Safe Stop).
