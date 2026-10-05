# DANH SÁCH THÀNH VIÊN NHÓM — TEAMMATES.md
## Nhóm: LaoGaKho (Lão Gà Kho)
**Học phần:** VinUni AI Thực Chiến - Track 2 / Track 4 - Day 04  
**Chủ đề:** Topic T1 — Camera Degradation Health Score  
**Repository:** https://github.com/QuocHiep123/track4_day4_minilab  

---

| STT | Họ và tên | Mã sinh viên (MSSV) | Vai trò trong nhóm (Workflow 120m) | Bản báo cáo cá nhân (VLearn) | Branch cá nhân |
| :---: | :--- | :---: | :--- | :--- | :--- |
| **1** | **Đặng Quốc Hiệp** | **2A202602755** | **Team Lead & Web UI / ADAS Demo Engineer** | [`reports/REPORT_DangQuocHiep_2A202602755.md`](reports/REPORT_DangQuocHiep_2A202602755.md) | `DangQuocHiep-2A202602755` / `feat/gun` |
| **2** | **Cao Đức Hiếu** | **2A202602701** | **Research Lead & Real-world Dataset Pipeline** | [`reports/REPORT_CaoDucHieu_2A202602701.md`](reports/REPORT_CaoDucHieu_2A202602701.md) | `CaoDucHieu-2A202602701` |
| **3** | **Nguyễn Nhật Thắng** | **2A202602727** | **Code Runner & Health Scorer Algorithm Developer** | [`reports/REPORT_NguyenNhatThang_2A202602727.md`](reports/REPORT_NguyenNhatThang_2A202602727.md) | `NguyenNhatThang-2A202602727` |
| **4** | **Trần Thành Thái** | **2A202602454** | **Benchmark & Downstream Perception Evaluator** | [`reports/REPORT_TranThanhThai_2A202602454.md`](reports/REPORT_TranThanhThai_2A202602454.md) | `TranThanhThai-2A202602454` |
| **5** | **Trần Công Thiện** | **2A202602579** | **Failure Case Analyst & Engineering Decision / Safe Stop** | [`reports/REPORT_TranCongThien_2A202602579.md`](reports/REPORT_TranCongThien_2A202602579.md) | `trancongthien-02579` |

---

### Hướng dẫn đối chiếu khi chấm bài trên VLearn:
- **Mã nguồn chung:** `src/degradation.py`, `src/health_score.py`, `src/benchmark.py`, `src/detector_eval.py`, `app.py`.
- **Bằng chứng chạy thực tế:**
  - Bảng số liệu: [`outputs/benchmark_table.md`](outputs/benchmark_table.md)
  - Đồ thị phân tích 4 góc độ: [`outputs/health_vs_confidence.png`](outputs/health_vs_confidence.png)
  - Giao diện ADAS HUD: [`outputs/hud_comparison.png`](outputs/hud_comparison.png)
  - Lưới 10 trạng thái hư hại: [`outputs/degradation_grid.png`](outputs/degradation_grid.png)
- **Slide trình chiếu 1 trang:** Xem tại [`REPORT.md`](REPORT.md) hoặc chạy web demo tại `http://localhost:5000/slide`.
