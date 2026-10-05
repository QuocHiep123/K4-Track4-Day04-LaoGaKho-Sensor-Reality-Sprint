import cv2
import numpy as np
from skimage.measure import shannon_entropy

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

class CameraHealthBenchmark:
    def __init__(self, yolo_model_name='yolov8n.pt'):
        """
        Setup benchmark framework. 
        Sử dụng YOLO làm proxy cho task nhận diện phía sau.
        """
        if HAS_YOLO:
            self.model = YOLO(yolo_model_name)
        else:
            self.model = None

    def calculate_metrics(self, image_color):
        """Tính toán các metrics cho một cấu hình ảnh"""
        gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
        
        # 1. Blur Score (Laplacian Variance - độ sắc nét)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 2. Saturation Ratio (Tỷ lệ pixel bị bão hòa sáng - Glare/Overexposure)
        # Giả sử pixel > 250 là bị bão hòa
        saturated_pixels = np.sum(gray > 250)
        total_pixels = gray.size
        saturation_ratio = saturated_pixels / total_pixels
        
        # 3. Entropy Metric (Lượng thông tin/chi tiết ảnh)
        entropy_score = shannon_entropy(gray)
        
        # 4. Detector Confidence (Proxy Metric cho ADAS)
        det_conf = 0.0
        if HAS_YOLO:
            results = self.model(image_color, verbose=False)
            confidences = [box.conf.item() for r in results for box in r.boxes]
            if confidences:
                det_conf = np.mean(confidences)
                
        return {
            'blur': blur_score,
            'saturation_ratio': saturation_ratio * 100, # Tính theo %
            'entropy': entropy_score,
            'confidence': det_conf
        }

    # --- Các hàm tạo lỗi (Degradation / Perturbation) ---
    def simulate_blur(self, image, ksize):
        """Tạo lỗi mờ (Blur) do out-of-focus hoặc sương mù (Gaussian Blur)"""
        return cv2.GaussianBlur(image, (ksize, ksize), 0)
        
    def simulate_brightness_change(self, image, alpha, beta):
        """Tạo lỗi sáng/tối (Brightness/Glare)
        alpha > 1: tăng sáng (glare), alpha < 1: giảm sáng (night)
        """
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return adjusted
        
    def simulate_rain_noise(self, image, mean=0, std=30):
        """Tạo lỗi nhiễu sensor/Mưa (Gaussian Noise)"""
        noise = np.random.normal(mean, std, image.shape).astype(np.int16)
        noisy_img = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return noisy_img

    def run_benchmark(self, image_path):
        """Chạy pipeline so sánh đối chứng Baseline vs Degraded (Tạo 3-5 mức degradation)"""
        baseline_img = cv2.imread(image_path)
        if baseline_img is None:
            print(f"Lỗi: Không thể đọc ảnh gốc tại '{image_path}'")
            print("Vui lòng thay thế bằng một file ảnh thực tế của bạn.")
            return
            
        print(f"{'Điều kiện (Condition)':<25} | {'Blur Score':<12} | {'Sat Ratio(%)':<12} | {'Entropy':<10} | {'Det Conf':<10}")
        print("-" * 78)
        
        # 1. Đo Baseline (Không chủ động gây lỗi)
        baseline_metrics = self.calculate_metrics(baseline_img)
        self.print_row("Baseline (Gốc)", baseline_metrics)
        
        # 2. Degradation 1: Blur nhẹ (k=15)
        blur_light = self.simulate_blur(baseline_img, 15)
        self.print_row("Deg 1: Blur nhẹ", self.calculate_metrics(blur_light))
        
        # 3. Degradation 2: Blur nặng (k=31)
        blur_heavy = self.simulate_blur(baseline_img, 31)
        self.print_row("Deg 2: Blur nặng", self.calculate_metrics(blur_heavy))
        
        # 4. Degradation 3: Glare/Quá sáng (alpha=1.5, beta=50)
        glare_img = self.simulate_brightness_change(baseline_img, 1.5, 50)
        self.print_row("Deg 3: Glare (Sáng chói)", self.calculate_metrics(glare_img))
        
        # 5. Degradation 4: Night/Quá tối (alpha=0.3, beta=0)
        night_img = self.simulate_brightness_change(baseline_img, 0.3, 0)
        self.print_row("Deg 4: Night (Thiếu sáng)", self.calculate_metrics(night_img))
        
        # 6. Degradation 5: Rain/Noise (std=30)
        noisy_img = self.simulate_rain_noise(baseline_img, std=30)
        self.print_row("Deg 5: Rain/Noise", self.calculate_metrics(noisy_img))

    def print_row(self, name, m):
        print(f"{name:<25} | {m['blur']:<12.2f} | {m['saturation_ratio']:<12.2f} | {m['entropy']:<10.2f} | {m['confidence']:<10.2f}")

if __name__ == "__main__":
    benchmark = CameraHealthBenchmark()
    
    # BẠN CẦN THAY FILE DƯỚI ĐÂY BẰNG 1 ẢNH THỰC TẾ (ví dụ ảnh dashcam ngoài đường)
    sample_image = 'test_adas_image.jpg' 
    benchmark.run_benchmark(sample_image)
