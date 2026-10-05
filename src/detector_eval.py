"""
Object Detection Evaluator & ADAS HUD Visualizer using YOLOv8n
Measures impact of sensor degradation on downstream perception metrics:
- Detection count retention rate
- Mean confidence score drop
- ADAS HUD overlay with real-time Camera Health Badge & Fusion Decision
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np

# Patch PyTorch / Torchvision operator mismatch gracefully
import torch
try:
    my_lib = torch.library.Library('torchvision', 'DEF')
    my_lib.define('nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor')
except Exception:
    pass

import torchvision.ops

def fast_nms(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    b_np = boxes.detach().cpu().numpy()
    s_np = scores.detach().cpu().numpy()
    boxes_wh = np.column_stack([b_np[:, 0], b_np[:, 1], b_np[:, 2] - b_np[:, 0], b_np[:, 3] - b_np[:, 1]])
    idx = cv2.dnn.NMSBoxes(boxes_wh.tolist(), s_np.tolist(), score_threshold=0.001, nms_threshold=float(iou_threshold))
    if len(idx) == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    return torch.tensor(np.array(idx).flatten(), dtype=torch.int64, device=boxes.device)

torchvision.ops.nms = fast_nms
torchvision.ops.boxes.nms = fast_nms

from ultralytics import YOLO

# Relevant ADAS traffic classes from COCO
ADAS_CLASSES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
    9: 'traffic light',
    11: 'stop sign'
}


@dataclass
class DetectionItem:
    cls_name: str
    conf: float
    bbox: Tuple[int, int, int, int]


@dataclass
class DetectorReport:
    num_detections: int
    mean_confidence: float
    detections: List[DetectionItem]
    latency_ms: float


class DetectorEvaluator:
    def __init__(self, model_path: str = 'yolov8n.pt', conf_thresh: float = 0.25):
        self.conf_thresh = conf_thresh
        self.model = YOLO(model_path)

    def evaluate(self, image: np.ndarray) -> DetectorReport:
        import time
        start_t = time.perf_counter()
        
        results = self.model(image, conf=self.conf_thresh, verbose=False)[0]
        dets: List[DetectionItem] = []
        
        for b in results.boxes:
            cls_id = int(b.cls[0])
            conf = float(b.conf[0])
            # Filter for ADAS relevant classes
            if cls_id in ADAS_CLASSES:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                dets.append(DetectionItem(
                    cls_name=ADAS_CLASSES[cls_id],
                    conf=round(conf, 3),
                    bbox=(x1, y1, x2, y2)
                ))
                
        latency_ms = round((time.perf_counter() - start_t) * 1000.0, 2)
        mean_conf = round(float(np.mean([d.conf for d in dets])), 3) if dets else 0.0
        
        return DetectorReport(
            num_detections=len(dets),
            mean_confidence=mean_conf,
            detections=dets,
            latency_ms=latency_ms
        )

    def render_adas_hud(
        self,
        image: np.ndarray,
        det_report: DetectorReport,
        health_report: Any
    ) -> np.ndarray:
        """Render automotive HUD overlay showing sensor health, decision, and detected objects."""
        annotated = image.copy()
        h, w = annotated.shape[:2]
        
        # 1. Draw detection bounding boxes
        for item in det_report.detections:
            x1, y1, x2, y2 = item.bbox
            color = (0, 255, 0) if item.cls_name in ['car', 'bus', 'truck'] else (255, 128, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            label = f"{item.cls_name} {item.conf:.2f}"
            t_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - 20), (x1 + t_size[0] + 6, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
        # 2. Top HUD Status Bar
        hud_h = 70
        hud_overlay = annotated.copy()
        cv2.rectangle(hud_overlay, (0, 0), (w, hud_h), (18, 22, 28), -1)
        cv2.addWeighted(hud_overlay, 0.88, annotated, 0.12, 0, annotated)
        
        # Status Color Badge
        if health_report.status == "HEALTHY":
            status_color = (40, 225, 60)       # Vivid Green
        elif health_report.status == "DEGRADED":
            status_color = (30, 210, 255)      # Amber / Yellow
        else:
            status_color = (45, 50, 245)       # Bright Alert Red
            
        # Row 1: Title (Left) + Action & Sensor Fusion Weight (Right)
        cv2.putText(annotated, "ADAS SENSOR HEALTH MONITOR", (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (215, 225, 235), 2, cv2.LINE_AA)
        
        action_label = f"DECISION: {health_report.action} | WEIGHT: {health_report.camera_fusion_weight:.2f}"
        (act_w, _), _ = cv2.getTextSize(action_label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.putText(annotated, action_label, (max(w - act_w - 16, int(w * 0.45)), 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Row 2: Large Health Score Badge (Left) + Detailed Sub-metrics (Right)
        health_text = f"HEALTH: {health_report.health_score:.1f}% [{health_report.status}]"
        cv2.putText(annotated, health_text, (16, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.70, status_color, 2, cv2.LINE_AA)
        
        sub_metrics = f"Blur: {health_report.blur_score:.2f} | Exp: {health_report.exposure_score:.2f} | Info: {health_report.entropy_score:.2f}"
        (sub_w, _), _ = cv2.getTextSize(sub_metrics, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.putText(annotated, sub_metrics, (max(w - sub_w - 16, int(w * 0.50)), 56), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190, 210, 230), 1, cv2.LINE_AA)
        
        # Bottom Diagnostics Strip
        sub_h = 28
        sub_overlay = annotated.copy()
        cv2.rectangle(sub_overlay, (0, h - sub_h), (w, h), (12, 15, 18), -1)
        cv2.addWeighted(sub_overlay, 0.85, annotated, 0.15, 0, annotated)
        
        diag_text = (
            f"Detections: {det_report.num_detections} (Mean Conf: {det_report.mean_confidence:.2f}) | "
            f"IQA Latency: {health_report.latency_ms:.1f}ms | YOLOv8n: {det_report.latency_ms:.1f}ms"
        )
        cv2.putText(annotated, diag_text, (16, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (175, 215, 245), 1, cv2.LINE_AA)
        
        return annotated
