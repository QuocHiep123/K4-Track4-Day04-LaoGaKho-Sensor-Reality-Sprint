"""
Interactive ADAS Camera Health & Degradation Assessment Web App
Provides:
1. Live Interactive Demo (Image upload, presets, degradation sliders, HUD visualization)
2. Slide Presentation Deck (1-Page / 1-Slide conforming to Section 6 of Lab Rubric)
"""

import os
import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory

from src.degradation import apply_degradation, DEGRADATION_DISPATCH
from src.health_score import CameraHealthScorer
from src.detector_eval import DetectorEvaluator

app = Flask(__name__, static_folder='static', template_folder='templates')
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

scorer = CameraHealthScorer()
evaluator = DetectorEvaluator()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/slide')
def slide():
    return render_template('slide.html')


@app.route('/outputs/<path:filename>')
def serve_outputs(filename):
    return send_from_directory('outputs', filename)


@app.route('/samples/<path:filename>')
def serve_samples(filename):
    return send_from_directory('data/samples', filename)


@app.route('/api/samples')
def list_samples():
    import glob
    files = glob.glob('data/samples/*.jpg')
    samples = []
    for f in files:
        name = os.path.basename(f)
        samples.append({
            'id': name,
            'name': name.replace('_', ' ').replace('.jpg', '').title(),
            'url': f'/samples/{name}'
        })
    return jsonify(samples)


@app.route('/api/evaluate', methods=['POST'])
def evaluate_api():
    try:
        data = request.json or {}
        img_source = data.get('image_source', 'day_drive.jpg')
        deg_type = data.get('deg_type', 'none')
        severity = int(data.get('severity', 3))
        run_detector = bool(data.get('run_detector', True))

        # 1. Load Image
        if img_source.startswith('data:image'):
            # Base64 uploaded image
            header, encoded = img_source.split(',', 1)
            img_bytes = base64.b64decode(encoded)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            sample_path = os.path.join('data/samples', os.path.basename(img_source))
            if not os.path.exists(sample_path):
                sample_path = 'data/samples/day_drive.jpg'
            img = cv2.imread(sample_path)

        if img is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        # Resize if image is extremely massive to maintain fast real-time response
        if max(img.shape[:2]) > 1400:
            scale = 1400.0 / max(img.shape[:2])
            img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

        # 2. Apply Degradation
        if deg_type in DEGRADATION_DISPATCH and severity > 0:
            proc_img = apply_degradation(img, deg_type, severity)
        else:
            proc_img = img.copy()

        # 3. Compute Health Score
        h_rep = scorer.evaluate(proc_img)

        # 4. Run Object Detection & Render HUD
        if run_detector:
            d_rep = evaluator.evaluate(proc_img)
            hud_img = evaluator.render_adas_hud(proc_img, d_rep, h_rep)
            det_info = {
                'num_detections': d_rep.num_detections,
                'mean_confidence': d_rep.mean_confidence,
                'detections': [{'class': item.cls_name, 'conf': item.conf} for item in d_rep.detections],
                'latency_ms': d_rep.latency_ms
            }
        else:
            hud_img = proc_img
            det_info = {
                'num_detections': 0,
                'mean_confidence': 0.0,
                'detections': [],
                'latency_ms': 0.0
            }

        # Encode resulting images to JPEG base64 for real-time frontend display
        _, buf_hud = cv2.imencode('.jpg', hud_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        hud_b64 = base64.b64encode(buf_hud).decode('utf-8')

        _, buf_deg = cv2.imencode('.jpg', proc_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        deg_b64 = base64.b64encode(buf_deg).decode('utf-8')

        return jsonify({
            'health_score': h_rep.health_score,
            'status': h_rep.status,
            'action': h_rep.action,
            'camera_fusion_weight': h_rep.camera_fusion_weight,
            'blur_score': h_rep.blur_score,
            'exposure_score': h_rep.exposure_score,
            'entropy_score': h_rep.entropy_score,
            'contrast_score': h_rep.contrast_score,
            'soiling_penalty': h_rep.soiling_penalty,
            'overexposed_ratio': h_rep.overexposed_ratio,
            'underexposed_ratio': h_rep.underexposed_ratio,
            'scorer_latency_ms': h_rep.latency_ms,
            'detector': det_info,
            'hud_image': f"data:image/jpeg;base64,{hud_b64}",
            'degraded_image': f"data:image/jpeg;base64,{deg_b64}",
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("[*] Starting ADAS Camera Degradation Web App on http://127.0.0.1:5000 ...")
    app.run(host='127.0.0.1', port=5000, debug=False)
