"""
T1 - Camera degradation health score (Sensor Reality Sprint)

Cai dat:
    pip install opencv-python numpy matplotlib
    pip install ultralytics          # tuy chon, de do detector confidence

Chay benchmark tren thu muc anh hoac 1 video:
    python t1_health.py --images data/            # thu muc anh .jpg/.png
    python t1_health.py --images road.mp4         # lay moi 30 frame 1 anh
    python t1_health.py --images data/ --yolo     # them detector confidence

Demo webcam real-time:
    python t1_health.py --webcam [--yolo]
    Phim: 0 = sach, b = blur, d = dem, o = choi sang, n = nhieu, r = mua,
          + / - = tang/giam muc do (1-5), q = thoat
"""
import argparse, csv, glob, os
import cv2
import numpy as np

# ---------------------------------------------------------------- degradation
def deg_blur(img, s):
    k = 4 * s + 1
    return cv2.GaussianBlur(img, (k, k), 0)

def deg_dark(img, s):  # gia lap ban dem: toi + nhieu cam bien
    x = img.astype(np.float32) / 255.0
    x = (x ** (1 + 0.5 * s)) * (1 - 0.12 * s)
    x = x * 255 + np.random.normal(0, 2 * s, img.shape)
    return np.clip(x, 0, 255).astype(np.uint8)

def deg_overexpose(img, s):  # gia lap glare / choi nang
    return np.clip(img.astype(np.float32) * (1 + 0.35 * s) + 15 * s, 0, 255).astype(np.uint8)

def deg_noise(img, s):
    return np.clip(img + np.random.normal(0, 8 * s, img.shape), 0, 255).astype(np.uint8)

def deg_rain(img, s):
    h, w = img.shape[:2]
    layer = np.zeros_like(img)
    for _ in range(120 * s):
        x, y = np.random.randint(0, w), np.random.randint(0, h)
        L = np.random.randint(10, 25)
        cv2.line(layer, (x, y), (x + L // 4, y + L), (200, 200, 200), 1)
    out = cv2.addWeighted(img, 1.0, cv2.GaussianBlur(layer, (3, 3), 0), 0.6, 0)
    return cv2.GaussianBlur(out, (2 * s + 1, 2 * s + 1), 0)  # kinh uot -> hoi mo

CORRUPTIONS = {"blur": deg_blur, "dark": deg_dark, "overexpose": deg_overexpose,
               "noise": deg_noise, "rain": deg_rain}

# -------------------------------------------------------------------- metrics
def metrics(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(g, cv2.CV_64F).var()           # cao = net
    sat = np.mean((g < 10) | (g > 245))                  # ti le pixel chet den / chay trang
    hist = np.bincount(g.ravel(), minlength=256) / g.size
    ent = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))  # 0..8 bit
    return {"blur": blur, "saturation": sat, "entropy": ent, "brightness": g.mean()}

# Nguong co dinh -> diem trade-off de thao luan (ngay/dem nen khac nhau)
BLUR_REF = 150.0

def health_score(m):
    blur_s = np.clip(m["blur"] / BLUR_REF, 0, 1)
    exp_s = 0.5 * np.clip(1 - m["saturation"] / 0.4, 0, 1) \
          + 0.5 * np.clip(1 - abs(m["brightness"] - 120) / 110, 0, 1)
    ent_s = np.clip((m["entropy"] - 3) / 4, 0, 1)
    return float(100 * (0.4 * blur_s + 0.35 * exp_s + 0.25 * ent_s))

def status(h):
    if h >= 70: return "OK", (0, 200, 0)
    if h >= 40: return "DEGRADED - down-weight", (0, 200, 255)
    return "UNTRUSTED", (0, 0, 255)

# ------------------------------------------------------------------- detector
class Detector:
    def __init__(self):
        from ultralytics import YOLO
        self.m = YOLO("yolov8n.pt")
    def __call__(self, img):
        r = self.m(img, verbose=False)[0]
        c = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.array([])
        return (float(c.mean()) if len(c) else 0.0), int(len(c)), r

# ------------------------------------------------------------------ benchmark
def load_images(path, max_n=20):
    if os.path.isdir(path):
        files = sorted(sum([glob.glob(os.path.join(path, e)) for e in ("*.jpg", "*.jpeg", "*.png")], []))
        imgs = [(os.path.basename(f), cv2.imread(f)) for f in files[:max_n]]
    else:
        cap, imgs, i = cv2.VideoCapture(path), [], 0
        while len(imgs) < max_n:
            ok, fr = cap.read()
            if not ok: break
            if i % 30 == 0: imgs.append((f"frame{i}", fr))
            i += 1
    out = []
    for n, im in imgs:
        if im is None: continue
        s = 640 / max(im.shape[:2])
        out.append((n, cv2.resize(im, None, fx=s, fy=s)))
    return out

def benchmark(path, use_yolo, outdir="results"):
    os.makedirs(outdir, exist_ok=True)
    imgs = load_images(path)
    if not imgs:
        print("Khong tim thay anh."); return
    det = Detector() if use_yolo else None
    rows = []
    for name, img in imgs:
        cases = [("clean", 0, img)] + [(c, s, f(img, s)) for c, f in CORRUPTIONS.items() for s in range(1, 6)]
        for c, s, im in cases:
            m = metrics(im)
            r = {"image": name, "corruption": c, "level": s, **m, "health": health_score(m)}
            if det:
                r["det_conf"], r["n_det"], _ = det(im)
            rows.append(r)
        print("done", name)

    with open(f"{outdir}/results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

    # Bang tom tat truoc/sau (dua vao slide)
    keys = ["blur", "saturation", "entropy", "health"] + (["det_conf", "n_det"] if det else [])
    groups = {}
    for r in rows: groups.setdefault((r["corruption"], r["level"]), []).append(r)
    summary = []
    print("\n%-11s %3s " % ("corruption", "lv") + " ".join("%10s" % k for k in keys))
    for (c, s), g in groups.items():
        avg = {k: np.mean([x[k] for x in g]) for k in keys}
        summary.append({"corruption": c, "level": s, **avg})
        print("%-11s %3d " % (c, s) + " ".join("%10.3f" % avg[k] for k in keys))
    with open(f"{outdir}/summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary[0].keys()); w.writeheader(); w.writerows(summary)

    if det:
        h = np.array([r["health"] for r in rows]); d = np.array([r["det_conf"] for r in rows])
        print(f"\nPearson(health, detector conf) = {np.corrcoef(h, d)[0, 1]:.3f}")

    # Bieu do
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    clean_h = [x["health"] for x in summary if x["corruption"] == "clean"][0]
    plt.figure(figsize=(7, 4))
    for c in CORRUPTIONS:
        ys = [clean_h] + [x["health"] for x in summary if x["corruption"] == c]
        plt.plot(range(6), ys, marker="o", label=c)
    plt.axhline(70, ls="--", c="g"); plt.axhline(40, ls="--", c="r")
    plt.xlabel("Degradation level (0 = clean)"); plt.ylabel("Health score")
    plt.title("Camera health score vs degradation"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{outdir}/health_vs_level.png", dpi=150)
    if det:
        plt.figure(figsize=(5, 4)); plt.scatter(h, d, s=8, alpha=.5)
        plt.xlabel("Health score"); plt.ylabel("Mean detector confidence")
        plt.title("Health score vs YOLO confidence"); plt.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(f"{outdir}/health_vs_conf.png", dpi=150)

    # Luoi anh mau: moi corruption o level 1, 3, 5
    _, img = imgs[0]; tiles = []
    for c, f in CORRUPTIONS.items():
        row = [img] + [f(img, s) for s in (1, 3, 5)]
        row = [overlay(im.copy(), f"{c} L{s}" if s else "clean") for im, s in zip(row, (0, 1, 3, 5))]
        tiles.append(np.hstack([cv2.resize(t, (320, 240)) for t in row]))
    cv2.imwrite(f"{outdir}/samples_grid.jpg", np.vstack(tiles))
    print(f"\nDa luu ket qua vao thu muc '{outdir}/'")

# ---------------------------------------------------------------------- live
def overlay(img, label="", extra=""):
    m = metrics(img); h = health_score(m); st, col = status(h)
    cv2.rectangle(img, (0, 0), (img.shape[1], 70), (0, 0, 0), -1)
    cv2.putText(img, f"{label}  Health {h:5.1f}  {st}", (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, col, 2)
    cv2.putText(img, f"blur {m['blur']:.0f}  sat {m['saturation']:.2f}  ent {m['entropy']:.2f} {extra}",
                (8, 55), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    return img

def webcam(use_yolo):
    det = Detector() if use_yolo else None
    cap = cv2.VideoCapture(0); mode, lv = None, 3
    keymap = {ord("b"): "blur", ord("d"): "dark", ord("o"): "overexpose",
              ord("n"): "noise", ord("r"): "rain", ord("0"): None}
    while True:
        ok, fr = cap.read()
        if not ok: break
        if mode: fr = CORRUPTIONS[mode](fr, lv)
        extra = ""
        if det:
            conf, n, r = det(fr); fr = r.plot(); extra = f" yolo {conf:.2f} ({n})"
        cv2.imshow("T1 camera health", overlay(fr, f"{mode or 'clean'} L{lv if mode else 0}", extra))
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"): break
        if k in keymap: mode = keymap[k]
        if k in (ord("+"), ord("=")): lv = min(5, lv + 1)
        if k == ord("-"): lv = max(1, lv - 1)
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images"); ap.add_argument("--webcam", action="store_true")
    ap.add_argument("--yolo", action="store_true")
    a = ap.parse_args()
    if a.webcam: webcam(a.yolo)
    elif a.images: benchmark(a.images, a.yolo)
    else: ap.print_help()
