import cv2
import numpy as np
import random
import time
import json

QR_SIDE_MM = 50.0        # Physical side length of the reference QR code (mm)
OUTPUT_FILE = 'positions.json'  # File where XYZ positions are written each frame

# QR depth
qr_detector = cv2.QRCodeDetector()

def detect_qr(frame):
    ok, points = qr_detector.detect(frame)
    if ok and points is not None:
        pts = points[0]  # shape (4, 2)
        return pts
    return None

def qr_side_pixels(pts):
    sides = [np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)]
    return float(np.mean(sides))

def estimate_focal_length(frame_width, hfov_deg=60.0):
    return frame_width / (2.0 * np.tan(np.radians(hfov_deg / 2.0)))

def compute_xyz(pixel_x, pixel_y, depth_z, focal_px, cx, cy):
    x = (pixel_x - cx) * depth_z / focal_px
    y = (pixel_y - cy) * depth_z / focal_px
    return round(x, 1), round(y, 1), round(depth_z, 1)

clicked_points = []

def click_center(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError('Cannot open webcam (index 0)')

# Warm up camera and get a valid first frame
for _ in range(30):
    cap.read()
time.sleep(0.1)
ret, first = cap.read()
if not ret or first is None:
    for i in range(10):
        ret, first = cap.read()
        if ret:
            break
        time.sleep(0.1)
if not ret:
    cap.release()
    raise RuntimeError('Failed to read frame from webcam')

cv2.namedWindow('Selecciona objetos', cv2.WINDOW_NORMAL)
cv2.imshow('Selecciona objetos', first)

# Permitir al usuario seleccionar múltiples objetos haciendo clic en el centro de cada uno
clicked_points.clear()
cv2.setMouseCallback('Selecciona objetos', click_center)
print("Click centers of objects, press 'c' to confirm, 'r' to reset, Esc to cancel.")
while True:
    disp = first.copy()
    for p in clicked_points:
        cx, cy = p
        box_sz = 40
        x1 = max(0, cx - box_sz//2)
        y1 = max(0, cy - box_sz//2)
        x2 = min(first.shape[1]-1, cx + box_sz//2)
        y2 = min(first.shape[0]-1, cy + box_sz//2)
        cv2.rectangle(disp, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.circle(disp, p, 3, (0,255,0), -1)
    cv2.imshow('Selecciona objetos', disp)
    k = cv2.waitKey(30) & 0xFF
    if k == ord('c'):
        break
    elif k == ord('r'):
        clicked_points.clear()
    elif k == 27:
        cap.release()
        cv2.destroyAllWindows()
        raise SystemExit('User canceled')

# Crear regiones de interés (ROIs) alrededor de los puntos seleccionados
box_size = 80
rois = []
h, w = first.shape[:2]
for (cx, cy) in clicked_points:
    x = max(0, int(cx - box_size//2))
    y = max(0, int(cy - box_size//2))
    x2 = min(w, x + box_size)
    y2 = min(h, y + box_size)
    rois.append((x, y, x2 - x, y2 - y))

if not rois:
    print('No objects selected, exiting.')
    cap.release()
    cv2.destroyAllWindows()
    raise SystemExit

# Create MultiTracker (handle different OpenCV versions).
# If OpenCV's MultiTracker API is missing, use a small fallback implementation.
class FallbackMultiTracker:
    def __init__(self):
        self.trackers = []
    def add(self, tracker, *args):
        # Accept (tracker, frame, bbox) or (tracker, bbox)
        if len(args) == 2:
            frame, bbox = args
            try:
                tracker.init(frame, bbox)
            except Exception:
                try:
                    tracker.init(frame, tuple(bbox))
                except Exception:
                    pass
        elif len(args) == 1:
            bbox = args[0]
            try:
                tracker.init(None, bbox)
            except Exception:
                pass
        self.trackers.append(tracker)
    def update(self, frame):
        boxes = []
        oks = []
        for tr in self.trackers:
            try:
                res = tr.update(frame)
                if isinstance(res, tuple) and len(res) == 2:
                    ok, box = res
                else:
                    ok = True
                    box = res
                boxes.append(box)
                oks.append(bool(ok))
            except Exception:
                boxes.append((0,0,0,0))
                oks.append(False)
        return all(oks), boxes

# Try to obtain a MultiTracker from cv2, otherwise use fallback.
multi = None
try:
    multi = cv2.legacy.MultiTracker_create()
except Exception:
    try:
        multi = cv2.MultiTracker_create()
    except Exception:
        try:
            multi = cv2.legacy.MultiTracker()
        except Exception:
            multi = FallbackMultiTracker()

# Añadir tracker a cada ROI y asignarle un color
def _create_tracker():
    creators = [
        lambda: cv2.legacy.TrackerCSRT_create(),
        lambda: cv2.TrackerCSRT_create(),
        lambda: cv2.TrackerCSRT.create(),
    ]

    for create in creators:
        try:
            tracker = create()
            print(f"Tracker created: {type(tracker).__name__}")
            return tracker
        except AttributeError:
            continue


colors = []
for r in rois:
    tr = _create_tracker()
    if tr is None:
        print('Failed to create a tracker for ROI, skipping this object.')
        continue
    # Try common add/init signatures
    try:
        multi.add(tr, first, r)
    except Exception:
        try:
            multi.add(tr, r)
        except Exception as e:
            try:
                tr.init(first, r)
            except Exception:
                print('Failed to add/init tracker:', e)
    colors.append((random.randint(50,255), random.randint(50,255), random.randint(50,255)))

cv2.destroyWindow('Selecciona objetos')

# ─── Prepare depth-mapping state ─────────────────────────────────────────────
frame_h, frame_w = first.shape[:2]
focal_px = estimate_focal_length(frame_w)  # estimated from ~60° HFOV
cx, cy = frame_w / 2.0, frame_h / 2.0      # principal point (image centre)
last_depth_z = None  # most recent depth estimate from QR (mm)

print(f'Focal length estimate: {focal_px:.1f} px  |  Frame: {frame_w}x{frame_h}')
print(f'QR reference size: {QR_SIDE_MM} mm  |  Output → {OUTPUT_FILE}')
print('Tracking started. Press Esc to stop.\n')

# ─── Tracking + XYZ estimation loop ──────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        print('Frame read failed, stopping.')
        break

    # --- QR code depth update --------------------------------------------------
    qr_pts = detect_qr(frame)
    if qr_pts is not None:
        side_px = qr_side_pixels(qr_pts)
        if side_px > 0:
            # Z = (known_size_mm * focal_px) / apparent_size_px
            last_depth_z = (QR_SIDE_MM * focal_px) / side_px
        # Draw QR outline
        for j in range(4):
            p1 = tuple(qr_pts[j].astype(int))
            p2 = tuple(qr_pts[(j + 1) % 4].astype(int))
            cv2.line(frame, p1, p2, (255, 0, 255), 2)
        if last_depth_z is not None:
            qr_cx, qr_cy = np.mean(qr_pts, axis=0).astype(int)
            cv2.putText(frame, f'Z:{last_depth_z:.0f}mm', (qr_cx, qr_cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    # --- Update trackers -------------------------------------------------------
    ok, boxes = multi.update(frame)

    frame_positions = []  # collect XYZ for all objects this frame

    for i, b in enumerate(boxes):
        bx, by, bw, bh = [int(v) for v in b]
        # Bounding-box centre in pixels
        obj_cx = bx + bw // 2
        obj_cy = by + bh // 2

        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), colors[i], 2)

        if last_depth_z is not None:
            x_mm, y_mm, z_mm = compute_xyz(obj_cx, obj_cy, last_depth_z,
                                           focal_px, cx, cy)
            label = f'ID:{i+1} X:{x_mm} Y:{y_mm} Z:{z_mm}'
            frame_positions.append({
                'id': i + 1,
                'pixel': [obj_cx, obj_cy],
                'xyz_mm': [x_mm, y_mm, z_mm]
            })
        else:
            label = f'ID:{i+1} (no QR → no depth)'
            frame_positions.append({
                'id': i + 1,
                'pixel': [obj_cx, obj_cy],
                'xyz_mm': None
            })

        cv2.putText(frame, label, (bx, by - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)

    # --- Output positions (console + file) ------------------------------------
    if frame_positions:
        payload = json.dumps(frame_positions)
        print(payload)
        try:
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(frame_positions, f)
        except OSError:
            pass

    cv2.imshow('Tracking', frame)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
