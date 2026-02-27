import json, time, cv2

from ball_detector import detect_ball, draw_ball
from qr_depth import QRDepth, focal_length, pixel_to_xyz
from tracker import BallTracker

OUTPUT = 'positions.json'
REDETECT_EVERY = 5   # frames before re-scanning when lost


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError('Cannot open webcam')

    # Warm up
    for _ in range(30):
        cap.read()
    time.sleep(0.1)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError('No frame from webcam')

    h, w = frame.shape[:2]
    f_px = focal_length(w)
    cx, cy = w / 2.0, h / 2.0
    print(f'Frame {w}x{h}  |  focal ~{f_px:.0f}px')

    # Auto-detect ball on first frame
    tracker = BallTracker()
    det = detect_ball(frame)
    if det:
        tracker.init(frame, det['bbox'])
        print(f'Ball found at {det["center"]}')
    else:
        print('No ball yet — will keep searching.')

    qr = QRDepth()
    lost_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # QR depth
        qr.update(frame, f_px)
        qr.draw(frame)

        # Track
        if tracker.ok or tracker.bbox is not None:
            ok, _ = tracker.update(frame)
        else:
            ok = False

        # Re-detect if lost
        if not ok:
            lost_count += 1
            if lost_count >= REDETECT_EVERY or tracker.bbox is None:
                det = detect_ball(frame)
                if det:
                    tracker.init(frame, det['bbox'])
                    lost_count = 0
                    print(f'Ball re-detected at {det["center"]}')
                else:
                    cv2.putText(frame, 'Searching...', (10, h - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            lost_count = 0

        # Visualise detection circle
        draw_ball(frame, detect_ball(frame))

        # XYZ output
        pos = None
        c = tracker.center
        if c and tracker.ok:
            if qr.depth_z is not None:
                xyz = pixel_to_xyz(c[0], c[1], qr.depth_z, f_px, cx, cy)
                label = f'X:{xyz[0]} Y:{xyz[1]} Z:{xyz[2]}'
                pos = {'pixel': list(c), 'xyz_mm': list(xyz)}
            else:
                label = 'No QR depth'
                pos = {'pixel': list(c), 'xyz_mm': None}
            tracker.draw(frame, label)
        elif tracker.bbox is not None:
            tracker.draw(frame, 'LOST')

        if pos:
            print(json.dumps(pos))
            try:
                with open(OUTPUT, 'w') as f:
                    json.dump(pos, f)
            except OSError:
                pass

        cv2.imshow('V3D Tracking', frame)
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
