import cv2
import numpy as np

MIN_RADIUS = 10


def detect_ball(frame):
    """Return {'center', 'radius', 'bbox'} for the biggest circular blob, or None."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    _, bw = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)

    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perim = cv2.arcLength(c, True)
    if perim == 0 or area < np.pi * MIN_RADIUS ** 2:
        return None
    if 4 * np.pi * area / perim ** 2 < 0.4:          # circularity check
        return None

    (cx, cy), r = cv2.minEnclosingCircle(c)
    if r < MIN_RADIUS:
        return None

    return {'center': (int(cx), int(cy)), 'radius': int(r), 'bbox': cv2.boundingRect(c)}


def draw_ball(frame, det, color=(0, 255, 0)):
    if det:
        cv2.circle(frame, det['center'], det['radius'], color, 2)
        cv2.circle(frame, det['center'], 3, color, -1)
