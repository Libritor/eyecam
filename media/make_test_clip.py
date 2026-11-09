"""A synthetic test clip and still for the Mind to Matter gate: a bright creature (body, head,
tail, four walking legs, an eye) crossing a dark textured floor. The still is frame 90.

    python media/make_test_clip.py
"""
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
W, H, FPS, SECS = 960, 540, 30, 6


def frame(i, floor):
    img = cv2.merge([floor, floor, floor]).copy()
    cx = int(200 + (W - 400) * (i / (FPS * SECS)))
    cy = 300 + int(10 * np.sin(i / 4))
    body = np.zeros((H, W), np.uint8)
    cv2.ellipse(body, (cx, cy), (120, 60), 0, 0, 360, 255, -1)
    cv2.circle(body, (cx + 130, cy - 50), 45, 255, -1)
    cv2.ellipse(body, (cx - 150, cy + 10), (70, 22), -20, 0, 360, 255, -1)
    for k, dx in enumerate((-60, -20, 30, 70)):
        ph = int(25 * np.sin(i / 3 + k))
        cv2.rectangle(body, (cx + dx, cy + 40), (cx + dx + 24, cy + 110 + ph), 255, -1)
    shade = np.broadcast_to((np.linspace(0.55, 1.0, W, dtype=np.float32)[None, :] * 255).astype(np.uint8), (H, W))
    col = cv2.merge([np.full((H, W), 60, np.uint8), np.minimum(shade, 230), shade.copy()])
    img[body > 0] = col[body > 0]
    cv2.circle(img, (cx + 150, cy - 60), 8, (10, 10, 10), -1)
    return img


def main():
    rng = np.random.default_rng(3)
    floor = (rng.random((H, W)) * 40 + 20).astype(np.uint8)
    cv2.imwrite(os.path.join(HERE, "test_creature.png"), frame(90, floor))
    print("wrote test_creature.png")
    for four, ext in (("avc1", "mp4"), ("H264", "mp4"), ("VP90", "webm"), ("vp09", "webm"), ("mp4v", "mp4")):
        path = os.path.join(HERE, f"test_creature.{ext}")
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*four), FPS, (W, H))
        if not vw.isOpened():
            print(four, "not available")
            continue
        for i in range(FPS * SECS):
            vw.write(frame(i, floor))
        vw.release()
        if os.path.getsize(path) > 10000:
            print("wrote", os.path.basename(path), "with", four, os.path.getsize(path), "bytes"
                  + ("  (mp4v: browsers cannot play this one; use it only to test the pipeline)" if four == "mp4v" else ""))
            break
        os.remove(path)


if __name__ == "__main__":
    main()
