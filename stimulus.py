"""Guided-cursor flicker stimulus (the eye-tracker-free mode of the paper).

A square cursor with a white fixation dot slides along serpentine scan lines.
Over bright parts of the target it flickers yellow/blue at STIM_FREQ_HZ
(2 frames on / 2 frames off at 60 Hz); over dark parts it stays dark. Cursor
position is logged every frame with the LSL clock so it aligns with the EEG
stream recorded by acquire.py.

Run acquire.py FIRST (in another terminal), then:
    python stimulus.py --target "text:NO" --session runs/live1
Press ESC to abort.
"""

import argparse
import csv
import os
import time

import numpy as np
import pygame

import config
import targets

try:
    from pylsl import local_clock
except ImportError:
    local_clock = time.perf_counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="text:NO")
    ap.add_argument("--session", default=os.path.join("runs", "live1"))
    ap.add_argument("--seconds-per-cell", type=float, default=4.0,
                    help="dwell per scan position (paper used ~6.7 s)")
    ap.add_argument("--windowed", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.session, exist_ok=True)
    target = targets.load_target(args.target)
    grid_h, grid_w = target.shape
    np.save(os.path.join(args.session, "target.npy"), target)

    pygame.init()
    flags = 0 if args.windowed else pygame.FULLSCREEN
    screen = pygame.display.set_mode((0, 0) if not args.windowed else (1280, 720),
                                     flags, vsync=1)
    W, H = screen.get_size()
    clock = pygame.time.Clock()

    # Map the scan grid onto the screen with margins.
    margin = 0.08
    cell_w = W * (1 - 2 * margin) / grid_w
    cell_h = H * (1 - 2 * margin) / grid_h
    x0, y0 = W * margin, H * margin

    frames_per_cell = max(1, int(round(args.seconds_per_cell * config.DISPLAY_FPS)))
    half_period = max(1, int(round(config.DISPLAY_FPS / config.STIM_FREQ_HZ / 2)))

    log_path = os.path.join(args.session, "cursor_log.csv")
    f = open(log_path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["time", "grid_x", "grid_y", "luminance", "flicker_on"])

    cells = []
    for gy in range(grid_h):
        xs = range(grid_w) if gy % 2 == 0 else range(grid_w - 1, -1, -1)
        cells.extend((gx, gy) for gx in xs)

    total = len(cells) * args.seconds_per_cell
    print(f"scan: {grid_w}x{grid_h} cells, {args.seconds_per_cell:.1f} s each "
          f"-> {total/60:.1f} min. ESC to abort.")

    frame = 0
    running = True
    for gx, gy in cells:
        if not running:
            break
        lum = float(target[gy, gx])
        for _ in range(frames_per_cell):
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (
                        ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                    running = False
            if not running:
                break

            flicker_on = (frame // half_period) % 2 == 0 and lum > 0.05
            screen.fill((0, 0, 0))
            cx = x0 + gx * cell_w
            cy = y0 + gy * cell_h
            rect = pygame.Rect(int(cx), int(cy),
                               max(int(cell_w), config.CURSOR_PX),
                               max(int(cell_h), config.CURSOR_PX))
            if lum > 0.05:
                color = ((255, 255, 0) if flicker_on else (0, 0, 255))
                shade = tuple(int(c * (0.3 + 0.7 * lum)) for c in color)
                pygame.draw.rect(screen, shade, rect)
            dot = (rect.centerx, rect.centery)
            pygame.draw.circle(screen, (255, 255, 255), dot, 4)
            pygame.display.flip()

            w.writerow([f"{local_clock():.6f}", gx, gy, f"{lum:.4f}",
                        1 if flicker_on else 0])
            frame += 1
            clock.tick(config.DISPLAY_FPS)

    f.close()
    pygame.quit()
    print(f"cursor log -> {log_path}")
    if not running:
        print("aborted early; partial log kept")


if __name__ == "__main__":
    main()
