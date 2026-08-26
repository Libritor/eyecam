"""Guided-cursor flicker stimulus (the eye-tracker-free mode of the paper).

A square cursor with a white fixation dot slides along serpentine scan lines.
Over bright parts of the target it flickers yellow/blue at STIM_FREQ_HZ
(2 frames on / 2 frames off at 60 Hz); over dark parts it stays dark. Cursor
position is logged every frame with the LSL clock so it aligns with the EEG
stream recorded by osc_acquire.py / acquire.py.

Controls during a scan: SPACE pauses at the next row boundary (also triggered
by losing window focus), ESC aborts keeping the partial log.

Standalone:  run the recorder first (other terminal), then
    python stimulus.py --target "text:NO" --session runs/live1
run_session.py imports run_raster() instead.
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


def serpentine_cells(grid_w, grid_h):
    cells = []
    for gy in range(grid_h):
        xs = range(grid_w) if gy % 2 == 0 else range(grid_w - 1, -1, -1)
        cells.extend((gx, gy) for gx in xs)
    return cells


def _pause_screen(screen):
    """Blocking PAUSED overlay. Returns True to resume, False to abort."""
    font = pygame.font.SysFont(None, 48)
    msg = font.render("PAUSED  -  SPACE to resume, ESC to abort", True,
                      (200, 200, 200))
    screen.fill((0, 0, 0))
    screen.blit(msg, msg.get_rect(center=screen.get_rect().center))
    pygame.display.flip()
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    return False
                if ev.key == pygame.K_SPACE:
                    return True
        pygame.time.wait(50)


def run_raster(screen, target, seconds_per_cell, log_path, controls=None,
               auto=False):
    """Run the raster scan on an existing pygame screen.

    controls (optional) may provide on_frame(), on_row_end(gy) and
    should_pause(). auto=True never blocks on the interactive PAUSED
    overlay (unattended runs). Returns "completed" or "aborted"; the log
    is kept either way.
    """
    grid_h, grid_w = target.shape
    W, H = screen.get_size()
    clock = pygame.time.Clock()

    margin = 0.08
    cell_w = W * (1 - 2 * margin) / grid_w
    cell_h = H * (1 - 2 * margin) / grid_h
    x0, y0 = W * margin, H * margin

    frames_per_cell = max(1, int(round(seconds_per_cell * config.DISPLAY_FPS)))
    half_period = max(1, int(round(config.DISPLAY_FPS / config.STIM_FREQ_HZ / 2)))

    cells = serpentine_cells(grid_w, grid_h)
    status = "completed"
    frame = 0
    slow_frames = 0
    pause_requested = False

    # buffering=1: rows must reach live tailers (phantom subject, runner
    # watchdog) within a frame, not sit in an 8 KB stdio buffer.
    f = open(log_path, "w", newline="", buffering=1)
    w = csv.writer(f)
    w.writerow(["time", "grid_x", "grid_y", "luminance", "flicker_on"])
    try:
        for idx, (gx, gy) in enumerate(cells):
            row_start = idx > 0 and cells[idx - 1][1] != gy
            if row_start:
                if controls is not None:
                    controls.on_row_end(cells[idx - 1][1])
                if pause_requested or (controls is not None
                                       and controls.should_pause()):
                    pause_requested = False
                    if not auto and not _pause_screen(screen):
                        status = "aborted"
                        break
            lum = float(target[gy, gx])
            aborted = False
            for _ in range(frames_per_cell):
                if controls is not None:
                    controls.on_frame()
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT or (
                            ev.type == pygame.KEYDOWN
                            and ev.key == pygame.K_ESCAPE):
                        aborted = True
                    elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_SPACE:
                        pause_requested = True
                    elif ev.type == pygame.WINDOWFOCUSLOST:
                        pause_requested = True
                if aborted:
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
                pygame.draw.circle(screen, (255, 255, 255),
                                   (rect.centerx, rect.centery), 4)
                pygame.display.flip()

                w.writerow([f"{local_clock():.6f}", gx, gy, f"{lum:.4f}",
                            1 if flicker_on else 0])
                frame += 1
                if clock.tick(config.DISPLAY_FPS) > 18:
                    slow_frames += 1
            if aborted:
                status = "aborted"
                break
    finally:
        f.close()

    if frame and slow_frames / frame > 0.02:
        print(f"WARNING: {slow_frames}/{frame} frames missed vsync (>18 ms) — "
              "flicker spectral purity degraded")
    return status


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
    np.save(os.path.join(args.session, "target.npy"), target)

    pygame.init()
    flags = 0 if args.windowed else pygame.FULLSCREEN
    screen = pygame.display.set_mode(
        (1280, 720) if args.windowed else (0, 0), flags, vsync=1)

    grid_h, grid_w = target.shape
    total = grid_w * grid_h * args.seconds_per_cell
    print(f"scan: {grid_w}x{grid_h} cells, {args.seconds_per_cell:.1f} s each "
          f"-> {total/60:.1f} min. SPACE pauses at row end, ESC aborts.")

    log_path = os.path.join(args.session, "cursor_log.csv")
    status = run_raster(screen, target, args.seconds_per_cell, log_path)
    pygame.quit()
    print(f"{status}; cursor log -> {log_path}")


if __name__ == "__main__":
    main()
