"""Stabilization v2: level horizon, smoother path, synthesized edges, 2K render.

1. motion between consecutive 720p frames (motion.py, as in stabilize.py);
2. absolute roll of every frame from the scene's vertical lines, fused with
   the motion into a drift-free roll track (path2.roll_track);
3. L1-optimal path that keeps the horizon level, may let the crop leave the
   frame by up to 10 % per side (path2.solve2);
4. render from the super-resolved frames (upscale.py) to 1440x2560, filling
   everything outside the frame from neighbouring frames (render2.py).

usage: stabilize2.py FILLED.mkv SR.mkv OUT.mkv
writes OUT.mkv (rgb48 FFV1, lossless, 24 fps) and OUT.path.json
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from vio import read_frames  # noqa: E402
from stabilize import estimate  # noqa: E402
import path2  # noqa: E402
import render2  # noqa: E402

OUT_W, OUT_H, SCALE = 1440, 2560, 2


def frames_from(path, w, h):
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb48le", "-"],
                           stdout=subprocess.PIPE)
    size = w * h * 6
    while True:
        buf = dec.stdout.read(size)
        if len(buf) < size:
            break
        yield np.frombuffer(buf, np.uint16).reshape(h, w, 3).astype(np.float32) / 65535
    dec.wait()


def main(filled, sr, dst):
    t0 = time.time()
    rgb = read_frames(filled)
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in rgb]
    del rgb
    motion, info = estimate(gray)
    meas = path2.measure_roll(gray)
    roll = path2.roll_track(meas, motion[:, 2])
    q = motion.copy()
    q[:, 2] = np.radians(np.diff(roll))      # rotation from the drift-free track
    q[:, 3] = 0.0                            # zoom is the path's own (see stabilize.C_ZOOM)
    path, ext = path2.solve2(q, roll)
    n = len(gray)
    phi = np.degrees(np.arctan2(path[:, 1], path[:, 0]))
    print(f"path: {n} frames, horizon within {np.abs(phi - roll).max():.1f} deg of level, "
          f"zoom {path2.BASE_ZOOM / path[:, 0].max():.3f}-{path2.BASE_ZOOM / path[:, 0].min():.3f}, "
          f"synthesized edge in {np.mean(ext.max(1) > 1) * 100:.0f}% of frames ({time.time() - t0:.0f} s)",
          flush=True)
    json.dump({"motion": q.tolist(), "roll": roll.tolist(), "roll_measured": meas.tolist(),
               "path": path.tolist(), "edges": ext.tolist()},
              open(Path(dst).with_suffix(".path.json"), "w"))

    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb48le",
                            "-s", f"{OUT_W}x{OUT_H}", "-r", "24", "-i", "-", "-c:v", "ffv1", "-level", "3",
                            "-slices", "16", "-pix_fmt", "rgb48le", dst], stdin=subprocess.PIPE)
    src = frames_from(sr, render2.W * SCALE, render2.H * SCALE)
    window, loaded = {}, 0
    stats = []
    for t in range(n):
        while loaded < min(n, t + render2.K + 1):
            window[loaded] = next(src)
            loaded += 1
        for k in [k for k in window if k < t - render2.K]:
            del window[k]
        out, inf = render2.render_frame(t, window, {k: gray[k] for k in window}, path[t],
                                        path2.BASE_ZOOM, OUT_W, OUT_H, SCALE)
        enc.stdin.write((np.clip(out, 0, 1) * 65535 + 0.5).astype(np.uint16).tobytes())
        stats.append((inf["missing"], len(inf["used"]), inf["uncovered"]))
        if t % 24 == 0:
            print(f"frame {t}: synthesized {inf['missing'] * 100:.1f}% from {len(inf['used'])} neighbours, "
                  f"{(time.time() - t0) / (t + 1):.1f} s/frame", flush=True)
    enc.stdin.close()
    assert enc.wait() == 0
    s = np.array(stats)
    print(f"{dst}: {n} frames; synthesized area mean {s[:, 0].mean() * 100:.1f}% max {s[:, 0].max() * 100:.1f}%, "
          f"inpainted (no neighbour) max {s[:, 2].max() * 100:.2f}%, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(*sys.argv[1:4])
