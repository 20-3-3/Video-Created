"""Measure shake: RMS of the camera path's high-frequency part (above ~1 Hz).

usage: jitter.py VIDEO [VIDEO ...]
Every video is measured at 720x1280 (resized if needed) so pixel figures
compare; px = shift of the frame centre, deg = roll, % = zoom.
"""
import sys
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
sys.path.insert(0, str(Path(__file__).parent))
from vio import read_frames  # noqa: E402
from motion import pair_motion, params  # noqa: E402


def path_params(path):
    fr = read_frames(path, resize=True)
    g = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in fr]
    p = []
    for t in range(1, len(g)):
        m, _, inf = pair_motion(g[t - 1], g[t])
        p.append(params(m) if (m is not None and inf["inliers"] >= 30) else [np.nan] * 4)
    p = np.array(p)
    for k in range(4):
        bad = np.isnan(p[:, k])
        p[bad, k] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), p[~bad, k])
    return p


def shake(p, segs, fps=24, sigma=6):
    c = np.cumsum(p, axis=0)
    hp = c - gaussian_filter1d(c, sigma, axis=0, mode="nearest")
    out = {}
    for name, (a, b) in segs.items():
        s = slice(int(a * fps), min(int(b * fps), len(hp)))
        out[name] = (np.sqrt(np.mean(hp[s, 0] ** 2 + hp[s, 1] ** 2)),
                     np.degrees(np.sqrt(np.mean(hp[s, 2] ** 2))), 100 * np.sqrt(np.mean(hp[s, 3] ** 2)))
    return out


if __name__ == "__main__":
    segs = {"whole": (0, 18), "0-3s shaky": (0, 3), "3-7s": (3, 7), "7-8.5s swing": (7, 8.5),
            "8.5-12.5s group": (8.5, 12.5), "12.5-17.5s walk": (12.5, 17.5)}
    res = {path: shake(path_params(path), segs) for path in sys.argv[1:]}
    for name in segs:
        print(f"{name:18s} " + " | ".join(
            f"{Path(pth).stem}: {res[pth][name][0]:5.1f}px {res[pth][name][1]:4.2f}deg {res[pth][name][2]:4.2f}%"
            for pth in res))
