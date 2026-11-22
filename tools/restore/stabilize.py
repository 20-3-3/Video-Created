"""Stabilize a handheld clip with an L1-optimal camera path.

Follows Grundmann, Kwatra & Essa, "Auto-Directed Video Stabilization with
Robust L1 Optimal Camera Paths" (CVPR 2011), the method behind YouTube's
stabilizer. For every frame t a correction B_t (similarity: zoom, rotation,
shift) maps the crop window onto the input frame. Linear programming picks
the B_t that minimise the L1 norm of the 1st, 2nd and 3rd derivative of the
resulting camera path, so the virtual camera holds still, pans at a constant
speed or eases between the two, while every corner of the crop window stays
inside the input frame (never a black edge).

Two additions for a group selfie:
- proximity: an L1 cost on how far B_t moves the crop from the centre, so the
  path follows the operator's slow reframing (people at the edges stay in
  shot) and only the fast shake is taken out;
- adaptive zoom: the crop starts wide (BASE_ZOOM) and may tighten up to
  BASE_ZOOM / A_MIN where the shake needs the room; tightening is charged,
  and zoom changes are smoothed like rotation, so they are slow.

usage: stabilize.py FILLED.mkv OUT.mkv
writes OUT.mkv (rgb48 FFV1, lossless) and OUT.path.json (motion + solution)
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).parent))
from vio import read_frames, W, H  # noqa: E402
from motion import pair_motion, params  # noqa: E402

CX, CY = (W - 1) / 2, (H - 1) / 2
OUT_W, OUT_H = 1080, 1920      # rendered straight to vertical HD in the same resampling
# light denoise on the native 720p frames (the phone already denoised; this
# takes the chroma speckle and compression shimmer before grain goes on)
DENOISE = "hqdn3d=1.5:3:3:4.5"
# path smoothness weights (Grundmann et al.: w1=10, w2=1, w3=100)
W1, W2, W3 = 10.0, 1.0, 100.0
# weight of the roll parameter against shift (px). Pixel-equivalent would be
# ~450 (corner travel); rolling the horizon is what reads as shake most, and
# in a portrait frame every degree of roll costs ~10 px of the horizontal
# margin, so roll is weighted higher to get the margin first.
C_AB = 2000.0
# zoom is not part of the motion that gets smoothed (frame-to-frame scale from a
# selfie is mostly parallax between face and background, compensating it pumps
# the zoom); it is only the room the path takes, and changes of it are
# weighted on their own. With these weights the crop holds ~1.23 through the
# shaky first 8 s, eases out to ~1.09 for the calm group shot (people at the
# edges stay in) and back in for the walk, each change over ~1.3 s.
C_ZOOM = 2500.0
BASE_ZOOM = 1.08
A_MIN, A_MAX = 1.08 / 1.24, 1.0   # effective zoom between 1.08 and 1.24
B_MAX = np.sin(np.radians(6))
# proximity cost per frame: zoom-in (per unit of a), roll (per unit of b), shift (per px)
LAM = (800.0, 300.0, 0.4, 0.4)
MIN_INLIERS = 30


def estimate(gray):
    ms, info = [], []
    for t in range(1, len(gray)):
        m, _, inf = pair_motion(gray[t - 1], gray[t])
        inf["t"] = t
        if m is None or inf["inliers"] < MIN_INLIERS:
            inf["fallback"] = True
            m = None
        ms.append(m)
        info.append(inf)
    # unreliable pairs: interpolate the motion parameters of their neighbours
    p = np.array([params(m) if m is not None else [np.nan] * 4 for m in ms])
    for k in range(4):
        bad = np.isnan(p[:, k])
        if bad.any():
            p[bad, k] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), p[~bad, k])
    return p, info


def to_matrix(q):
    dx, dy, ang, logs = q
    s = np.exp(logs)
    return np.array([[s * np.cos(ang), -s * np.sin(ang), dx],
                     [s * np.sin(ang), s * np.cos(ang), dy], [0, 0, 1.0]])


def solve(motion_params, zoom=BASE_ZOOM, lam=LAM, a_range=(A_MIN, A_MAX), margin=1.0):
    """motion_params[t-1] = similarity of frame t-1 -> frame t (centred coords).

    Returns per-frame (a, b, x, y) of B_t = [[a, -b, x], [b, a, y]], which maps
    crop-window coordinates (centred, crop size W/zoom x H/zoom) into frame t.
    """
    n = len(motion_params) + 1
    f = [np.linalg.inv(to_matrix(q)) for q in motion_params]  # F_{t+1}: frame t+1 -> frame t
    n1, n2, n3 = n - 1, n - 2, n - 3
    np_ = 4 * n
    ou = np_                      # proximity slacks
    o1 = ou + 4 * n
    o2 = o1 + 4 * n1
    o3 = o2 + 4 * n2
    nv = o3 + 4 * n3

    # residual R_t = F_{t+1} B_{t+1} - B_t as (L_t p + k_t), one 4-row block per t
    rows, cols, vals, kc = [], [], [], np.zeros(4 * n1)
    for t in range(n1):
        fa, fb, fx, fy = f[t][0, 0], f[t][1, 0], f[t][0, 2], f[t][1, 2]
        i, j, r = 4 * t, 4 * (t + 1), 4 * t
        # Ra = fa a' - fb b' - a ; Rb = fb a' + fa b' - b
        # Rx = fa x' - fb y' + fx - x ; Ry = fb x' + fa y' + fy - y
        ent = [(r, j, fa), (r, j + 1, -fb), (r, i, -1),
               (r + 1, j, fb), (r + 1, j + 1, fa), (r + 1, i + 1, -1),
               (r + 2, j + 2, fa), (r + 2, j + 3, -fb), (r + 2, i + 2, -1),
               (r + 3, j + 2, fb), (r + 3, j + 3, fa), (r + 3, i + 3, -1)]
        for a, b, v in ent:
            rows.append(a), cols.append(b), vals.append(v)
        kc[r + 2], kc[r + 3] = fx, fy
    L = sp.csr_matrix((vals, (rows, cols)), shape=(4 * n1, np_))

    def diff(m, k):
        d1 = sp.eye(m.shape[0] - 4, m.shape[0], k=4) - sp.eye(m.shape[0] - 4, m.shape[0])
        return d1 @ m, d1 @ k

    D1, k1 = L, kc
    D2, k2 = diff(D1, k1)
    D3, k3 = diff(D2, k2)

    def pad(m):
        return sp.hstack([m, sp.csr_matrix((m.shape[0], nv - m.shape[1]))])

    def slack(off, cnt):
        return sp.csr_matrix((-np.ones(cnt), (np.arange(cnt), off + np.arange(cnt))), shape=(cnt, nv))

    blocks, rhs = [], []
    for D, k, off, cnt in ((D1, k1, o1, n1), (D2, k2, o2, n2), (D3, k3, o3, n3)):
        Dp, e = pad(D), slack(off, 4 * cnt)
        blocks += [Dp + e, -Dp + e]
        rhs += [-k, k]

    # proximity: u_t >= |p_t - (1, 0, 0, 0)|
    pref = np.tile([1.0, 0, 0, 0], n)
    ident, e = pad(sp.eye(np_)), slack(ou, np_)
    blocks += [ident + e, -ident + e]
    rhs += [pref, -pref]

    # inclusion: every crop corner, mapped by B_t, lands inside the input frame
    cw, ch = CX / zoom, CY / zoom
    xb, yb = CX - margin, CY - margin
    rows, cols, vals, rhs_inc, r = [], [], [], [], 0
    for t in range(n):
        a, b, x, y = 4 * t, 4 * t + 1, 4 * t + 2, 4 * t + 3
        for sx in (-1, 1):
            for sy in (-1, 1):
                px, py = sx * cw, sy * ch
                for sign in (1, -1):
                    rows += [r] * 3; cols += [a, b, x]; vals += [sign * px, -sign * py, sign]  # X
                    rhs_inc.append(xb); r += 1
                    rows += [r] * 3; cols += [b, a, y]; vals += [sign * px, sign * py, sign]   # Y
                    rhs_inc.append(yb); r += 1
    blocks.append(sp.csr_matrix((vals, (rows, cols)), shape=(r, nv)))
    rhs.append(np.array(rhs_inc))

    A = sp.vstack(blocks).tocsr()
    bvec = np.concatenate(rhs)
    wpar = np.array([C_ZOOM, C_AB, 1.0, 1.0])
    c = np.zeros(nv)
    c[ou:o1] = np.tile(lam, n)
    c[o1:o2] = W1 * np.tile(wpar, n1)
    c[o2:o3] = W2 * np.tile(wpar, n2)
    c[o3:] = W3 * np.tile(wpar, n3)
    bounds = [a_range, (-B_MAX, B_MAX), (None, None), (None, None)] * n + [(0, None)] * (nv - np_)
    res = linprog(c, A_ub=A, b_ub=bvec, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    return res.x[:np_].reshape(n, 4), res.fun


def warp_matrix(pt, zoom=BASE_ZOOM, out_w=OUT_W, out_h=OUT_H):
    """output pixel -> input pixel (2x3), for cv2.warpAffine with WARP_INVERSE_MAP."""
    a, b, x, y = pt
    bm = np.array([[a, -b, x], [b, a, y], [0, 0, 1.0]])
    c_in = np.array([[1, 0, CX], [0, 1, CY], [0, 0, 1.0]])
    c_out = np.array([[1, 0, (out_w - 1) / 2], [0, 1, (out_h - 1) / 2], [0, 0, 1.0]])
    s = np.diag([W / out_w / zoom, H / out_h / zoom, 1.0])
    return (c_in @ bm @ s @ np.linalg.inv(c_out))[:2]


def render(frames16, path, dst, zoom=BASE_ZOOM):
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb48le",
                            "-s", f"{OUT_W}x{OUT_H}", "-r", "24", "-i", "-", "-c:v", "ffv1", "-level", "3",
                            "-pix_fmt", "rgb48le", dst], stdin=subprocess.PIPE)
    for fr, pt in zip(frames16, path):
        out = cv2.warpAffine(fr, warp_matrix(pt, zoom), (OUT_W, OUT_H),
                             flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_REFLECT)
        enc.stdin.write(out.tobytes())
    enc.stdin.close()
    assert enc.wait() == 0


def main(src, dst):
    frames16 = read_frames(src, pix_fmt="rgb48le", pre=DENOISE)
    gray = [cv2.cvtColor((f >> 8).astype(np.uint8), cv2.COLOR_RGB2GRAY) for f in frames16]
    mp, info = estimate(gray)
    mp[:, 3] = 0.0  # smooth shift and roll only (see C_ZOOM)
    path, cost = solve(mp)
    zooms = BASE_ZOOM / path[:, 0]
    print(f"{len(frames16)} frames, zoom {zooms.min():.3f}-{zooms.max():.3f} (mean {zooms.mean():.3f}), "
          f"path cost {cost:.0f}, fallback pairs {[i['t'] for i in info if i.get('fallback')]}")
    json.dump({"base_zoom": BASE_ZOOM, "motion": mp.tolist(), "path": path.tolist(), "info": info},
              open(Path(dst).with_suffix(".path.json"), "w"))
    render(frames16, path, dst)


if __name__ == "__main__":
    main(*sys.argv[1:3])
