"""Camera path v2: level horizon, smoother motion, room for synthesized edges.

Same L1-optimal path as stabilize.py (Grundmann et al. 2011), with:
- an absolute roll track: lines in the scene that are vertical (towers, lamp
  posts, standing people; LSD segments within 20 deg of vertical, median
  weighted by length) give the true roll of each frame but are noisy; the
  frame-to-frame motion is precise but drifts (in the walk it follows the
  face, not the street). A complementary filter takes the slow part from
  the verticals and the fast part from the motion. The track replaces the
  rotation of the motion, and the path is pulled towards a level horizon;
- synthesized edges: each side of the crop window may leave the frame by up
  to EXT of the frame size, charged per pixel per frame; render2.py fills
  what is outside from neighbouring frames.

usage: from path2 import roll_track, solve2
"""
import cv2
import numpy as np
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linprog

W, H = 720, 1280
CX, CY = (W - 1) / 2, (H - 1) / 2
W1, W2, W3 = 10.0, 1.0, 100.0
C_ROLL, C_ZOOM = 2000.0, 2500.0
BASE_ZOOM, A_MIN, A_MAX = 1.08, 1.08 / 1.24, 1.0
B_MAX = np.sin(np.radians(20))
LAM = (800.0, 2500.0, 0.4, 0.4)   # zoom-in, roll away from level, shift (px)
EXT = (0.10, 0.10)                 # max synthesized edge: share of width, of height
MU = 0.5                           # cost per px of synthesized edge per frame


def measure_roll(gray, min_len=35, max_dev=20):
    """Per frame: weighted median deviation of near-vertical lines (deg), total length."""
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    out = np.full((len(gray), 2), np.nan)
    for t, g in enumerate(gray):
        lines = lsd.detect(g)[0]
        if lines is None:
            continue
        l = lines.reshape(-1, 4)
        dx, dy = l[:, 2] - l[:, 0], l[:, 3] - l[:, 1]
        s = np.where(dy < 0, -1, 1)
        dx, dy = dx * s, dy * s
        ln, dev = np.hypot(dx, dy), np.degrees(np.arctan2(-dx, dy))
        m = (ln >= min_len) & (np.abs(dev) <= max_dev)
        if m.sum() < 3:
            continue
        o = np.argsort(dev[m])
        cw = np.cumsum(ln[m][o])
        out[t] = dev[m][o][np.searchsorted(cw, cw[-1] / 2)], ln[m].sum()
    return out


def roll_track(meas, motion_rot, sigma=12):
    """Complementary filter: low frequencies from `meas` (deg), high from the motion chain."""
    chain = np.degrees(np.concatenate([[0], np.cumsum(motion_rot)]))
    r, conf = meas[:, 0], meas[:, 1]
    w0 = np.where(np.isfinite(r), np.clip(np.nan_to_num(conf) / 1500, 0.05, 1), 0.0)
    rz = np.nan_to_num(r)
    hp = chain - gaussian_filter1d(chain, sigma, mode="nearest")
    w = w0.copy()
    for _ in range(5):                       # down-weight measurements far from the track
        lp = gaussian_filter1d(rz * w, sigma, mode="nearest") / np.maximum(
            gaussian_filter1d(w, sigma, mode="nearest"), 1e-9)
        track = lp + hp
        w = w0 / (1 + (np.abs(rz - track) / 3.0) ** 2)
    return track


def to_matrix(q):
    dx, dy, ang, logs = q
    s = np.exp(logs)
    return np.array([[s * np.cos(ang), -s * np.sin(ang), dx],
                     [s * np.sin(ang), s * np.cos(ang), dy], [0, 0, 1.0]])


def solve2(motion, roll_deg, zoom=BASE_ZOOM, lam=LAM, ext=EXT, mu=MU, w=(W1, W2, W3), margin=1.0):
    """motion[t-1] = (dx, dy, rot, logscale) of frame t-1 -> t, rot already from the roll track.

    Returns path (n x 4: a, b, x, y of B_t, crop -> frame t, centred coords) and the
    synthesized-edge depth used per frame (n x 4: left, right, top, bottom, px).
    """
    n = len(motion) + 1
    f = [np.linalg.inv(to_matrix(q)) for q in motion]
    n1, n2, n3 = n - 1, n - 2, n - 3
    np_ = 4 * n
    ou, oe = np_, 2 * np_                      # proximity slacks, edge slacks
    o1 = oe + 4 * n
    o2, o3 = o1 + 4 * n1, o1 + 4 * n1 + 4 * n2
    nv = o3 + 4 * n3

    rows, cols, vals, kc = [], [], [], np.zeros(4 * n1)
    for t in range(n1):
        fa, fb, fx, fy = f[t][0, 0], f[t][1, 0], f[t][0, 2], f[t][1, 2]
        i, j, r = 4 * t, 4 * (t + 1), 4 * t
        for a_, b_, v in [(r, j, fa), (r, j + 1, -fb), (r, i, -1), (r + 1, j, fb), (r + 1, j + 1, fa),
                          (r + 1, i + 1, -1), (r + 2, j + 2, fa), (r + 2, j + 3, -fb), (r + 2, i + 2, -1),
                          (r + 3, j + 2, fb), (r + 3, j + 3, fa), (r + 3, i + 3, -1)]:
            rows.append(a_), cols.append(b_), vals.append(v)
        kc[r + 2], kc[r + 3] = fx, fy
    L = sp.csr_matrix((vals, (rows, cols)), shape=(4 * n1, np_))

    def diff(m, k):
        d1 = sp.eye(m.shape[0] - 4, m.shape[0], k=4) - sp.eye(m.shape[0] - 4, m.shape[0])
        return d1 @ m, d1 @ k

    def pad(m):
        return sp.hstack([m, sp.csr_matrix((m.shape[0], nv - m.shape[1]))])

    def slack(off, cnt):
        return sp.csr_matrix((-np.ones(cnt), (np.arange(cnt), off + np.arange(cnt))), shape=(cnt, nv))

    D2, k2 = diff(L, kc)
    D3, k3 = diff(D2, k2)
    blocks, rhs = [], []
    for D, k, off, cnt in ((L, kc, o1, n1), (D2, k2, o2, n2), (D3, k3, o3, n3)):
        Dp, e = pad(D), slack(off, 4 * cnt)
        blocks += [Dp + e, -Dp + e]
        rhs += [-k, k]

    # proximity to (1, level roll, 0, 0)
    pref = np.stack([np.ones(n), np.sin(np.radians(roll_deg)), np.zeros(n), np.zeros(n)], 1).ravel()
    ident, e = pad(sp.eye(np_)), slack(ou, np_)
    blocks += [ident + e, -ident + e]
    rhs += [pref, -pref]

    # inclusion in the frame grown by the synthesized edges e_t = (left, right, top, bottom)
    cw, ch = CX / zoom, CY / zoom
    xb, yb = CX - margin, CY - margin
    rows, cols, vals, rr, r = [], [], [], [], 0
    for t in range(n):
        a, b, x, y = 4 * t, 4 * t + 1, 4 * t + 2, 4 * t + 3
        eL, eR, eT, eB = oe + 4 * t, oe + 4 * t + 1, oe + 4 * t + 2, oe + 4 * t + 3
        for sx in (-1, 1):
            for sy in (-1, 1):
                px, py = sx * cw, sy * ch
                # X = a px - b py + x <= xb + eR ; -X <= xb + eL
                rows += [r] * 4; cols += [a, b, x, eR]; vals += [px, -py, 1, -1]; rr.append(xb); r += 1
                rows += [r] * 4; cols += [a, b, x, eL]; vals += [-px, py, -1, -1]; rr.append(xb); r += 1
                # Y = b px + a py + y <= yb + eB ; -Y <= yb + eT
                rows += [r] * 4; cols += [b, a, y, eB]; vals += [px, py, 1, -1]; rr.append(yb); r += 1
                rows += [r] * 4; cols += [b, a, y, eT]; vals += [-px, -py, -1, -1]; rr.append(yb); r += 1
    blocks.append(sp.csr_matrix((vals, (rows, cols)), shape=(r, nv)))
    rhs.append(np.array(rr))

    A = sp.vstack(blocks).tocsr()
    bvec = np.concatenate(rhs)
    wpar = np.array([C_ZOOM, C_ROLL, 1.0, 1.0])
    c = np.zeros(nv)
    c[ou:oe] = np.tile(lam, n)
    c[oe:o1] = mu
    c[o1:o2] = w[0] * np.tile(wpar, n1)
    c[o2:o3] = w[1] * np.tile(wpar, n2)
    c[o3:] = w[2] * np.tile(wpar, n3)
    ex, ey = ext[0] * W, ext[1] * H
    bounds = ([(A_MIN, A_MAX), (-B_MAX, B_MAX), (None, None), (None, None)] * n + [(0, None)] * np_
              + [(0, ex), (0, ex), (0, ey), (0, ey)] * n + [(0, None)] * (nv - o1))
    res = linprog(c, A_ub=A, b_ub=bvec, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    return res.x[:np_].reshape(n, 4), res.x[oe:o1].reshape(n, 4)
