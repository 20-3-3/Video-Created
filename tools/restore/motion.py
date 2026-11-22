"""Frame-to-frame camera motion for the stabilizer.

For each pair of consecutive frames: Shi-Tomasi corners spread over a grid,
pyramidal Lucas-Kanade tracking with a forward/backward check, then a
similarity transform (shift, rotation, scale) by RANSAC. The dominant motion
is the background (trees, wall, paving carry most of the texture); a second
RANSAC on the remaining tracks gives the secondary motion, which in this
selfie is mostly the near face and body, kept for diagnostics.

Transforms are 3x3 matrices in frame-centred pixel coordinates mapping a
point of frame t-1 to the same scene point in frame t.
"""
import cv2
import numpy as np

GRID = (6, 10)          # cells across x, y
PER_CELL = 40
LK = dict(winSize=(31, 31), maxLevel=5,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))


def corners(gray):
    h, w = gray.shape
    pts = []
    for i in range(GRID[0]):
        for j in range(GRID[1]):
            x0, x1 = w * i // GRID[0], w * (i + 1) // GRID[0]
            y0, y1 = h * j // GRID[1], h * (j + 1) // GRID[1]
            c = cv2.goodFeaturesToTrack(gray[y0:y1, x0:x1], PER_CELL, 0.003, 8, blockSize=7)
            if c is not None:
                pts.append(c.reshape(-1, 2) + (x0, y0))
    return np.concatenate(pts).astype(np.float32) if pts else np.zeros((0, 2), np.float32)


def track(g0, g1):
    p0 = corners(g0)
    if len(p0) < 8:
        return p0, p0, np.zeros(len(p0), bool)
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0.reshape(-1, 1, 2), None, **LK)
    pb, st2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, **LK)
    p1, pb = p1.reshape(-1, 2), pb.reshape(-1, 2)
    ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(pb - p0, axis=1) < 0.8)
    return p0, p1, ok


def similarity(p0, p1, thresh=1.5):
    if len(p0) < 8:
        return None, np.zeros(len(p0), bool)
    m, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=thresh,
                                         maxIters=5000, confidence=0.999, refineIters=30)
    if m is None:
        return None, np.zeros(len(p0), bool)
    return m, inl.ravel().astype(bool)


def centred(m, w, h):
    """2x3 pixel-coordinate transform -> 3x3 in frame-centred coordinates."""
    a = np.vstack([m, [0, 0, 1]])
    t = np.array([[1, 0, (w - 1) / 2], [0, 1, (h - 1) / 2], [0, 0, 1]])
    return np.linalg.inv(t) @ a @ t


def params(a):
    """3x3 similarity -> (dx, dy, angle rad, log scale)."""
    s = np.hypot(a[0, 0], a[1, 0])
    return np.array([a[0, 2], a[1, 2], np.arctan2(a[1, 0], a[0, 0]), np.log(s)])


def from_params(dx, dy, ang, logs):
    s = np.exp(logs)
    c, n = s * np.cos(ang), s * np.sin(ang)
    return np.array([[c, -n, dx], [n, c, dy], [0, 0, 1.0]])


def pair_motion(g0, g1):
    h, w = g0.shape
    p0, p1, ok = track(g0, g1)
    p0, p1 = p0[ok], p1[ok]
    m, inl = similarity(p0, p1)
    info = {"tracks": int(len(p0)), "inliers": int(inl.sum())}
    if m is None:
        return None, None, info
    rest0, rest1 = p0[~inl], p1[~inl]
    m2, inl2 = similarity(rest0, rest1, thresh=3.0)
    info["inliers2"] = int(inl2.sum())
    sec = centred(m2, w, h) if (m2 is not None and inl2.sum() >= 25) else None
    if sec is not None:
        info["sec_centroid"] = rest0[inl2].mean(axis=0).round(1).tolist()
    return centred(m, w, h), sec, info
