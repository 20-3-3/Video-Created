"""Render the stabilized frames, synthesizing what lies outside the frame.

Where the crop window of frame t leaves the picture, the missing pixels are
taken from neighbouring frames, which saw that part of the scene while the
hand moved (Adobe calls this "synthesize edges"; Matsushita et al. 2006
"motion inpainting"; Liu et al. 2021 fuse neighbours the same way):

- motion from t to a neighbour n is dense DIS optical flow (at 360x640),
  split into a similarity fitted to it plus a residual; the residual is
  continued past the border of t by normalized convolution and fades out
  with distance, so the fill follows the parallax at the edge instead of
  one rigid shift;
- neighbours are tried nearest first (up to +-K), each weighted by
  exp(-|n-t|/TAU) and by a soft mask of its own frame, and averaged, so the
  fill changes smoothly from frame to frame; per-channel gain matches each
  neighbour to t on the overlap;
- seam match: the fill is also made on a band just inside frame t, and its
  difference to t there is continued smoothly over the whole fill (a
  low-frequency, Poisson-like correction), so exposure and sky gradients
  line up and only the fill's texture is kept;
- the real pixels blend into the fill over a short feather; anything no
  neighbour covers is inpainted (Telea) at low resolution.

Frames are given at `scale` times the 720x1280 analysis resolution
(scale 2 for the 2x super-resolved frames).
"""
import cv2
import numpy as np

W, H = 720, 1280
CX, CY = (W - 1) / 2, (H - 1) / 2
K, TAU = 12, 3.0
FEATHER = 6.0          # px at 720p: real pixels fade into the fill over this
BAND = 24.0            # px at 720p inside the frame where fill and real are compared
RES_DECAY = 120.0      # px at 720p over which the extrapolated residual fades
MARGIN = 200           # px at 720p around the frame for the residual canvas
FS = 0.5               # flow scale
CS = 0.25              # residual canvas scale


def sample(img, x, y, interp=cv2.INTER_LINEAR, border=cv2.BORDER_REPLICATE):
    """cv2.remap at arbitrary point lists (remap maps must stay under 32767 per side)."""
    n = x.size
    cols = 4096
    rows = (n + cols - 1) // cols
    pad = rows * cols - n
    mx = np.pad(x.ravel(), (0, pad)).reshape(rows, cols).astype(np.float32)
    my = np.pad(y.ravel(), (0, pad)).reshape(rows, cols).astype(np.float32)
    out = cv2.remap(img, mx, my, interp, borderMode=border)
    return out.reshape(rows * cols, *img.shape[2:])[:n].reshape(*x.shape, *img.shape[2:])


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


class FlowField:
    """Maps points of frame t (any position, also outside it) to frame n."""

    def __init__(self, gt, gn):
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        a = cv2.resize(gt, None, fx=FS, fy=FS, interpolation=cv2.INTER_AREA)
        b = cv2.resize(gn, None, fx=FS, fy=FS, interpolation=cv2.INTER_AREA)
        flow = dis.calc(a, b, None) / FS                       # 720p px, on the FS grid
        h, w = a.shape
        gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
        px, py = (gx + 0.5) / FS - 0.5, (gy + 0.5) / FS - 0.5  # 720p coords of the grid
        src = np.stack([px, py], -1).reshape(-1, 2)
        dst = src + flow.reshape(-1, 2)
        sel = np.random.default_rng(0).choice(len(src), min(4000, len(src)), replace=False)
        m, _ = cv2.estimateAffinePartial2D(src[sel], dst[sel], method=cv2.RANSAC,
                                           ransacReprojThreshold=2.0, maxIters=3000)
        self.g = m if m is not None else np.float32([[1, 0, 0], [0, 1, 0]])
        glob = src @ self.g[:, :2].T + self.g[:, 2]
        res = (dst - glob).reshape(h, w, 2)
        # residual on a coarse canvas that reaches MARGIN px past the frame
        cw, ch = int((W + 2 * MARGIN) * CS), int((H + 2 * MARGIN) * CS)
        canvas = np.zeros((ch, cw, 2), np.float32)
        known = np.zeros((ch, cw), np.float32)
        small = cv2.resize(res, (int(W * CS), int(H * CS)), interpolation=cv2.INTER_AREA)
        o = int(MARGIN * CS)
        canvas[o:o + small.shape[0], o:o + small.shape[1]] = small
        known[o:o + small.shape[0], o:o + small.shape[1]] = 1
        num = cv2.GaussianBlur(canvas * known[..., None], (0, 0), 30 * CS)
        den = cv2.GaussianBlur(known, (0, 0), 30 * CS)[..., None]
        ext = num / np.maximum(den, 1e-4)
        dist = cv2.distanceTransform((1 - known).astype(np.uint8), cv2.DIST_L2, 3) / CS
        ext *= np.exp(-dist / RES_DECAY)[..., None]
        self.res = np.where(known[..., None] > 0, canvas, ext).astype(np.float32)

    def __call__(self, x, y):
        """x, y: 720p coords in frame t (float32 arrays) -> coords in frame n."""
        gx = self.g[0, 0] * x + self.g[0, 1] * y + self.g[0, 2]
        gy = self.g[1, 0] * x + self.g[1, 1] * y + self.g[1, 2]
        mx = ((x + MARGIN + 0.5) * CS - 0.5).astype(np.float32)
        my = ((y + MARGIN + 0.5) * CS - 0.5).astype(np.float32)
        r = sample(self.res, mx, my)
        return (gx + r[..., 0]).astype(np.float32), (gy + r[..., 1]).astype(np.float32)


def border_dist(x, y):
    """Distance (720p px) to the frame border: positive inside, negative outside."""
    return np.minimum(np.minimum(x, W - 1 - x), np.minimum(y, H - 1 - y))


def inside_soft(x, y, feather=FEATHER):
    """1 well inside the 720p frame, ramping to 0 at its border."""
    return smoothstep(0.0, feather, border_dist(x, y))


def spread(diff, mask, step):
    """Smoothly continue `diff` (known where mask) over the whole image, at 1/step resolution."""
    h, w = mask.shape
    sw, sh = max(1, w // step), max(1, h // step)
    m = cv2.resize(mask.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
    d = cv2.resize(diff * mask[..., None], (sw, sh), interpolation=cv2.INTER_AREA)
    out = np.zeros_like(d)
    wsum = np.zeros_like(m)
    for sigma, k in ((2.0, 1.0), (6.0, 0.3), (18.0, 0.1)):   # near the seam first, far fallback
        num = cv2.GaussianBlur(d, (0, 0), sigma)
        den = cv2.GaussianBlur(m, (0, 0), sigma)
        wk = np.clip(den * 8, 0, 1) * k * (1 - np.clip(wsum, 0, 1))
        out += wk[..., None] * num / np.maximum(den, 1e-6)[..., None]
        wsum += wk
    out /= np.maximum(wsum, 1e-6)[..., None]
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)


def to_pix(x, y, scale):
    """720p coords -> pixel coords of a frame `scale` times larger."""
    return ((x + 0.5) * scale - 0.5).astype(np.float32), ((y + 0.5) * scale - 0.5).astype(np.float32)


def output_coords(pt, zoom, out_w, out_h):
    """720p coords in frame t sampled by each output pixel (crop window via B_t)."""
    a, b, x, y = pt
    gy, gx = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
    ux = (gx - (out_w - 1) / 2) * (W / out_w) / zoom
    uy = (gy - (out_h - 1) / 2) * (H / out_h) / zoom
    return (a * ux - b * uy + x + CX).astype(np.float32), (b * ux + a * uy + y + CY).astype(np.float32)


def render_frame(t, frames, gray, pt, zoom, out_w, out_h, scale):
    """frames / gray: dict index -> image (frames float32 at `scale`, gray uint8 720p)."""
    x, y = output_coords(pt, zoom, out_w, out_h)
    mx, my = to_pix(x, y, scale)
    real = cv2.remap(frames[t], mx, my, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
    dist = border_dist(x, y)
    alpha = smoothstep(0.0, FEATHER, dist)
    info = {"missing": float(np.mean(dist < 0)), "used": [], "uncovered": 0.0}
    if not (dist < 0.5).any():
        return real, info
    # the fill is also computed on a band inside the frame, to match it to t there
    need = dist < BAND
    ys, xs = np.nonzero(need)
    qx, qy = x[need], y[need]
    acc = np.zeros((len(qx), 3), np.float32)
    wsum = np.zeros(len(qx), np.float32)
    order = sorted((n for n in frames if n != t and abs(n - t) <= K), key=lambda n: (abs(n - t), n))
    for n in order:
        ff = FlowField(gray[t], gray[n])
        nx, ny = ff(qx, qy)
        wn = inside_soft(nx, ny) * np.exp(-abs(n - t) / TAU)
        ok = wn > 1e-3
        if not ok.any():
            continue
        px, py = to_pix(nx[ok], ny[ok], scale)
        col = sample(frames[n], px, py)
        # exposure match on the overlap band just inside frame t
        band = ok & (alpha[need] > 0.5)
        if band.sum() > 200:
            tx, ty = to_pix(qx[ok][band[ok]], qy[ok][band[ok]], scale)
            ref = sample(frames[t], tx, ty)
            gain = np.clip(np.median(ref, 0) / np.maximum(np.median(col[band[ok]], 0), 1e-4), 0.85, 1.15)
            col = col * gain
        acc[ok] += col * wn[ok, None]
        wsum[ok] += wn[ok]
        info["used"].append(n)
        short = wsum[dist[need] < 0.5] < 0.6
        if not short.any() or (len(info["used"]) >= 8 and short.mean() < 0.002):
            break
    fill = np.zeros_like(real)
    have = np.zeros(real.shape[:2], bool)
    covered = wsum > 1e-3
    fill[ys[covered], xs[covered]] = acc[covered] / wsum[covered, None]
    have[ys[covered], xs[covered]] = True
    # seam match: what the fill differs from t on the band inside the frame is
    # continued smoothly over the whole fill (a low-frequency Poisson-like
    # correction), so sky gradients and exposure line up and only the fill's
    # texture is kept
    band = have & (dist >= 1.5)
    if band.sum() > 50:
        fill += spread(real - fill, band, step=max(4, int(8 * scale))) * have[..., None]
    hole = (~have) & (alpha < 0.5)
    if hole.any():
        # nothing saw these pixels: inpaint from what is around, softly
        info["uncovered"] = float(hole.mean())
        base = np.where(have[..., None], fill, real)
        s = 0.25
        sm = cv2.resize(base, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        hm = cv2.resize(hole.astype(np.uint8), (sm.shape[1], sm.shape[0]), interpolation=cv2.INTER_NEAREST)
        hm = cv2.dilate(hm, np.ones((3, 3), np.uint8))
        u8 = np.clip(sm * 255 + 0.5, 0, 255).astype(np.uint8)
        inp = cv2.inpaint(u8, hm, 5, cv2.INPAINT_TELEA).astype(np.float32) / 255
        inp = cv2.GaussianBlur(cv2.resize(inp, (real.shape[1], real.shape[0])), (0, 0), 3 * scale)
        fill[hole] = inp[hole]
        have |= hole
    # real pixels blend into the fill over the feather; where there is no fill
    # (border pixels nobody else saw) the real, edge-reflected pixels stay
    a3 = np.where(have, alpha, 1.0)[..., None]
    return real * a3 + fill * (1 - a3), info
