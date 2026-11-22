"""Mid-frame synthesis from two neighbours with dense DIS optical flow.

Backward-warps both neighbours to t=0.5 using the linear flow approximation
(F_t0 = -t(1-t)F01 + t^2 F10, F_t1 = -F_t0 at t=0.5) and blends them with
weights from a forward/backward consistency check, so regions that are
occluded in one neighbour come from the other.
"""
import cv2
import numpy as np


def _dis():
    d = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    d.setFinestScale(0)
    d.setPatchSize(12)
    d.setPatchStride(4)
    d.setGradientDescentIterations(25)
    d.setVariationalRefinementIterations(10)
    d.setVariationalRefinementAlpha(20.0)
    d.setUseMeanNormalization(True)
    d.setUseSpatialPropagation(True)
    return d


def _remap(img, flow):
    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + flow[..., 0], gy + flow[..., 1],
                     interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def midframe(a, b):
    """a, b: HxWx3 uint8 RGB. Returns the synthesized frame halfway between them."""
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    dis = _dis()
    f01 = dis.calc(ga, gb, None)
    f10 = dis.calc(gb, ga, None)
    ft0 = (-0.25 * f01 + 0.25 * f10).astype(np.float32)
    ft1 = -ft0
    wa = _remap(a.astype(np.float32), ft0)
    wb = _remap(b.astype(np.float32), ft1)
    # consistency: a pixel of A whose flow round-trips back to itself is visible in both
    err_a = np.linalg.norm(f01 + _remap(f10, f01), axis=2)
    err_b = np.linalg.norm(f10 + _remap(f01, f10), axis=2)
    # the +0.02 floor makes pixels that fail the check in both neighbours fall
    # back to a plain 50/50 blend instead of being divided towards black
    va = np.exp(-_remap(err_a, ft0) / 2.0) + 0.02
    vb = np.exp(-_remap(err_b, ft1) / 2.0) + 0.02
    wsum = va + vb
    out = (wa * va[..., None] + wb * vb[..., None]) / wsum[..., None]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)
