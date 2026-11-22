"""Declip a mono recording with consistent iterative hard thresholding.

The phone limited a shout at 11.0-11.9 s into flat tops at +-1.0, and the AAC
encode then rippled those tops between ~0.95 and 1.08, so histogram-based
detection (ffmpeg adeclip) misses most of them. Here every sample above THETA
is treated as clipped. Each 2048-sample Hann frame that holds clipped samples
is rebuilt as the sparsest spectrum (oversampled DFT, growing number of
kept coefficients) that matches the reliable samples exactly and stays at or
beyond the clip level on the clipped ones (Kitic et al., "Consistent
iterative hard thresholding for signal declipping", 2013). Frames without
clipping pass through untouched.

usage: declip.py IN.wav OUT.wav [THETA]
"""
import sys

import numpy as np
import soundfile as sf

N, HOP, OVS = 2048, 512, 2
ITERS, STEP, TOL = 400, 2, 1e-4


def declip_frame(xw, w, theta):
    rel = np.abs(xw) < theta * w
    pos = (~rel) & (xw > 0)
    neg = (~rel) & (xw < 0)
    if not (pos.any() or neg.any()):
        return xw
    y = xw.copy()
    k = STEP
    for _ in range(ITERS):
        z = np.fft.rfft(y, N * OVS)
        if k < z.size:
            z[np.argsort(np.abs(z))[:-k]] = 0
        yh = np.fft.irfft(z, N * OVS)[:N]
        y = yh.copy()
        y[rel] = xw[rel]
        y[pos] = np.maximum(yh[pos], theta * w[pos])
        y[neg] = np.minimum(yh[neg], -theta * w[neg])
        if np.linalg.norm(y - yh) <= TOL * np.linalg.norm(y):
            break
        k += STEP
    return y


def declip(x, theta):
    w = np.hanning(N + 1)[:N]
    pad = np.concatenate([np.zeros(N), x, np.zeros(N)])
    out = np.zeros_like(pad)
    norm = np.zeros_like(pad)
    clipped = np.abs(pad) >= theta
    for s in range(0, len(pad) - N + 1, HOP):
        seg = pad[s:s + N]
        if clipped[s:s + N].any():
            out[s:s + N] += declip_frame(seg * w, w, theta)
        else:
            out[s:s + N] += seg * w
        norm[s:s + N] += w
    y = out / np.maximum(norm, 1e-9)
    y = y[N:N + len(x)]
    y[~clipped[N:N + len(x)]] = x[~clipped[N:N + len(x)]]  # reliable samples exact
    return y


if __name__ == "__main__":
    src, dst = sys.argv[1:3]
    theta = float(sys.argv[3]) if len(sys.argv) > 3 else 0.94
    x, sr = sf.read(src)
    y = declip(x, theta)
    n = int(np.sum(np.abs(x) >= theta))
    print(f"{n} clipped samples (>= {theta}); peak {np.abs(x).max():.3f} -> {np.abs(y).max():.3f}")
    sf.write(dst, y, sr, subtype="FLOAT")
