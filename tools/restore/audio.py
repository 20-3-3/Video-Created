"""Clean up and level the phone audio of the clip.

Chain (mono, 48 kHz, float):
1. declip (declip.py): the 11.0-11.9 s shout was limited flat at full scale;
2. high-pass 80 Hz, 4th order: handling thumps and wind rumble below the voices;
3. wind ducker: per STFT frame, if the 80-320 Hz band carries more than
   about a quarter of the 400-3000 Hz voice band's energy, that is wind or
   handling, not voice (shouted voices here sit around 280 Hz and up), so the
   low bins are turned down towards that ratio (at most -14 dB), smoothed in
   time and faded out by 320 Hz;
4. EQ for the phone mic: -1.5 dB at 380 Hz (boxiness), +2 dB presence at
   3.2 kHz, +2.5 dB shelf from 8 kHz (air the AAC encode had thinned);
5. leveller: short-term loudness (400 ms, K-weighted) is pulled towards a
   target by at most +6 / -8 dB, with a gate so pauses are not raised;
6. compressor 2.5:1 (RMS, 8 ms attack, 140 ms release, soft knee);
7. gain to -14 LUFS integrated, then a look-ahead limiter on 4x oversampled
   peaks at -1.2 dBTP;
8. 20 ms fade-in, fade-out over the last 0.7 s (matches the picture), padded
   or trimmed to the video length; written as dual-mono stereo.

usage: audio.py IN.wav OUT.wav DURATION_S
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

sys.path.insert(0, str(Path(__file__).parent))
from declip import declip  # noqa: E402

SR = 48000
TARGET_LUFS = -14.0   # of the stereo file; BS.1770 sums channels, so dual mono measures +3 dB
MONO_TARGET = TARGET_LUFS - 10 * np.log10(2)
CEILING_DBTP = -1.2


def db(x):
    return 20 * np.log10(np.maximum(x, 1e-12))


def undb(g):
    return 10 ** (g / 20)


# --- filters ---------------------------------------------------------------
def biquad(kind, f0, gain_db=0.0, q=0.707):
    """RBJ audio-EQ-cookbook biquad -> sos row."""
    a = undb(gain_db / 2)
    w0 = 2 * np.pi * f0 / SR
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2 * q)
    if kind == "peak":
        b = [1 + alpha * a, -2 * cw, 1 - alpha * a]
        den = [1 + alpha / a, -2 * cw, 1 - alpha / a]
    elif kind == "highshelf":
        s = 2 * np.sqrt(a) * alpha
        b = [a * ((a + 1) + (a - 1) * cw + s), -2 * a * ((a - 1) + (a + 1) * cw), a * ((a + 1) + (a - 1) * cw - s)]
        den = [(a + 1) - (a - 1) * cw + s, 2 * ((a - 1) - (a + 1) * cw), (a + 1) - (a - 1) * cw - s]
    else:
        raise ValueError(kind)
    b, den = np.array(b) / den[0], np.array(den) / den[0]
    return np.concatenate([b, den])


def kweight(x):
    """ITU-R BS.1770 K-weighting (48 kHz coefficients)."""
    x = signal.lfilter([1.53512485958697, -2.69169618940638, 1.19839281085285],
                       [1.0, -1.69065929318241, 0.73248077421585], x)
    return signal.lfilter([1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621], x)


def integrated_lufs(x):
    """BS.1770 integrated loudness (mono), 400 ms blocks, 75 % overlap, gated."""
    k = kweight(x) ** 2
    blk, hop = int(0.4 * SR), int(0.1 * SR)
    z = np.array([k[i:i + blk].mean() for i in range(0, len(k) - blk + 1, hop)])
    lk = -0.691 + 10 * np.log10(np.maximum(z, 1e-12))
    z = z[lk > -70]
    rel = -0.691 + 10 * np.log10(z.mean()) - 10
    z = z[-0.691 + 10 * np.log10(z) > rel]
    return -0.691 + 10 * np.log10(z.mean())


def smooth_gain(g_db, att, rel, rate):
    """One-pole smoothing, `att` s when the gain goes down, `rel` s when up."""
    ka, kr = np.exp(-1 / (att * rate)), np.exp(-1 / (rel * rate))
    out = np.empty_like(g_db)
    s = g_db[0]
    for i, v in enumerate(g_db):
        k = ka if v < s else kr
        s = k * s + (1 - k) * v
        out[i] = s
    return out


# --- stages ----------------------------------------------------------------
def wind_duck(x, ratio_db=-6.0, floor_db=-14.0):
    n, hop = 2048, 256
    f, _, z = signal.stft(x, SR, nperseg=n, noverlap=n - hop)
    p = np.abs(z) ** 2
    low = (f >= 80) & (f < 320)
    voice = (f >= 400) & (f < 3000)
    e_low = p[low].sum(0) + 1e-12
    e_voice = p[voice].sum(0) + 1e-12
    excess = 10 * np.log10(e_low / e_voice) - ratio_db          # dB above the allowed ratio
    # only where the low band is noise-like: a voiced frame has its harmonics
    # standing out of the band (spectral flatness ~0.03-0.05), wind and
    # handling noise are flat (~0.2-0.4)
    pl = p[low] + 1e-14
    flat = np.exp(np.mean(np.log(pl), 0)) / np.mean(pl, 0)
    noisy = np.clip((flat - 0.08) / (0.25 - 0.08), 0, 1)
    g = -np.clip(excess, 0, -floor_db) * noisy
    g = smooth_gain(g, 0.02, 0.15, SR / hop)
    ramp = np.clip((320 - f) / 120, 0, 1)                       # full below 200 Hz, none at 320
    gain = undb(g[None, :] * ramp[:, None])
    _, y = signal.istft(z * gain, SR, nperseg=n, noverlap=n - hop)
    return y[:len(x)], g


def leveller(x, target=-20.0, up=6.0, down=8.0, gate=-42.0):
    k = kweight(x) ** 2
    hop = int(0.02 * SR)
    win = int(0.4 * SR)
    c = np.concatenate([[0], np.cumsum(k)])
    centres = np.arange(0, len(x), hop)
    lo, hi = np.clip(centres - win // 2, 0, len(x)), np.clip(centres + win // 2, 0, len(x))
    st = -0.691 + 10 * np.log10(np.maximum((c[hi] - c[lo]) / np.maximum(hi - lo, 1), 1e-12))
    g = np.clip(target - st, -down, up)
    # gate: in pauses hold whatever gain the speech before them had, so the
    # pause is neither pulled up on its own nor makes the gain pump
    voiced = st > gate
    idx = np.where(voiced, np.arange(len(g)), 0)
    np.maximum.accumulate(idx, out=idx)
    g = np.where(voiced, g, g[idx])
    g = smooth_gain(g, 0.08, 0.6, 1 / 0.02)
    return x * undb(np.interp(np.arange(len(x)), centres, g))


def compressor(x, thresh=-22.0, ratio=2.5, knee=6.0, att=0.008, rel=0.14):
    env = np.sqrt(np.maximum(signal.lfilter([1 - np.exp(-1 / (0.01 * SR))],
                                            [1, -np.exp(-1 / (0.01 * SR))], x ** 2), 1e-14))
    lv = db(env)
    over = lv - thresh
    gr = np.where(over <= -knee / 2, 0.0,
                  np.where(over >= knee / 2, over * (1 / ratio - 1),
                           (1 / ratio - 1) * (over + knee / 2) ** 2 / (2 * knee)))
    gr = smooth_gain(gr, att, rel, SR)
    return x * undb(gr)


def limiter(x, ceiling_db=CEILING_DBTP, look=0.0015, rel=0.08):
    up = signal.resample_poly(x, 4, 1)
    peak = np.abs(up).reshape(-1, 4).max(1)[:len(x)]            # true-peak estimate per sample
    la = int(look * SR)
    # look-ahead: running max over the window ahead
    from scipy.ndimage import maximum_filter1d
    pk = maximum_filter1d(peak, size=2 * la + 1, mode="nearest")
    need = np.minimum(0.0, ceiling_db - db(pk))
    g = smooth_gain(need, 0.0005, rel, SR)
    g = np.minimum(g, need)                                     # never let a peak through
    y = x * undb(g)
    return y


def main(src, dst, duration):
    x, sr = sf.read(src)
    assert sr == SR
    if x.ndim > 1:
        x = x.mean(1)
    x = x - x.mean()
    x = declip(x, 0.94)
    x = signal.sosfilt(signal.butter(4, 80, "highpass", fs=SR, output="sos"), x)
    x, wind_g = wind_duck(x)
    eq = np.array([biquad("peak", 380, -1.5, 1.0), biquad("peak", 3200, 2.0, 0.8),
                   biquad("highshelf", 8000, 2.5, 0.7)])
    x = signal.sosfilt(eq, x)
    x = leveller(x)
    x = compressor(x)
    x = x * undb(MONO_TARGET - integrated_lufs(x))
    x = limiter(x)
    n = int(round(float(duration) * SR))
    x = np.pad(x, (0, max(0, n - len(x))))[:n]
    fi, fo = int(0.02 * SR), int(0.7 * SR)
    x[:fi] *= np.linspace(0, 1, fi)
    x[-fo:] *= np.cos(np.linspace(0, np.pi / 2, fo)) ** 2
    sf.write(dst, np.stack([x, x], 1), SR, subtype="FLOAT")
    print(f"{dst}: {n / SR:.3f} s, {integrated_lufs(x) + 10 * np.log10(2):.1f} LUFS stereo (target {TARGET_LUFS}), "
          f"wind ducking up to {-wind_g.min():.1f} dB, in use {100 * np.mean(wind_g < -1):.0f}% of frames")


if __name__ == "__main__":
    main(*sys.argv[1:4])
