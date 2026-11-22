"""Finishing pass: look LUT, sharpening, diffusion, vignette, grain, fades, encode.

Everything up to the encoder stays in 16-bit RGB; the grain doubles as the
dither for the single 8-bit quantisation at the end. Size and frame rate
come from the input (1080x1920 at 24 fps for v1, 1440x2560 at 60 fps for v2).

1. ffmpeg: 3D LUT (tetrahedral) + contrast-adaptive sharpening (CAS);
2. diffusion in linear light, energy-preserving like a 1/8 Black Pro-Mist:
   a little of the light above ~70 % spreads around (the sky softly bleeds
   over the dark towers and trees), slightly warm; radius scales with size;
3. vignette: an ellipse that follows the 9:16 frame, corners -1/3 stop;
4. grain: luma only, fine, strongest in the mid-tones, new every frame;
   kept light so it survives a messenger's re-encode as texture;
5. fade from black over the first 0.25 s and to black over the last 0.7 s;
6. x264 High (level picked by x264), tune film, BT.709 tagged, keyframe
   every 2 s; AAC audio when given.

usage: finish.py STAB.mkv LOOK.cube OUT.mp4 [AUDIO.wav] [--crf N] [--sharpen S] [--grain G]
"""
import argparse
import os
import subprocess
from fractions import Fraction

import cv2
import numpy as np

GAMMA = 2.4
GLOW, GLOW_TINT = 0.18, np.array([1.0, 0.95, 0.88], np.float32)
VIGNETTE = 0.20
FADE_IN, FADE_OUT = 0.25, 0.7
SEED = 2010
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def vignette_mask(w, h):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx, ny = (xx - (w - 1) / 2) / ((w - 1) / 2), (yy - (h - 1) / 2) / ((h - 1) / 2)
    r = np.sqrt((nx ** 2 + ny ** 2) / 2)          # 0 centre, 1 corners
    return (1 - VIGNETTE * smoothstep(0.30, 1.0, r))[..., None]


def diffuse(lin):
    h, w = lin.shape[:2]
    k = h / 1920                                   # radii were set on a 1920-high frame
    y = lin @ LUMA
    bright = lin * smoothstep(0.35, 0.85, y)[..., None]
    small = cv2.resize(bright, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    halo = 0.6 * cv2.GaussianBlur(small, (0, 0), 5 * k) + 0.4 * cv2.GaussianBlur(small, (0, 0), 18 * k)
    halo = cv2.resize(halo, (w, h), interpolation=cv2.INTER_LINEAR)
    # the light that spreads out is taken from where it came from
    return lin + GLOW * (halo * GLOW_TINT - bright)


def grain(rng, v, amount):
    h, w = v.shape[:2]
    n = cv2.GaussianBlur(rng.standard_normal((h, w), dtype=np.float32), (0, 0), 0.85)
    n /= n.std()
    y = v @ LUMA
    amp = amount * (0.45 + 0.55 * 4 * y * (1 - y))
    return v + (amp * n)[..., None]


def fade(i, n, fps):
    if os.environ.get("NO_FADES"):   # look tests on short excerpts
        return 1.0
    t = i / fps
    return min(1.0, (t + 1 / fps) / FADE_IN, (n - i) / (FADE_OUT * fps))


def probe(src):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                          "-show_entries", "stream=width,height,r_frame_rate,nb_read_packets",
                          "-of", "csv=p=0", src], capture_output=True, text=True, check=True).stdout
    w, h, rate, n = out.strip().split(",")
    return int(w), int(h), Fraction(rate), int(n)   # packets: FFV1 is intra-only


def main(src, lut, dst, audio=None, crf=19, sharpen=0.35, grain_amount=0.008):
    w, h, rate, n = probe(src)
    fps = float(rate)
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", src, "-vf",
                            f"format=rgb48le,lut3d=file={lut}:interp=tetrahedral,"
                            f"cas=strength={sharpen},format=rgb48le",
                            "-f", "rawvideo", "-pix_fmt", "rgb48le", "-"], stdout=subprocess.PIPE)
    enc_cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb48le",
               "-s", f"{w}x{h}", "-r", str(rate), "-i", "-"]
    if audio:
        enc_cmd += ["-i", audio]
    gop = int(round(2 * fps))
    enc_cmd += ["-sws_flags", "spline+accurate_rnd+full_chroma_int",
                "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
                "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-tune", "film",
                "-profile:v", "high", "-x264-params", f"keyint={gop}:min-keyint={gop // 4}",
                "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
                "-color_range", "tv"]
    if audio:
        enc_cmd += ["-c:a", "aac", "-b:a", "256k", "-ar", "48000"]
    enc_cmd += ["-movflags", "+faststart", dst]
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE)

    vig = vignette_mask(w, h)
    rng = np.random.default_rng(SEED)
    size = w * h * 3 * 2
    for i in range(n):
        buf = dec.stdout.read(size)
        if len(buf) < size:
            break
        v = np.frombuffer(buf, np.uint16).reshape(h, w, 3).astype(np.float32) / 65535
        lin = v ** GAMMA
        lin = diffuse(lin) * vig * fade(i, n, fps) ** GAMMA
        v = np.clip(lin, 0, 1) ** (1 / GAMMA)
        v = grain(rng, v, grain_amount)
        enc.stdin.write((np.clip(v, 0, 1) * 65535 + 0.5).astype(np.uint16).tobytes())
    enc.stdin.close()
    assert enc.wait() == 0 and dec.wait() == 0
    print(f"{dst}: {n} frames, {w}x{h} at {fps:g} fps")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src"), ap.add_argument("lut"), ap.add_argument("dst"), ap.add_argument("audio", nargs="?")
    ap.add_argument("--crf", type=int, default=19)
    ap.add_argument("--sharpen", type=float, default=0.35)
    ap.add_argument("--grain", type=float, default=0.008)
    a = ap.parse_args()
    main(a.src, a.lut, a.dst, a.audio, a.crf, a.sharpen, a.grain)
