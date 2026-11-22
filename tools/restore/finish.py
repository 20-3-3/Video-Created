"""Finishing pass: look LUT, sharpening, diffusion, vignette, grain, fades, encode.

Everything up to the encoder stays in 16-bit RGB; the grain doubles as the
dither for the single 8-bit quantisation at the end.

1. ffmpeg: 3D LUT (tetrahedral) + contrast-adaptive sharpening (CAS), which
   gives back the edge contrast the stabilizer's crop/upscale took;
2. diffusion in linear light, energy-preserving like a 1/8 Black Pro-Mist:
   a little of the light above ~70 % spreads around (the sky softly bleeds
   over the dark towers and trees), slightly warm;
3. vignette: an ellipse that follows the 9:16 frame, corners -1/3 stop;
4. grain: luma only, fine (~2 px), strongest in the mid-tones, new every
   frame; kept light so it survives a messenger's re-encode as texture;
5. fade from black over the first 0.25 s and to black over the last 0.7 s;
6. x264 High, CRF 19, tune film, BT.709 tagged; AAC audio when given.

usage: finish.py STAB.mkv LOOK.cube OUT.mp4 [AUDIO.wav]
"""
import os
import subprocess
import sys

import cv2
import numpy as np

W, H, FPS = 1080, 1920, 24
GAMMA = 2.4
SHARPEN = 0.35
GLOW, GLOW_TINT = 0.18, np.array([1.0, 0.95, 0.88], np.float32)
VIGNETTE = 0.20
GRAIN = 0.008
FADE_IN, FADE_OUT = 0.25, 0.7
SEED = 2010
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def vignette_mask():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    nx, ny = (xx - (W - 1) / 2) / ((W - 1) / 2), (yy - (H - 1) / 2) / ((H - 1) / 2)
    r = np.sqrt((nx ** 2 + ny ** 2) / 2)          # 0 centre, 1 corners
    return (1 - VIGNETTE * smoothstep(0.30, 1.0, r))[..., None]


def diffuse(lin):
    y = lin @ LUMA
    bright = lin * smoothstep(0.35, 0.85, y)[..., None]
    small = cv2.resize(bright, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    halo = 0.6 * cv2.GaussianBlur(small, (0, 0), 5) + 0.4 * cv2.GaussianBlur(small, (0, 0), 18)
    halo = cv2.resize(halo, (W, H), interpolation=cv2.INTER_LINEAR)
    # the light that spreads out is taken from where it came from
    return lin + GLOW * (halo * GLOW_TINT - bright)


def grain(rng, v):
    n = cv2.GaussianBlur(rng.standard_normal((H, W), dtype=np.float32), (0, 0), 0.85)
    n /= n.std()
    y = v @ LUMA
    amp = GRAIN * (0.45 + 0.55 * 4 * y * (1 - y))
    return v + (amp * n)[..., None]


def fade(i, n):
    if os.environ.get("NO_FADES"):   # look tests on short excerpts
        return 1.0
    t = i / FPS
    return min(1.0, (t + 1 / FPS) / FADE_IN, (n - i) / (FADE_OUT * FPS))


def main(src, lut, dst, audio=None):
    # frame count from the packets (FFV1 is intra-only), no decoding needed
    n = int(subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
                            "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", src],
                           capture_output=True, text=True, check=True).stdout)
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", src, "-vf",
                            f"format=rgb48le,lut3d=file={lut}:interp=tetrahedral,"
                            f"cas=strength={SHARPEN},format=rgb48le",
                            "-f", "rawvideo", "-pix_fmt", "rgb48le", "-"], stdout=subprocess.PIPE)
    enc_cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb48le",
               "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-"]
    if audio:
        enc_cmd += ["-i", audio]
    enc_cmd += ["-sws_flags", "spline+accurate_rnd+full_chroma_int",
                "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
                "-c:v", "libx264", "-preset", "slow", "-crf", "19", "-tune", "film",
                "-profile:v", "high", "-level", "4.1", "-x264-params", "keyint=48:min-keyint=12",
                "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
                "-color_range", "tv"]
    if audio:
        enc_cmd += ["-c:a", "aac", "-b:a", "256k", "-ar", "48000"]
    enc_cmd += ["-movflags", "+faststart", dst]
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE)

    vig = vignette_mask()
    rng = np.random.default_rng(SEED)
    size = W * H * 3 * 2
    for i in range(n):
        buf = dec.stdout.read(size)
        if len(buf) < size:
            break
        v = np.frombuffer(buf, np.uint16).reshape(H, W, 3).astype(np.float32) / 65535
        lin = v ** GAMMA
        lin = diffuse(lin) * vig * fade(i, n) ** GAMMA
        v = np.clip(lin, 0, 1) ** (1 / GAMMA)
        v = grain(rng, v)
        enc.stdin.write((np.clip(v, 0, 1) * 65535 + 0.5).astype(np.uint16).tobytes())
    enc.stdin.close()
    assert enc.wait() == 0 and dec.wait() == 0
    print(f"{dst}: {n} frames")


if __name__ == "__main__":
    main(*sys.argv[1:])
