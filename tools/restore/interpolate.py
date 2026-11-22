"""24 -> 60 fps with RIFE 4.25 on the stabilized 2K frames.

Output frame k shows time k/60 s = source position 0.4 k: every 5 output
frames, 2 are source frames passed through untouched and 3 are synthesized
at 0.2 / 0.4 / 0.6 / 0.8 between two source frames. Interpolating after
stabilization keeps the motion between the two frames small and smooth,
which is where flow-based interpolation is cleanest. Motion is estimated at
720x1280 and applied to the full frames (Rife.hires).

usage: interpolate.py STAB.mkv RIFE_flownet.pkl OUT.mkv [N_OUT]
OUT.mkv: rgb48 FFV1, 60 fps; N_OUT defaults to 2.5 x the source frames
"""
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from rife import Rife  # noqa: E402

SRC_FPS, OUT_FPS = 24, 60


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                          "-show_entries", "stream=width,height,nb_read_packets", "-of", "csv=p=0", path],
                         capture_output=True, text=True, check=True).stdout.strip().split(",")
    return int(out[0]), int(out[1]), int(out[2])


def main(src, weights, dst, n_out=None):
    w, h, n = probe(src)
    n_out = int(n_out) if n_out else int(round(n * OUT_FPS / SRC_FPS))
    torch.set_num_threads(4)
    rife = Rife(weights)
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", src, "-f", "rawvideo", "-pix_fmt", "rgb48le", "-"],
                           stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb48le",
                            "-s", f"{w}x{h}", "-r", str(OUT_FPS), "-i", "-", "-c:v", "ffv1", "-level", "3",
                            "-slices", "16", "-pix_fmt", "rgb48le", dst], stdin=subprocess.PIPE)
    size = w * h * 6
    cache = {}

    def frame(i):
        i = min(i, n - 1)
        while i not in cache:
            buf = dec.stdout.read(size)
            assert len(buf) == size, "short read"
            k = len(cache) and max(cache) + 1
            arr = np.frombuffer(buf, np.uint16).reshape(h, w, 3).astype(np.float32) / 65535
            cache[k] = torch.from_numpy(arr).permute(2, 0, 1)[None]
            for old in [j for j in cache if j < k - 2]:
                del cache[old]
        return cache[i]

    t0, synth = time.time(), 0
    for k in range(n_out):
        pos = k * SRC_FPS / OUT_FPS
        i = math.floor(pos + 1e-9)
        a = pos - i
        if a < 1e-6 or i >= n - 1:
            out = frame(i)
        else:
            out = rife.hires(frame(i), frame(i + 1), float(a))
            synth += 1
        arr = out[0].permute(1, 2, 0).numpy()
        enc.stdin.write((np.clip(arr, 0, 1) * 65535 + 0.5).astype(np.uint16).tobytes())
        if k % 60 == 0:
            print(f"{k}/{n_out} frames, {(time.time() - t0) / (k + 1):.2f} s/frame", flush=True)
    enc.stdin.close()
    dec.stdout.close()
    assert enc.wait() == 0
    print(f"{dst}: {n_out} frames at {OUT_FPS} fps ({synth} synthesized) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(*sys.argv[1:])
