"""2x super-resolution of every frame: 720x1280 -> 1440x2560 (vertical 2K).

Model: 2x LiveActionV1 SPAN (jcj83429, OpenModelDB; CC BY-NC-SA 4.0), a
small SPAN network trained for live-action film/video: it upscales and takes
out compression blocking and mosquito noise without inventing texture, so it
replaces the earlier hqdn3d pass. Architecture loaded with spandrel, weights
as plain tensors (weights_only=True). Runs in bfloat16 on CPU (AMX):
~3.5 s a frame, output within 0.25 levels of fp32 on average.

usage: upscale.py FILLED.mkv SPAN.pth OUT.mkv
OUT.mkv: rgb48 FFV1 (lossless), 24 fps
"""
import subprocess
import sys
import time

import numpy as np
import spandrel
import torch

W, H, S = 720, 1280, 2


def load(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("params_ema", "params", "state_dict"):
        if isinstance(sd, dict) and key in sd:
            sd = sd[key]
    model = spandrel.ModelLoader().load_from_state_dict(sd)
    assert model.scale == S, model.scale
    return model.model.eval()


def main(src, weights, dst):
    torch.set_num_threads(4)
    net = load(weights)
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", src, "-sws_flags",
                            "spline+accurate_rnd+full_chroma_int+full_chroma_inp", "-vf",
                            "scale=in_color_matrix=bt709:in_range=tv:out_range=pc,format=rgb48le",
                            "-f", "rawvideo", "-pix_fmt", "rgb48le", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb48le",
                            "-s", f"{W * S}x{H * S}", "-r", "24", "-i", "-", "-c:v", "ffv1", "-level", "3",
                            "-slices", "16", "-pix_fmt", "rgb48le", dst], stdin=subprocess.PIPE)
    size, n, t0 = W * H * 6, 0, time.time()
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        while True:
            buf = dec.stdout.read(size)
            if len(buf) < size:
                break
            x = torch.from_numpy(np.frombuffer(buf, np.uint16).reshape(H, W, 3).astype(np.float32) / 65535)
            y = net(x.permute(2, 0, 1)[None]).float().clamp(0, 1)[0].permute(1, 2, 0).numpy()
            enc.stdin.write((y * 65535 + 0.5).astype(np.uint16).tobytes())
            n += 1
            if n % 20 == 0:
                print(f"{n} frames, {(time.time() - t0) / n:.2f} s/frame", flush=True)
    enc.stdin.close()
    assert enc.wait() == 0 and dec.wait() == 0
    print(f"{dst}: {n} frames in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(*sys.argv[1:4])
