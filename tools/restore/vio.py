"""Small helpers: read/write raw video frames through ffmpeg pipes.

Every YUV<->RGB conversion names the BT.709 matrix and the range explicitly;
left to itself swscale falls back to BT.601 for untagged RGB input.
"""
import subprocess
import numpy as np

W, H = 720, 1280
SWS = "spline+accurate_rnd+full_chroma_int+full_chroma_inp"


def _run(cmd, data=None):
    return subprocess.run(cmd, input=data, check=True, capture_output=True).stdout


def read_yuv420(path, w=W, h=H):
    """Every decoded frame as raw yuv420p bytes (no conversion at all)."""
    raw = _run(["ffmpeg", "-v", "error", "-i", path, "-fps_mode", "passthrough",
                "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"])
    fs = w * h * 3 // 2
    return [raw[i:i + fs] for i in range(0, len(raw) - fs + 1, fs)]


def read_frames(path, pix_fmt="rgb24", w=W, h=H, pre=None, resize=False):
    """Decode every frame of a BT.709 limited-range file into an RGB array.

    `pre` is a filter chain run on the decoded frames as yuv444p16 (still
    limited range), e.g. a denoiser, before the conversion to RGB.
    `resize` scales the frames to w x h (area average) on the way.
    """
    vf = f"scale=in_color_matrix=bt709:in_range=tv:out_range=pc,format={pix_fmt}"
    if pre:
        vf = f"format=yuv444p16le,{pre}," + vf
    if resize:
        vf = f"scale={w}:{h}:flags=area," + vf
    raw = _run(["ffmpeg", "-v", "error", "-i", path, "-sws_flags", SWS, "-vf", vf,
                "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"])
    dt = np.uint16 if pix_fmt.endswith("48le") else np.uint8
    arr = np.frombuffer(raw, dtype=dt)
    n = arr.size // (w * h * 3)
    return arr[: n * w * h * 3].reshape(n, h, w, 3)


def yuv420_to_rgb(buf, w=W, h=H):
    raw = _run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p",
                "-s", f"{w}x{h}", "-i", "-", "-sws_flags", SWS,
                "-vf", "scale=in_color_matrix=bt709:in_range=tv:out_range=pc,format=rgb24",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], buf)
    return np.frombuffer(raw, np.uint8).reshape(h, w, 3)


def rgb_to_yuv420(rgb):
    h, w = rgb.shape[:2]
    return _run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{w}x{h}", "-i", "-", "-sws_flags", SWS,
                 "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
                 "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"], rgb.tobytes())


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
