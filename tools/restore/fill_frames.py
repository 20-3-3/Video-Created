"""Turn the phone clip into constant 24 fps by synthesizing its dropped frames.

The source is 24 fps with 10 frames missing (83 ms gaps). Every real frame is
put on its slot of the 1/24 s grid untouched (bit-exact yuv420p); only the
empty slots are synthesized. Two synthesizers were compared slot by slot:
ffmpeg minterpolate (block motion compensation) and dense DIS optical flow
(flowinterp.py). Each slot uses the one that looked clean at full resolution.

usage: fill_frames.py SOURCE.mp4 OUT.mkv
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from vio import read_yuv420, yuv420_to_rgb, rgb_to_yuv420, W, H  # noqa: E402
from flowinterp import midframe  # noqa: E402

FPS = 24
# slot -> synthesizer, chosen by eye on full-resolution crops
CHOICE = {6: "dis", 17: "mi", 57: "mi", 78: "mi", 120: "dis",
          126: "mi", 134: "dis", 178: "dis", 195: "dis", 206: "mi"}


def frame_slots(src):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "frame=pts_time", "-of", "csv=p=0", src],
                         check=True, capture_output=True, text=True).stdout
    t = np.array([float(x) for x in out.split()])
    slots = np.round(t * FPS).astype(int)
    assert len(np.unique(slots)) == len(slots), "two frames on one slot"
    return slots


def main(src, dst):
    slots = frame_slots(src)
    n = slots.max() + 1
    missing = sorted(set(range(n)) - set(slots))
    assert missing == sorted(CHOICE), f"unexpected gaps {missing}"

    real = read_yuv420(src)
    seq = [None] * n
    for buf, s in zip(real, slots):
        seq[s] = buf

    with tempfile.TemporaryDirectory() as tmp:
        mi_path = f"{tmp}/mi.mkv"
        # minterpolate copies frames that sit exactly on the grid and only
        # interpolates the empty slots; tpad gives it a successor for the last frame
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-an", "-vf",
                        f"setpts=round(PTS*TB*{FPS})/({FPS}*TB),tpad=stop=1:stop_mode=clone,"
                        f"minterpolate=fps={FPS}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
                        "-c:v", "ffv1", mi_path], check=True)
        mi = read_yuv420(mi_path)
        for s in missing:
            if CHOICE[s] == "mi":
                seq[s] = mi[s]
            else:
                a, b = yuv420_to_rgb(seq[s - 1]), yuv420_to_rgb(seq[s + 1])
                seq[s] = rgb_to_yuv420(midframe(a, b))
            print(f"slot {s:3d} ({s / FPS:6.3f}s): {CHOICE[s]}")

    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p",
                            "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-color_primaries", "bt709", "-color_trc", "bt709",
                            "-colorspace", "bt709", "-color_range", "tv",
                            "-c:v", "ffv1", "-level", "3", dst], stdin=subprocess.PIPE)
    for buf in seq:
        enc.stdin.write(buf)
    enc.stdin.close()
    assert enc.wait() == 0
    print(f"{dst}: {n} frames, {len(missing)} synthesized")


if __name__ == "__main__":
    main(*sys.argv[1:3])
