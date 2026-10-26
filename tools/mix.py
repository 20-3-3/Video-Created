"""Process the per-role voice tracks, retime the source picture and ambience (section speeds and
freeze frames written by voiceover.py), duck the ambience under the voices and encode the result."""
import json, os, re, subprocess, sys

import numpy as np

WORK, VIDEO, OUT, STEM = sys.argv[1:5]
SR = 48000
FPS = 30

FX = {
    # deep, dry announcer: full chest, no boxiness, smooth top, broadcast-style leveling
    "narrator": "highpass=f=55,bass=g=4:f=120:w=0.8,equalizer=f=230:t=q:w=1:g=1.5,"
                "equalizer=f=500:t=q:w=1.2:g=-2,treble=g=-2:f=9000:w=0.6,deesser=i=0.3,"
                "acompressor=threshold=-24dB:ratio=3:attack=10:release=180:makeup=2.5",
    # platform loudspeaker: narrow band, slight overdrive, hall reflections
    "pa": "highpass=f=320,lowpass=f=3600,equalizer=f=1700:t=q:w=1:g=5,volume=8dB,asoftclip=type=tanh,"
          "volume=-6dB,aecho=0.85:0.6:75|170|290|450:0.35|0.22|0.14|0.08",
    # attendant standing on the platform: a little room around the voice
    "attendant": "highpass=f=90,equalizer=f=200:t=q:w=1:g=1.5,"
                 "acompressor=threshold=-22dB:ratio=2.5:attack=5:release=150:makeup=1.5,"
                 "aecho=0.9:0.35:55|120|210:0.16|0.09|0.05",
}
# a narrator track that is not TTS (narrator_take.py) comes with its own chain
if os.path.exists(f"{WORK}/narrator_fx.txt"):
    FX["narrator"] = open(f"{WORK}/narrator_fx.txt").read().strip()
TARGET = {"narrator": -16.0, "pa": -17.5, "attendant": -16.5}  # LUFS of speech (gated)
ROOM_TONE = (45.3, 47.3)  # steady platform drone without effects, looped under freeze frames
XFADE = 0.04


def run(args, **kw):
    return subprocess.run(args, capture_output=True, check=True, **kw)


def integrated(path):
    err = run(["ffmpeg", "-hide_banner", "-i", path, "-af", "ebur128=framelog=quiet", "-f", "null", "-"]).stderr.decode()
    return float(re.findall(r"I:\s+(-?[\d.]+) LUFS", err)[-1])


raw = ["-f", "f32le", "-ar", str(SR), "-ac", "1"]
for role, fx in FX.items():
    src, dst = f"{WORK}/track_{role}.f32", f"{WORK}/fx_{role}.wav"
    run(["ffmpeg", "-y", "-v", "error", *raw, "-i", src, "-af", fx, "-c:a", "pcm_f32le", dst])
    lufs = integrated(dst)
    gain = TARGET[role] - lufs
    run(["ffmpeg", "-y", "-v", "error", "-i", dst, "-af", f"volume={gain:.2f}dB", "-c:a", "pcm_f32le", f"{WORK}/lv_{role}.wav"])
    print(f"{role:9s} {lufs:6.1f} LUFS -> gain {gain:+.1f} dB")

# voice stem = sum of roles
run(["ffmpeg", "-y", "-v", "error", *sum([["-i", f"{WORK}/lv_{r}.wav"] for r in FX], []),
     "-filter_complex", f"amix=inputs={len(FX)}:normalize=0,alimiter=limit=0.89:level=false",
     "-c:a", "pcm_f32le", f"{WORK}/voice.wav"])

# timeline: play pieces (source span at a speed) and freeze frames, in source order
src_dur = float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", VIDEO]).stdout)
tl = json.load(open(f"{WORK}/timeline.json"))
holds = sorted(tuple(h) for h in tl["holds"] if h[1] > 0)
pieces = []
for a, b, sp in tl["sections"]:
    b = src_dur if b is None else b
    pos = a
    for h, d in holds:
        if a <= h < b:
            if h > pos:
                pieces.append(("play", pos, h, sp))
            pieces.append(("hold", h, d))
            pos = h
    pieces.append(("play", pos, b, sp))
frames = lambda t: round(t * FPS)
dur = sum((frames(p[2]) - frames(p[1])) / FPS / p[3] if p[0] == "play" else frames(p[2]) / FPS for p in pieces)


def video_graph():
    n = len(pieces)
    out = [f"[0:v]split={n}" + "".join(f"[x{i}]" for i in range(n))]
    for i, p in enumerate(pieces):
        if p[0] == "play":
            out.append(f"[x{i}]trim=start_frame={frames(p[1])}:end_frame={frames(p[2])},setpts=(PTS-STARTPTS)/{p[3]}[y{i}]")
        else:
            k = frames(p[2])
            out.append(f"[x{i}]trim=start_frame={frames(p[1])}:end_frame={frames(p[1]) + 1},setpts=PTS-STARTPTS,"
                       f"loop=loop={k - 1}:size=1:start=0,setpts=N/{FPS}/TB[y{i}]")
    out.append("".join(f"[y{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0,fps={FPS}[vout]")
    return ";".join(out)


def ambience():
    """Source soundtrack on the new timeline; freeze frames get looped room tone with crossfades."""
    src = np.frombuffer(run(["ffmpeg", "-v", "error", "-i", VIDEO, "-vn", "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"]).stdout,
                        np.float32).reshape(-1, 2)
    tone = src[round(ROOM_TONE[0] * SR):round(ROOM_TONE[1] * SR)]
    x = round(XFADE * SR)
    ramp = np.sin(np.linspace(0, np.pi / 2, x))[:, None].astype(np.float32)  # equal-power

    def loop_tone(n):
        out, ov = tone.copy(), round(0.25 * SR)
        r = np.sin(np.linspace(0, np.pi / 2, ov))[:, None].astype(np.float32)
        while len(out) < n:
            out = np.concatenate([out[:-ov], out[-ov:] * r[::-1] + tone[:ov] * r, tone[ov:]])
        return out[:n]

    out = np.zeros((round(dur * SR) + SR, 2), np.float32)
    pos = 0
    for i, p in enumerate(pieces):
        if p[0] == "play":
            seg = src[round(p[1] * SR):round(p[2] * SR)]
            if p[3] != 1:
                seg = np.frombuffer(run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-",
                                         "-af", f"atempo={p[3]}", "-f", "f32le", "-"], input=seg.tobytes()).stdout,
                                    np.float32).reshape(-1, 2).copy()
            want = round((frames(p[2]) - frames(p[1])) / FPS / p[3] * SR)
            seg = np.pad(seg, ((0, max(0, want - len(seg))), (0, 0)), mode="reflect")[:want].copy()
            if i > 0 and pieces[i - 1][0] == "hold":
                seg[:x] *= ramp
            if i + 1 < len(pieces) and pieces[i + 1][0] == "hold":
                seg[-x:] *= ramp[::-1]
            out[pos:pos + len(seg)] += seg
            pos += len(seg)
        else:
            n = round(frames(p[2]) / FPS * SR)
            fill = loop_tone(n + 2 * x)
            fill[:x] *= ramp
            fill[-x:] *= ramp[::-1]
            out[pos - x:pos + n + x] += fill
            pos += n
    return out[:round(dur * SR)]


ambience().tofile(f"{WORK}/ambience.f32")
graph = (
    "[1:a]aformat=sample_rates=48000:channel_layouts=mono,asplit=2[v][sc];"
    "[0:a]volume=-1dB[bg];"
    "[bg][sc]sidechaincompress=threshold=0.045:ratio=2:attack=20:release=450:knee=4[duck];"
    "[v]pan=stereo|c0=c0|c1=c0[vs];"
    f"[duck][vs]amix=inputs=2:normalize=0,atrim=0:{dur}[mix]"
)
mix_wav = f"{WORK}/mix.wav"
run(["ffmpeg", "-y", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", f"{WORK}/ambience.f32",
     "-i", f"{WORK}/voice.wav", "-filter_complex", graph, "-map", "[mix]", "-c:a", "pcm_f32le", mix_wav])

# static gain to -14 LUFS (short-form video platforms), then a peak limiter with headroom for AAC overshoot
gain = -14.0 - integrated(mix_wav)
for _ in range(3):
    master = f"volume={gain:.2f}dB,alimiter=limit=0.708:attack=2:release=80:level=false:asc=1"
    run(["ffmpeg", "-y", "-v", "error", "-i", mix_wav, "-af", master, "-c:a", "pcm_f32le", f"{WORK}/master.wav"])
    got = integrated(f"{WORK}/master.wav")
    if abs(got + 14.0) < 0.15:
        break
    gain += -14.0 - got
run(["ffmpeg", "-y", "-v", "error", "-i", VIDEO, "-i", f"{WORK}/master.wav", "-filter_complex", video_graph(),
     "-map", "[vout]", "-map", "1:a", "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-maxrate", "3M",
     "-bufsize", "6M", "-pix_fmt", "yuv420p", "-profile:v", "high", "-c:a", "aac", "-b:a", "192k",
     "-metadata:s:a:0", "language=rus", "-movflags", "+faststart", "-t", f"{dur:.3f}", OUT])

# voice-only stem
run(["ffmpeg", "-y", "-v", "error", "-i", f"{WORK}/voice.wav", "-af", f"atrim=0:{dur}", "-c:a", "pcm_s16le", STEM])
print(f"pieces: {[(p[0], round(p[1], 3), round(p[2], 3)) for p in pieces]}")
print(f"duration {dur:.3f}s, mix {got:.2f} LUFS after {gain:+.2f} dB -> {OUT}")
