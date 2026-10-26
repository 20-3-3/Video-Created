"""Process the per-role voice tracks, duck the original soundtrack under them and mux with the video."""
import json, re, subprocess, sys

WORK, VIDEO, OUT, STEM = sys.argv[1:5]
SR = 48000

FX = {
    # close, dry storyteller: trim rumble, a little warmth, gentle leveling
    "narrator": "highpass=f=70,equalizer=f=140:t=q:w=1.2:g=2.5,equalizer=f=3500:t=q:w=1.5:g=1.5,"
                "acompressor=threshold=-22dB:ratio=3:attack=5:release=150:makeup=2",
    # platform loudspeaker: narrow band, slight overdrive, hall reflections
    "pa": "highpass=f=320,lowpass=f=3600,equalizer=f=1700:t=q:w=1:g=5,volume=8dB,asoftclip=type=tanh,"
          "volume=-6dB,aecho=0.85:0.6:75|170|290|450:0.35|0.22|0.14|0.08",
    # attendant standing on the platform: a little room around the voice
    "attendant": "highpass=f=90,equalizer=f=200:t=q:w=1:g=1.5,"
                 "acompressor=threshold=-22dB:ratio=2.5:attack=5:release=150:makeup=1.5,"
                 "aecho=0.9:0.35:55|120|210:0.16|0.09|0.05",
}
TARGET = {"narrator": -16.0, "pa": -17.5, "attendant": -16.5}  # LUFS of speech (gated)


def run(args, **kw):
    return subprocess.run(args, capture_output=True, text=True, check=True, **kw)


def integrated(path, fmt_args):
    err = run(["ffmpeg", "-hide_banner", *fmt_args, "-i", path, "-af", "ebur128=framelog=quiet", "-f", "null", "-"]).stderr
    return float(re.findall(r"I:\s+(-?[\d.]+) LUFS", err)[-1])


raw = ["-f", "f32le", "-ar", str(SR), "-ac", "1"]
for role, fx in FX.items():
    src, dst = f"{WORK}/track_{role}.f32", f"{WORK}/fx_{role}.wav"
    run(["ffmpeg", "-y", "-v", "error", *raw, "-i", src, "-af", fx, "-c:a", "pcm_f32le", dst])
    lufs = integrated(dst, [])
    gain = TARGET[role] - lufs
    run(["ffmpeg", "-y", "-v", "error", "-i", dst, "-af", f"volume={gain:.2f}dB", "-c:a", "pcm_f32le", f"{WORK}/lv_{role}.wav"])
    print(f"{role:9s} {lufs:6.1f} LUFS -> gain {gain:+.1f} dB")

# voice stem = sum of roles
run(["ffmpeg", "-y", "-v", "error", *sum([["-i", f"{WORK}/lv_{r}.wav"] for r in FX], []),
     "-filter_complex", f"amix=inputs={len(FX)}:normalize=0,alimiter=limit=0.89:level=false",
     "-c:a", "pcm_f32le", f"{WORK}/voice.wav"])

dur = float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", VIDEO]).stdout)
graph = (
    "[1:a]aformat=sample_rates=48000:channel_layouts=mono,asplit=2[v][sc];"
    "[0:a]aformat=sample_rates=48000:channel_layouts=stereo,volume=-1dB[bg];"
    "[bg][sc]sidechaincompress=threshold=0.045:ratio=2:attack=20:release=450:knee=4[duck];"
    "[v]pan=stereo|c0=c0|c1=c0[vs];"
    f"[duck][vs]amix=inputs=2:normalize=0,atrim=0:{dur}[mix]"
)
mix_wav = f"{WORK}/mix.wav"
run(["ffmpeg", "-y", "-v", "error", "-i", VIDEO, "-i", f"{WORK}/voice.wav", "-filter_complex", graph,
     "-map", "[mix]", "-c:a", "pcm_f32le", mix_wav])

# static gain to -14 LUFS (short-form video platforms), then a peak limiter with headroom for AAC overshoot
gain = -14.0 - integrated(mix_wav, [])
for _ in range(3):
    master = f"volume={gain:.2f}dB,alimiter=limit=0.708:attack=2:release=80:level=false:asc=1"
    run(["ffmpeg", "-y", "-v", "error", "-i", mix_wav, "-af", master, "-c:a", "pcm_f32le", f"{WORK}/master.wav"])
    got = integrated(f"{WORK}/master.wav", [])
    if abs(got + 14.0) < 0.15:
        break
    gain += -14.0 - got
run(["ffmpeg", "-y", "-v", "error", "-i", VIDEO, "-i", f"{WORK}/master.wav", "-map", "0:v", "-map", "1:a",
     "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-metadata:s:a:0", "language=rus",
     "-movflags", "+faststart", "-t", str(dur), OUT])
m = {"input_i": f"{got:.2f} after {gain:+.2f} dB"}

# voice-only stem, same loudness offset as in the final mix
run(["ffmpeg", "-y", "-v", "error", "-i", f"{WORK}/voice.wav", "-af", f"atrim=0:{dur}", "-c:a", "pcm_s16le", STEM])
print("mix input:", m["input_i"], "LUFS; done ->", OUT)
