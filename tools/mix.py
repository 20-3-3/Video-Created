"""Process the per-role voice tracks, duck the original soundtrack under them and mux with the video."""
import json, os, re, subprocess, sys

WORK, VIDEO, OUT, STEM = sys.argv[1:5]
SR = 48000
NARRATOR_SHIFT = float(os.environ.get("NARRATOR_SHIFT", "0.82"))  # pitch ratio, formants move with it


def narrator_graph(pitch, sub_db=-16.0, flanger_width=22):
    """Deep, velvety, slightly synthetic announcer."""
    return (
        f"[0:a]rubberband=pitch={pitch}:formant=shifted:pitchq=quality,"
        "highpass=f=40,bass=g=3.5:f=110:w=0.7,equalizer=f=420:t=q:w=1.2:g=-2.5,"
        "equalizer=f=3000:t=q:w=1.4:g=-2,treble=g=-3.5:f=7500:w=0.6,deesser=i=0.35,"
        "acompressor=threshold=-24dB:ratio=3:attack=10:release=200:makeup=3,asplit=2[dry][s];"
        # sub-octave double: adds weight and a faint synthetic edge
        f"[s]rubberband=pitch=0.5:formant=shifted:pitchq=quality,lowpass=f=900,volume={sub_db}dB[sub];"
        "[dry][sub]amix=inputs=2:normalize=0,"
        # slow, shallow flanger sweep: the 'electronic' sheen
        f"flanger=delay=1.5:depth=1.2:regen=15:width={flanger_width}:speed=0.25:shape=sinusoidal,"
        "alimiter=limit=0.9:level=false"
    )


FX = {
    "narrator": narrator_graph(NARRATOR_SHIFT),
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
    run(["ffmpeg", "-y", "-v", "error", *raw, "-i", src, "-filter_complex", fx, "-c:a", "pcm_f32le", dst])
    lufs = integrated(dst, [])
    gain = TARGET[role] - lufs
    run(["ffmpeg", "-y", "-v", "error", "-i", dst, "-af", f"volume={gain:.2f}dB", "-c:a", "pcm_f32le", f"{WORK}/lv_{role}.wav"])
    print(f"{role:9s} {lufs:6.1f} LUFS -> gain {gain:+.1f} dB")

# voice stem = sum of roles
run(["ffmpeg", "-y", "-v", "error", *sum([["-i", f"{WORK}/lv_{r}.wav"] for r in FX], []),
     "-filter_complex", f"amix=inputs={len(FX)}:normalize=0,alimiter=limit=0.89:level=false",
     "-c:a", "pcm_f32le", f"{WORK}/voice.wav"])

src_dur = float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", VIDEO]).stdout)

# per-section playback speed of the source (written by voiceover.py); the voice tracks are already on the new timeline
TIMELINE = [(a, src_dur if b is None else b, sp) for a, b, sp in json.load(open(f"{WORK}/timeline.json"))]
dur = sum((b - a) / sp for a, b, sp in TIMELINE)
n = len(TIMELINE)
FPS = 30


def sectioned(kind):
    """Filtergraph that plays each section of the source video/audio at its own speed and joins them."""
    if kind == "v":
        parts = [f"[x{i}]trim=start_frame={round(a * FPS)}:end_frame={round(b * FPS)},setpts=(PTS-STARTPTS)/{sp}[y{i}]"
                 for i, (a, b, sp) in enumerate(TIMELINE)]
        return (f"[0:v]split={n}" + "".join(f"[x{i}]" for i in range(n)) + ";" + ";".join(parts) + ";"
                + "".join(f"[y{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0,fps={FPS}[vout]")
    parts = [f"[b{i}]atrim=start_sample={round(a * SR)}:end_sample={round(b * SR)},asetpts=PTS-STARTPTS"
             + (f",atempo={sp}" if sp != 1 else "") + f"[c{i}]" for i, (a, b, sp) in enumerate(TIMELINE)]
    return ("[0:a]aformat=sample_rates=48000:channel_layouts=stereo,"
            f"asplit={n}" + "".join(f"[b{i}]" for i in range(n)) + ";" + ";".join(parts) + ";"
            + "".join(f"[c{i}]" for i in range(n)) + f"concat=n={n}:v=0:a=1,volume=-1dB[bg]")


graph = (
    "[1:a]aformat=sample_rates=48000:channel_layouts=mono,asplit=2[v][sc];"
    + sectioned("a") + ";"
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
run(["ffmpeg", "-y", "-v", "error", "-i", VIDEO, "-i", f"{WORK}/master.wav", "-filter_complex", sectioned("v"),
     "-map", "[vout]", "-map", "1:a", "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-maxrate", "3M",
     "-bufsize", "6M", "-pix_fmt", "yuv420p", "-profile:v", "high", "-c:a", "aac", "-b:a", "192k",
     "-metadata:s:a:0", "language=rus", "-movflags", "+faststart", "-t", f"{dur:.3f}", OUT])
m = {"input_i": f"{got:.2f} after {gain:+.2f} dB"}

# voice-only stem, same loudness offset as in the final mix
run(["ffmpeg", "-y", "-v", "error", "-i", f"{WORK}/voice.wav", "-af", f"atrim=0:{dur}", "-c:a", "pcm_s16le", STEM])
print("mix input:", m["input_i"], "LUFS; done ->", OUT)
