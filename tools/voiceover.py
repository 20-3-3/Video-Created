"""Build a voice-over track for stanciya_ep1 aligned to the burned-in kinetic captions.

Each line is synthesized whole (edge-tts, with word boundaries) so intonation stays natural,
then cut into chunks at caption-group boundaries. Every chunk starts when its first word
appears on screen and is gently time-stretched (rubberband) so its words track the captions.
"""
import asyncio
import hashlib
import json
import os
import re
import ssl
import subprocess
import sys

import edge_tts
import edge_tts.communicate as edge_comm
import numpy as np

edge_comm._SSL_CTX = ssl.create_default_context(cafile="/root/.ccr/ca-bundle.crt")

SR = 48000
WORK = sys.argv[1]
os.makedirs(f"{WORK}/cache", exist_ok=True)
os.makedirs(f"{WORK}/chunks", exist_ok=True)

ROLES = {
    "narrator": dict(voice=os.environ.get("NARRATOR_VOICE", "en-US-AndrewMultilingualNeural"),
                     pitch=os.environ.get("NARRATOR_PITCH", "-8Hz"), base_rate=-14, rates=(-18, -10)),
    "pa": dict(voice="ru-RU-SvetlanaNeural", pitch="+0Hz", base_rate=-10, rates=(-15, -5)),
    "attendant": dict(voice="en-US-EmmaMultilingualNeural", pitch="-10Hz", base_rate=-16, rates=(-20, -12)),
}

# (role, window_start, window_end, [(chunk text for TTS, [on-screen onset of each word])])
LINES = [
    ("narrator", 0.3, 3.4, [
        ("Ты засыпаешь…", [0.3, 0.6]),
        ("в последнем поезде.", [1.4, None, 2.4])]),
    ("narrator", 3.7, 8.2, [
        ("А просыпаешься…", [3.7]),
        ("на станции,", [5.0]),
        ("которой…", [6.0]),
        ("нет на схеме.", [6.8])]),
    ("pa", 11.3, 17.9, [
        ("Станция «Глубинная».", [11.3, 12.0]),
        ("Поезд дальше не идёт.", [13.2, 13.8, 14.2, 14.4]),
        ("Просьба освободить вагоны.", [15.2, 15.8, 16.8])]),
    ("narrator", 18.3, 21.5, [
        ("Ты выходишь.", [18.3, 18.6]),
        ("Двери закрываются.", [19.4, 19.9])]),
    ("narrator", 22.5, 27.3, [
        ("И ты видишь:", [22.6, None, 23.0]),
        ("в кабине машиниста…", [23.6, None, 24.6]),
        ("никого нет.", [26.0])]),
    ("narrator", 28.1, 34.2, [
        ("Табло над тоннелем считает:", [28.1, 28.6, None, 29.4]),
        ("следующий поезд —", [30.4, 31.2]),
        ("через тридцать три минуты.", [32.0, 32.4, 32.8])]),
    ("narrator", 34.4, 35.6, [
        ("Последний.", [34.4])]),
    ("narrator", 36.4, 43.2, [
        ("Ты думаешь:", [36.4]),
        ("техническая станция.", [37.6, 38.6]),
        ("Сейчас придёт дежурная,", [39.6, 40.0, 40.4]),
        ("и выведет наверх.", [41.4, None, 42.4])]),
    ("narrator", 43.8, 47.5, [
        ("На стене —", [43.8]),
        ("памятка.", [44.6]),
        ("Четыре правила.", [45.6, 46.4])]),
    ("narrator", 52.1, 57.1, [
        ("А внизу кто-то дописал от руки:", [52.1, 52.4, 52.8, 53.2, 54.0]),
        ("«Второе правило —", [54.8, 55.5]),
        ("ложь».", [56.3])]),
    ("narrator", 59.5, 64.8, [
        ("Из перехода выходит дежурная.", [59.5, None, 60.6, 61.4]),
        ("И задаёт один вопрос.", [62.4, None, 63.4, 63.8])]),
    ("attendant", 65.1, 69.9, [
        ("Молодой человек…", [65.1, 65.8]),
        ("какая сегодня станция?", [67.0, 67.8, 68.6])]),
    ("narrator", 70.6, 75.5, [
        ("Правило на стене или надпись от руки.", [70.6]),
        ("Кому ты веришь?", [74.0])]),
]

STRETCH_MIN, STRETCH_MAX = 0.87, 1.15  # duration scale limits applied by rubberband
WORD_LAG = 0.06  # decoded audio starts this much after edge-tts word offsets
MIN_GAP = 0.06


def letters(s):
    return re.sub(r"[^a-zа-яё]", "", s.lower())


def ffmpeg_decode(path, sr=SR):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def ffmpeg_filter(x, af):
    p = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-",
                        "-af", af, "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                       input=x.astype(np.float32).tobytes(), capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.float32).copy()


async def _synth(text, voice, rate, pitch, mp3_path):
    comm = edge_tts.Communicate(text, voice, rate=f"{rate:+d}%", pitch=pitch, boundary="WordBoundary")
    words = []
    with open(mp3_path, "wb") as f:
        async for ev in comm.stream():
            if ev["type"] == "audio":
                f.write(ev["data"])
            elif ev["type"] == "WordBoundary":
                words.append(dict(text=ev["text"], start=ev["offset"] / 1e7 + WORD_LAG, dur=ev["duration"] / 1e7))
    return words


def synth(text, role, rate):
    cfg = ROLES[role]
    key = hashlib.sha1(f"{cfg['voice']}|{rate}|{cfg['pitch']}|{text}".encode()).hexdigest()[:16]
    mp3, meta = f"{WORK}/cache/{key}.mp3", f"{WORK}/cache/{key}.json"
    if not os.path.exists(meta):
        for attempt in range(4):
            try:
                words = asyncio.run(_synth(text, cfg["voice"], rate, cfg["pitch"], mp3))
                break
            except Exception as e:  # transient websocket failures
                print("  retry", attempt, e, file=sys.stderr)
        else:
            raise RuntimeError(f"TTS failed: {text}")
        json.dump(words, open(meta, "w"), ensure_ascii=False)
    words = [w for w in json.load(open(meta)) if letters(w["text"])]
    return ffmpeg_decode(mp3), words


def group_words(chunks, words):
    """Assign TTS word boundaries to chunks by cumulative letter count."""
    targets = np.cumsum([len(letters(t)) for t, _ in chunks])
    groups, acc, k = [[] for _ in chunks], 0, 0
    for w in words:
        groups[k].append(w)
        acc += len(letters(w["text"]))
        if acc >= targets[k] and k < len(chunks) - 1:
            k += 1
    assert acc == targets[-1], (acc, targets, [w["text"] for w in words])
    return groups


def rms_db(x, i0, i1, win=int(0.01 * SR)):
    seg = x[i0:i1]
    n = max(1, len(seg) // win)
    fr = seg[:n * win].reshape(n, win) if len(seg) >= win else seg.reshape(1, -1)
    return 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-9), win


def cut_point(x, t0, t1):
    """Quietest 10 ms frame between the end of one word and the start of the next."""
    i0, i1 = int(t0 * SR), int(t1 * SR)
    if i1 - i0 < int(0.03 * SR):
        return max(i0, i1 - int(0.01 * SR))
    db, win = rms_db(x, i0, i1)
    return i0 + int(np.argmin(db)) * win + win // 2


def trim(x, thr_db=-50, pad=0.015):
    db, win = rms_db(x, 0, len(x))
    loud = np.where(db > thr_db)[0]
    if len(loud) == 0:
        return x, 0
    a = max(0, loud[0] * win - int(pad * SR))
    b = min(len(x), (loud[-1] + 1) * win + int(pad * SR))
    return x[a:b], a


def fade(x, ms=8):
    n = min(len(x) // 2, int(ms / 1000 * SR))
    if n:
        ramp = np.linspace(0, 1, n, dtype=np.float32)
        x = x.copy()
        x[:n] *= ramp
        x[-n:] *= ramp[::-1]
    return x


def split_line(audio, chunks, words):
    groups = group_words(chunks, words)
    bounds = [0]
    for g, nxt in zip(groups[:-1], groups[1:]):
        end_prev = g[-1]["start"] + g[-1]["dur"]
        bounds.append(cut_point(audio, end_prev, nxt[0]["start"]))
    bounds.append(len(audio))
    out = []
    for k, g in enumerate(groups):
        seg = audio[bounds[k]:bounds[k + 1]]
        seg, lead = trim(seg)
        t0 = (bounds[k] + lead) / SR
        # word onsets relative to the trimmed chunk start
        rel = [max(0.0, w["start"] - t0) for w in g]
        onsets = list(chunks[k][1]) + [None] * (len(g) - len(chunks[k][1]))
        out.append(dict(audio=seg, rel=rel, text=chunks[k][0], onsets=onsets))
    return out


def anchor_pairs(p):
    """(TTS offset, caption offset) for words with a known on-screen onset, relative to the chunk start."""
    t = np.array([r - p["rel"][0] for r, o in zip(p["rel"][1:], p["onsets"][1:]) if o is not None])
    c = np.array([o - p["onsets"][0] for o in p["onsets"][1:] if o is not None])
    return t, c


def fit_scale(p):
    """Least-squares duration scale mapping TTS word offsets onto caption onsets."""
    t, c = anchor_pairs(p)
    if len(t) == 0 or (t ** 2).sum() < 1e-6:
        return None
    return float((t * c).sum() / (t ** 2).sum())


def line_scale(parts):
    num = den = 0.0
    for p in parts:
        t, c = anchor_pairs(p)
        num += (t * c).sum()
        den += (t ** 2).sum()
    return num / den if den > 1e-6 else None


def stretch(x, scale):
    if abs(scale - 1) < 0.015:
        return x
    return ffmpeg_filter(x, f"rubberband=tempo={1 / scale:.4f}:window=short:formant=preserved:pitchq=quality")


def build():
    total = 80.0
    tracks = {r: np.zeros(int(total * SR), np.float32) for r in ROLES}
    report = []
    for li, (role, w0, w1, chunks) in enumerate(LINES):
        text = " ".join(t for t, _ in chunks)
        rate = ROLES[role]["base_rate"]
        audio, words = synth(text, role, rate)
        parts = split_line(audio, chunks, words)
        # pick a speaking rate so the line as a whole needs little stretching
        s = line_scale(parts)
        speech = sum(len(p["audio"]) for p in parts) / SR
        slots = sum((chunks[k + 1][1][0] if k + 1 < len(chunks) else w1) - chunks[k][1][0] for k in range(len(chunks)))
        if s is None or s * speech > 0.97 * slots:
            s = min(s or 1.0, 0.92 * slots / speech)
        new_rate = int(round(np.clip((1 + rate / 100) / s * 100 - 100, *ROLES[role]["rates"])))
        print(f"L{li + 1:02d} fit: scale={s:.2f} speech={speech:.2f}s slots={slots:.2f}s rate {rate:+d}% -> {new_rate:+d}%")
        if abs(new_rate - rate) >= 2:
            rate = new_rate
            audio, words = synth(text, role, rate)
            parts = split_line(audio, chunks, words)

        cursor = 0.0
        for k, p in enumerate(parts):
            start = max(p["onsets"][0], cursor + MIN_GAP if cursor else 0)
            limit = (parts[k + 1]["onsets"][0] if k + 1 < len(parts) else w1) - 0.05
            dur = len(p["audio"]) / SR
            sc = fit_scale(p) or 1.0
            sc = float(np.clip(sc, STRETCH_MIN, STRETCH_MAX))
            if start + dur * sc > limit:
                sc = max((limit - start) / dur, 0.7)
            seg = fade(stretch(p["audio"], sc))
            i = int(start * SR)
            tracks[role][i:i + len(seg)] += seg[:len(tracks[role]) - i]
            end = start + len(seg) / SR
            spoken = [round(start + r * sc, 2) for r in p["rel"]]
            report.append(dict(line=li + 1, role=role, rate=rate, text=p["text"], start=round(start, 2),
                               end=round(end, 2), stretch=round(sc, 3), captions=p["onsets"], spoken=spoken))
            cursor = end
            flag = "  <-- over window" if end > w1 + 0.05 else ""
            print(f"L{li + 1:02d} {role:9s} rate={rate:+3d}% {start:6.2f}-{end:6.2f} x{sc:.2f} "
                  f"words@{spoken} caps@{p['onsets']} | {p['text']}{flag}")
    for r, x in tracks.items():
        x.tofile(f"{WORK}/track_{r}.f32")
    json.dump(report, open(f"{WORK}/alignment.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    build()
