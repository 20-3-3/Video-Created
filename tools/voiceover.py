"""Build a voice-over track for stanciya_ep1 aligned to the burned-in kinetic captions.

Each line is synthesized whole so intonation stays natural, then cut into chunks at caption-group
boundaries. Every chunk starts when its first word appears on screen and is gently time-stretched
(atempo) so its words track the captions.

TTS_ENGINE picks the synthesizer: "edge" (edge-tts, reports word boundaries itself) or "google"
(Google Cloud Text-to-Speech, key in GOOGLE_TTS_API_KEY; Chirp 3 HD voices report no word timings,
so the words are located in the audio with faster-whisper).
"""
import asyncio
import base64
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
import requests

edge_comm._SSL_CTX = ssl.create_default_context(cafile="/root/.ccr/ca-bundle.crt")

SR = 48000
WORK = None  # output directory, set in __main__

ENGINE = os.environ.get("TTS_ENGINE", "edge")
ROLES = {
    "narrator": dict(voice=os.environ.get("NARRATOR_VOICE", "en-US-AndrewMultilingualNeural"),
                     pitch=os.environ.get("NARRATOR_PITCH", "-6Hz"), base_rate=-14, rates=(-18, -10)),
    "pa": dict(voice="ru-RU-SvetlanaNeural", pitch="+0Hz", base_rate=-10, rates=(-15, -5)),
    "attendant": dict(voice="en-US-EmmaMultilingualNeural", pitch="-10Hz", base_rate=-16, rates=(-20, -12)),
}
if ENGINE == "google":
    # Chirp 3 HD takes no pitch; voice colour is left to the role chains in mix.py
    GOOGLE_VOICES = dict(narrator=os.environ.get("NARRATOR_VOICE", "ru-RU-Chirp3-HD-Charon"),
                         pa=os.environ.get("PA_VOICE", "ru-RU-Chirp3-HD-Kore"),
                         attendant=os.environ.get("ATTENDANT_VOICE", "ru-RU-Chirp3-HD-Gacrux"))
    for role, voice in GOOGLE_VOICES.items():
        ROLES[role].update(voice=voice, pitch="")

# Playback speed per section of the source video (start, end in seconds, speed); the platform
# announcement scene (frames 314-542) keeps its original pace. Times in LINES are source times.
SPEED = float(os.environ.get("SPEEDUP", "1.1"))
SECTIONS = [(0.0, 314 / 30, SPEED), (314 / 30, 543 / 30, 1.0), (543 / 30, None, SPEED)]

# The passenger memo is read rule by rule: a rule starts when its highlight starts to sweep
# (first frame number) and the picture freezes on the finished highlight (second frame number)
# until the rule has been read.
MEMO = [
    ("Первое: не садитесь в вагон без света.", 1427, 1439),
    ("Второе: если дежурная спросит, какая сегодня станция, — не отвечайте.", 1457, 1469),
    ("Третье: не поднимайтесь по эскалатору, который едет вниз.", 1487, 1497),
    ("Четвёртое: поезд приходит один раз.", 1517, 1528),
]
MEMO_PRE_HOLD = (1425, 0.3)  # breath after "Четыре правила." before the first rule
MEMO_NEXT = 52.1  # source time of the narrator line that follows the memo
MEMO_PAUSE = 0.45  # silence after each rule
HOLDS = []  # (source time, seconds) freeze frames, sized by plan_memo()


def remap(t):
    """Source-video time -> output time."""
    out = sum(d for h, d in HOLDS if h < t)
    for a, b, sp in SECTIONS:
        if t <= a:
            break
        out += (min(t, b if b is not None else t) - a) / sp
    return out


def speed_at(t):
    return next(sp for a, b, sp in SECTIONS if b is None or t < b)


def to_rate(rate, speed):
    """Speaking rate that keeps a section's speaker in step with that section's playback speed."""
    return int(round(((1 + rate / 100) * speed - 1) * 100))

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

STRETCH_MIN, STRETCH_MAX = 0.92, 1.08  # duration scale limits applied by atempo
WORD_LAG = 0.06  # decoded audio starts this much after edge-tts word offsets
MIN_GAP = 0.06
# aligned word times can overlap the pause between chunks, so the quietest point is searched a bit wider
CUT_SLACK = 0.12 if ENGINE == "google" else 0.0


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


def _google_synth(text, voice, rate, wav_path):
    r = requests.post("https://texttospeech.googleapis.com/v1/text:synthesize",
                      headers={"X-Goog-Api-Key": os.environ["GOOGLE_TTS_API_KEY"]}, timeout=120,
                      json=dict(input=dict(text=text), voice=dict(languageCode=voice[:5], name=voice),
                                audioConfig=dict(audioEncoding="LINEAR16", sampleRateHertz=SR,
                                                 speakingRate=round(1 + rate / 100, 3))))
    if r.status_code != 200:
        raise RuntimeError(f"Google TTS {r.status_code}: {r.text[:500]}")
    with open(wav_path, "wb") as f:
        f.write(base64.b64decode(r.json()["audioContent"]))
    return align_words(wav_path, text)


_whisper = None


def align_words(path, text):
    """Word timings for a take of `text`: Whisper word timestamps matched to the script letter by letter.

    Letters Whisper heard differently (or wrote as digits) get no time of their own; script words left
    without any matched letter are interpolated between their neighbours.
    """
    global _whisper
    from difflib import SequenceMatcher
    from faster_whisper import WhisperModel
    if _whisper is None:
        _whisper = WhisperModel(os.environ.get("WHISPER_MODEL", "medium"), device="cpu", compute_type="int8")
    segs, _ = _whisper.transcribe(ffmpeg_decode(path, 16000), language="ru", word_timestamps=True,
                                  initial_prompt=text, beam_size=5, condition_on_previous_text=False)
    heard = [w for s in segs for w in s.words]
    # time at which each heard letter starts and ends, spread evenly over its word
    h_chars, h_t0, h_t1 = [], [], []
    for w in heard:
        ls = letters(w.word)
        step = (w.end - w.start) / max(1, len(ls))
        for k, ch in enumerate(ls):
            h_chars.append(ch)
            h_t0.append(w.start + k * step)
            h_t1.append(w.start + (k + 1) * step)
    script = [t for t in text.split() if letters(t)]
    s_chars, owner = [], []
    for i, t in enumerate(script):
        for ch in letters(t):
            s_chars.append(ch)
            owner.append(i)
    start, end = [None] * len(script), [None] * len(script)
    for a, b, n in SequenceMatcher(None, s_chars, h_chars, autojunk=False).get_matching_blocks():
        for k in range(n):
            i = owner[a + k]
            if start[i] is None:
                start[i] = h_t0[b + k]
            end[i] = h_t1[b + k]
    known = [i for i in range(len(script)) if start[i] is not None]
    if not known:
        raise RuntimeError(f"alignment failed: {text!r} heard as {''.join(w.word for w in heard)!r}")
    for i in range(len(script)):
        if start[i] is None:  # spread the gap between the matched neighbours by letter count
            lo = max((j for j in known if j < i), default=None)
            hi = min((j for j in known if j > i), default=None)
            t0 = end[lo] if lo is not None else 0.0
            t1 = start[hi] if hi is not None else t0 + 0.08 * sum(len(letters(t)) for t in script[i:])
            span = range((lo + 1) if lo is not None else 0, hi if hi is not None else len(script))
            n = [len(letters(script[j])) for j in span]
            k = list(span).index(i)
            start[i] = t0 + (t1 - t0) * sum(n[:k]) / sum(n)
            end[i] = t0 + (t1 - t0) * sum(n[:k + 1]) / sum(n)
    # Whisper hands a pause (often with a breath in it) to the word after it: move each word's edges
    # inwards to where the level comes within 20 dB of that word's own peak (30 ms smoothing)
    db, win = rms_db(ffmpeg_decode(path), 0, None)
    db = 10 * np.log10(np.convolve(10 ** (db / 10), np.ones(3) / 3, mode="same") + 1e-12)
    for i in range(len(script)):
        a, b = int(start[i] * SR / win), min(len(db), int(end[i] * SR / win))
        if b - a < 4:
            continue
        thr = max(db.max() - 30, db[a:b].max() - 20)
        while a < b - 3 and db[a] < thr:
            a += 1
        while b > a + 3 and db[b - 1] < thr:
            b -= 1
        start[i], end[i] = a * win / SR, b * win / SR
    return [dict(text=t, start=s, dur=e - s) for t, s, e in zip(script, start, end)]


def synth(text, role, rate):
    cfg = ROLES[role]
    key = hashlib.sha1(f"{cfg['voice']}|{rate}|{cfg['pitch']}|{text}".encode()).hexdigest()[:16]
    audio, meta = f"{WORK}/cache/{key}.{'wav' if ENGINE == 'google' else 'mp3'}", f"{WORK}/cache/{key}.json"
    if not os.path.exists(meta):
        for attempt in range(4):
            try:
                if ENGINE == "google":
                    words = _google_synth(text, cfg["voice"], rate, audio)
                else:
                    words = asyncio.run(_synth(text, cfg["voice"], rate, cfg["pitch"], audio))
                break
            except Exception as e:  # transient websocket / HTTP failures
                print("  retry", attempt, e, file=sys.stderr)
        else:
            raise RuntimeError(f"TTS failed: {text}")
        json.dump(words, open(meta, "w"), ensure_ascii=False)
    words = [w for w in json.load(open(meta)) if letters(w["text"])]
    return ffmpeg_decode(audio), words


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
    t0, t1 = min(t0, t1) - CUT_SLACK, max(t0, t1) + CUT_SLACK
    i0, i1 = max(0, int(t0 * SR)), int(t1 * SR)
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
    # WSOLA (atempo) rather than a phase vocoder: no metallic colouring on speech
    if abs(scale - 1) < 0.02:
        return x
    return ffmpeg_filter(x, f"atempo={1 / scale:.4f}")


def plan_memo():
    """Synthesize the memo rules and size the freeze frames so each rule ends before the next begins."""
    HOLDS[:] = [(MEMO_PRE_HOLD[0] / 30, MEMO_PRE_HOLD[1])]
    lines = []
    onsets = [f / 30 + 0.05 for _, f, _ in MEMO] + [MEMO_NEXT]
    for k, (text, _, hold_f) in enumerate(MEMO):
        sp = speed_at(onsets[k])
        audio, _ = synth(text, "narrator", to_rate(ROLES["narrator"]["base_rate"], sp))
        dur = len(trim(audio)[0]) / SR
        need = dur + MEMO_PAUSE - (onsets[k + 1] - onsets[k]) / sp
        HOLDS.append((hold_f / 30, max(0.0, np.ceil(need * 30) / 30)))
        lines.append(("narrator", onsets[k], onsets[k + 1] - 0.05, [(text, [onsets[k]])], True))
    return lines


def build():
    lines = sorted(LINES + plan_memo(), key=lambda line: line[1])
    total = remap(80.0)
    tracks = {r: np.zeros(int(total * SR), np.float32) for r in ROLES}
    report = []
    for li, (role, w0, w1, chunks, *fixed) in enumerate(lines):
        # speeding a section up by k makes the speaker k times faster too
        sp = speed_at(w0)
        w0, w1 = remap(w0), remap(w1)
        chunks = [(t, [None if o is None else round(remap(o), 3) for o in ons]) for t, ons in chunks]
        rates = tuple(to_rate(r, sp) for r in ROLES[role]["rates"])
        text = " ".join(t for t, _ in chunks)
        rate = to_rate(ROLES[role]["base_rate"], sp)
        audio, words = synth(text, role, rate)
        parts = split_line(audio, chunks, words)
        if fixed:  # memo rules: the freeze frames were sized for this exact take
            rates = (rate, rate)
        # pick a speaking rate so the line as a whole needs little stretching
        s = line_scale(parts)
        speech = sum(len(p["audio"]) for p in parts) / SR
        slots = sum((chunks[k + 1][1][0] if k + 1 < len(chunks) else w1) - chunks[k][1][0] for k in range(len(chunks)))
        if s is None or s * speech > 0.97 * slots:
            s = min(s or 1.0, 0.92 * slots / speech)
        new_rate = int(round(np.clip((1 + rate / 100) / s * 100 - 100, *rates)))
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
    json.dump(dict(sections=SECTIONS, holds=HOLDS), open(f"{WORK}/timeline.json", "w"))


if __name__ == "__main__":
    WORK = sys.argv[1]
    os.makedirs(f"{WORK}/cache", exist_ok=True)
    os.makedirs(f"{WORK}/chunks", exist_ok=True)
    build()
