"""Narrator track for stanciya_ep1 from a recorded take instead of TTS.

usage: narrator_take.py WORK TAKE.wav TAKE_WORDS.json [MEMO_DIR]

WORK is the directory voiceover.py wrote (timeline.json with section speeds and freeze frames).
TAKE.wav is the denoised recording of the narrator lines, TAKE_WORDS.json its word timings
(align_take.py). The take is cut into the caption chunks of voiceover.LINES at the quietest point
between words; every chunk starts when its first word appears on screen and is gently
time-stretched (atempo) so it ends before the next chunk's caption. The passenger memo rules are
not in the take: MEMO_DIR holds takes of each rule read in the same voice (clone_memo.py); the
take closest to the recorded voice that still leaves a pause before the next rule is placed on
it, EQ'd and levelled to match the take. Writes WORK/track_narrator.f32 (replacing the TTS narrator),
WORK/alignment_take.json, and WORK/narrator_fx.txt, the chain mix.py then uses for this voice.
"""
import json
import os
import sys

import numpy as np

import voiceover as vo

SR = vo.SR
SQUEEZE_MIN = 0.8  # shortest a chunk may be squeezed to fit before the next caption
EDGE = 0.25  # search window (s) for the cut before the first and after the last word of a line
PAD = 0.03  # silence kept around a trimmed chunk so soft onsets and releases survive
CUT_MAX_DB = -30  # a cut between chunks must be this far below the speech level, else they stay joined
# recorded narrator (phone, close mic): no rumble, less boom and mud, more presence, broadcast leveling
TAKE_FX = ("highpass=f=70,equalizer=f=280:t=q:w=1:g=-3,equalizer=f=550:t=q:w=1.4:g=-1.5,"
           "equalizer=f=3500:t=q:w=1.2:g=3,treble=g=1:f=8000:w=0.6,deesser=i=0.3,"
           "acompressor=threshold=-24dB:ratio=3:attack=10:release=180:makeup=2.5")


def chunk_words(text):
    return [w for w in text.split() if vo.letters(w)]


def cut_take(take, words, chunks, speech_db):
    """Split one line of the take into its caption chunks (text, per-word caption onsets).

    Where the reading runs on without a pause (the quietest point between the chunks is still
    within CUT_MAX_DB of the speech level), the chunks stay together as one part.
    """
    groups, k = [], 0
    for text, onsets in chunks:
        n = len(chunk_words(text))
        groups.append(dict(words=words[k:k + n], text=text, onsets=onsets + [None] * (n - len(onsets))))
        k += n
    db, win = vo.rms_db(take, 0, len(take))
    merged, bounds = [groups[0]], [vo.cut_point(take, max(0.0, words[0]["start"] - EDGE), words[0]["start"])]
    for g in groups[1:]:
        cut = vo.cut_point(take, merged[-1]["words"][-1]["end"], g["words"][0]["start"])
        if db[min(cut // win, len(db) - 1)] > speech_db + CUT_MAX_DB:
            m = merged[-1]
            merged[-1] = dict(words=m["words"] + g["words"], text=f"{m['text']} {g['text']}",
                              onsets=m["onsets"] + g["onsets"])
        else:
            merged.append(g)
            bounds.append(cut)
    end = words[-1]["end"]
    bounds.append(vo.cut_point(take, end, min(len(take) / SR, end + EDGE)))
    out = []
    for k, g in enumerate(merged):
        seg, lead = vo.trim(take[bounds[k]:bounds[k + 1]], pad=PAD)
        t0 = (bounds[k] + lead) / SR
        out.append(dict(audio=seg, rel=[max(0.0, w["start"] - t0) for w in g["words"]], text=g["text"],
                        onsets=g["onsets"], words=[w["text"] for w in g["words"]]))
    return out


def speech_frames(x, n=4096):
    fr = np.lib.stride_tricks.sliding_window_view(x, n)[::n // 2]
    return fr[np.sqrt((fr ** 2).mean(axis=1)) > 10 ** (-40 / 20)]


def ltas(x, n=4096):
    """Long-term average spectrum over the frames that carry speech, smoothed to 1/3 octave."""
    p = (np.abs(np.fft.rfft(speech_frames(x, n) * np.hanning(n), axis=1)) ** 2).mean(axis=0)
    f = np.fft.rfftfreq(n, 1 / SR)
    lo, hi = np.searchsorted(f, f * 2 ** (-1 / 6)), np.searchsorted(f, f * 2 ** (1 / 6), side="right")
    c = np.concatenate([[0], np.cumsum(p)])
    return (c[np.maximum(hi, lo + 1)] - c[lo]) / np.maximum(hi - lo, 1)


def speech_rms(x):
    return float(np.sqrt((speech_frames(x) ** 2).mean()))


def match(x, take, n=4096):
    """EQ x towards the tonal balance of the take (within ±6 dB, 70 Hz-10 kHz), then match its speech level."""
    f = np.fft.rfftfreq(n, 1 / SR)
    gain = np.clip(10 * np.log10(ltas(take, n) / ltas(x, n)), -6, 6)
    gain[(f < 70) | (f > 10000)] = 0
    gain -= gain[np.searchsorted(f, 1000)]
    h = np.fft.fftshift(np.fft.irfft(10 ** (gain / 20), n)) * np.hanning(n)
    y = np.convolve(x, h, mode="same").astype(np.float32)
    return y * np.float32(speech_rms(take) / speech_rms(y))


def pick_memo_take(memo_dir, rule, slot):
    """Closest-sounding take of a memo rule that ends MEMO_PAUSE before the slot does (else the shortest)."""
    takes = []
    for t in json.load(open(os.path.join(memo_dir, "takes.json"))):
        if t["rule"] == rule:
            # stricter gate than for the take: generated takes can end in a faint blip after the speech
            audio, _ = vo.trim(vo.ffmpeg_decode(os.path.join(memo_dir, t["file"])), thr_db=-45, pad=0.06)
            takes.append((t["similarity"], len(audio) / SR, t["file"], audio))
    fits = [t for t in takes if t[1] <= slot - vo.MEMO_PAUSE]
    sim, dur, name, audio = max(fits) if fits else min(takes, key=lambda t: t[1])
    print(f"memo {rule}: {name} {dur:.2f}s in a {slot:.2f}s slot, similarity {sim:.3f}")
    return audio


def place(track, report, li, role, parts, w1):
    """Put chunks on their captions: start at the caption, stretch to follow it, never run into the next one."""
    cursor = 0.0
    for k, p in enumerate(parts):
        start = max(p["onsets"][0], cursor + vo.MIN_GAP if cursor else 0)
        limit = (parts[k + 1]["onsets"][0] if k + 1 < len(parts) else w1) - 0.05
        dur = len(p["audio"]) / SR
        sc = float(np.clip(vo.fit_scale(p) or 1.0, vo.STRETCH_MIN, vo.STRETCH_MAX))
        if start + dur * sc > limit:
            sc = max((limit - start) / dur, SQUEEZE_MIN)
        seg = vo.fade(vo.stretch(p["audio"], sc))
        i = int(start * SR)
        track[i:i + len(seg)] += seg[:len(track) - i]
        end = start + len(seg) / SR
        spoken = [round(start + r * sc, 2) for r in p["rel"]]
        report.append(dict(line=li, role=role, text=p["text"], start=round(start, 2), end=round(end, 2),
                           stretch=round(sc, 3), captions=p["onsets"], spoken=spoken))
        flag = "  <-- runs past the next caption" if end > limit + 0.06 else ""
        print(f"L{li:02d} {start:6.2f}-{end:6.2f} x{sc:.2f} words@{spoken} caps@{p['onsets']} | {p['text']}{flag}")
        cursor = end


def main():
    work, take_path, words_path = sys.argv[1:4]
    memo_dir = sys.argv[4] if len(sys.argv) > 4 else None
    tl = json.load(open(f"{work}/timeline.json"))
    vo.HOLDS[:] = [tuple(h) for h in tl["holds"]]
    take = vo.ffmpeg_decode(take_path)
    words = [w for w in json.load(open(words_path)) if vo.letters(w["text"])]

    lines = [(role, w0, w1, chunks) for role, w0, w1, chunks in vo.LINES if role == "narrator"]
    memo_onsets = [f / 30 + 0.05 for _, f, _ in vo.MEMO] + [vo.MEMO_NEXT]
    if memo_dir:
        lines += [("memo", memo_onsets[k], memo_onsets[k + 1] - 0.05, [(text, [memo_onsets[k]])])
                  for k, (text, _, _) in enumerate(vo.MEMO)]
    lines.sort(key=lambda line: line[1])

    track = np.zeros(int(vo.remap(80.0) * SR), np.float32)
    report, wi, mi = [], 0, 0
    speech_db = np.percentile(vo.rms_db(take, 0, len(take))[0], 99)
    for li, (role, w0, w1, chunks) in enumerate(lines, 1):
        chunks = [(t, [None if o is None else round(vo.remap(o), 3) for o in ons]) for t, ons in chunks]
        if role == "memo":
            mi += 1
            start = chunks[0][1][0]
            audio = match(pick_memo_take(memo_dir, mi, vo.remap(w1) - 0.05 - start), take)
            parts = [dict(audio=audio, rel=[0.0], text=chunks[0][0], onsets=[start])]
        else:
            n = sum(len(chunk_words(t)) for t, _ in chunks)
            parts = cut_take(take, words[wi:wi + n], chunks, speech_db)
            wi += n
            # the take may differ in a word ending (дежурный / дежурная), not in the words themselves
            for p in parts:
                assert [vo.letters(w)[:4] for w in p["words"]] == [vo.letters(w)[:4] for w in chunk_words(p["text"])], p
        place(track, report, li, role, parts, vo.remap(w1))
    assert wi == len(words), f"{len(words) - wi} words of the take were not used"
    track.tofile(f"{work}/track_narrator.f32")
    open(f"{work}/narrator_fx.txt", "w").write(TAKE_FX + "\n")
    json.dump(report, open(f"{work}/alignment_take.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
