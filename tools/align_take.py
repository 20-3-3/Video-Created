"""Word timings for the recorded narrator take by CTC forced alignment (torchaudio MMS_FA).

usage: align_take.py TAKE.wav TRANSCRIPT.txt OUT.json
The transcript is what was actually said, in order; words are romanized for the aligner.
"""
import json
import re
import subprocess
import sys

import numpy as np
import torch
import torchaudio

TRANSLIT = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
                    ["a", "b", "v", "g", "d", "e", "yo", "zh", "z", "i", "y", "k", "l", "m", "n", "o", "p", "r", "s",
                     "t", "u", "f", "kh", "ts", "ch", "sh", "shch", "", "y", "", "e", "yu", "ya"]))


def romanize(word):
    return "".join(TRANSLIT.get(c, c if "a" <= c <= "z" else "") for c in word.lower())


take, transcript, out = sys.argv[1:4]
words = [w for w in open(transcript, encoding="utf-8").read().split() if romanize(w)]
SR = torchaudio.pipelines.MMS_FA.sample_rate
raw = subprocess.run(["ffmpeg", "-v", "error", "-i", take, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                     capture_output=True, check=True).stdout
wave = torch.from_numpy(np.frombuffer(raw, np.float32).copy())[None]

bundle = torchaudio.pipelines.MMS_FA
model = bundle.get_model(with_star=False).eval()
dictionary = bundle.get_dict(star=None)
tokens = [[dictionary[c] for c in romanize(w)] for w in words]
assert all(tokens), [w for w, t in zip(words, tokens) if not t]
with torch.inference_mode():
    emission, _ = model(wave)
flat = [t for ts in tokens for t in ts]
aligned, scores = torchaudio.functional.forced_align(emission, torch.tensor([flat], dtype=torch.int32), blank=0)
spans = torchaudio.functional.merge_tokens(aligned[0], scores[0].exp())
ratio = wave.shape[1] / emission.shape[1] / SR
res, k = [], 0
for w, ts in zip(words, tokens):
    sp = spans[k:k + len(ts)]
    k += len(ts)
    res.append(dict(text=w, start=round(sp[0].start * ratio, 3), end=round(sp[-1].end * ratio, 3),
                    score=round(float(np.mean([s.score for s in sp])), 3)))
json.dump(res, open(out, "w"), ensure_ascii=False, indent=0)
for r in res:
    print(f"{r['start']:6.2f}-{r['end']:6.2f} {r['score']:.2f} {r['text']}")
