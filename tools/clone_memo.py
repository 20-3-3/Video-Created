"""Read the passenger memo rules in the narrator's own voice (Chatterbox multilingual, MIT licence).

usage: clone_memo.py TAKE.wav REF_START REF_END OUT_DIR [TAKES]

The recorded take has no memo, so the four rules are synthesized zero-shot from a stretch of
the take (REF_START..REF_END seconds, clean speech). Each rule is generated TAKES times; the take
is saved as OUT_DIR/memo_<n>_take<t>.wav and listed in OUT_DIR/takes.json with its similarity to
the reference voice; narrator_take.py picks the closest take that fits the rule's slot.
"""
import json
import os
import subprocess
import sys

import librosa
import numpy as np
import soundfile as sf
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# stress marks (U+0301) as the model's Russian front end expects; without them it misplaces some
RULES = [
    "Пе\u0301рвое: не сади\u0301тесь в ваго\u0301н без све\u0301та.",
    "Второ\u0301е: е\u0301сли дежу\u0301рная спро\u0301сит, кака\u0301я сего\u0301дня ста\u0301нция, — не отвеча\u0301йте.",
    "Тре\u0301тье: не поднима\u0301йтесь по эскала\u0301тору, кото\u0301рый е\u0301дет вниз.",
    "Четвёртое: по\u0301езд прихо\u0301дит оди\u0301н раз.",
]


def main():
    take, ref_start, ref_end, out_dir = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    n_takes = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    os.makedirs(out_dir, exist_ok=True)
    ref = os.path.join(out_dir, "reference.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(ref_start), "-to", str(ref_end), "-i", take,
                    "-ac", "1", "-ar", "24000", ref], check=True)

    torch.manual_seed(7)
    model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")

    def embed(x, sr):
        e = model.ve.embeds_from_wavs([librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=16000)],
                                      sample_rate=16000)[0]
        return e / np.linalg.norm(e)

    ref_emb = embed(*librosa.load(ref, sr=None))
    takes = []
    for i, text in enumerate(RULES, 1):
        for t in range(1, n_takes + 1):
            wav = model.generate(text, language_id="ru", audio_prompt_path=ref,
                                 exaggeration=0.4, cfg_weight=0.5, temperature=0.7)
            x = wav.squeeze(0).numpy()
            sim = float(np.dot(embed(x, model.sr), ref_emb))
            name = f"memo_{i}_take{t}.wav"
            sf.write(os.path.join(out_dir, name), x, model.sr)
            takes.append(dict(rule=i, file=name, similarity=round(sim, 3)))
            print(f"rule {i} take {t}: {len(x) / model.sr:.2f}s similarity {sim:.3f}", flush=True)
    json.dump(takes, open(os.path.join(out_dir, "takes.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
