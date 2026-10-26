"""Denoise the recorded narrator take with DeepFilterNet3.

usage: denoise_take.py MODEL_DIR WEIGHTS.safetensors TAKE OUT.wav [ATTEN_LIM_DB]

MODEL_DIR holds the DeepFilterNet3 config.ini, WEIGHTS its weights as safetensors (loaded without
pickle). The phone recording has a steady low hum about 15-20 dB under the voice; attenuation is
limited to ATTEN_LIM_DB (default 30) so the voice keeps its natural floor and no gating artifacts.
"""
import sys

from df.enhance import enhance, init_df, load_audio, save_audio
from safetensors.torch import load_file

model_dir, weights, src, dst = sys.argv[1:5]
lim = float(sys.argv[5]) if len(sys.argv) > 5 else 30.0
model, state, _ = init_df(model_base_dir=model_dir, epoch="none", log_level="ERROR")
model.load_state_dict(load_file(weights))
model.eval()
audio, _ = load_audio(src, sr=state.sr())
save_audio(dst, enhance(model, state, audio, atten_lim_db=lim), state.sr())
