"""Creative grade for the clip, baked into a 65^3 .cube LUT (Rec.709 in/out).

The look is modelled on how Sony CineAlta footage (F35 / F65, S-Log) was
printed for features in the early 2010s, done on a phone clip:
- tone: a print-style S-curve on perceptual lightness: shadows dense but not
  crushed (black sits at ~2.5 %), mids with a little more contrast, and a long
  soft shoulder so whites roll off to ~95 % instead of clipping hard;
- colour: chroma is rebuilt in OkLab (hue-linear, so no hue drift when
  saturating): denser mid-tone colour, highlights and deep shadows quieter,
  greens pulled down and slightly towards olive, sky blues towards teal;
- split tone: shadows lean teal, highlights slightly warm, skin protected
  (hues 20-80 deg keep their hue and get a touch of warmth);
- subtractive density: saturated colours are darkened a little, like dye.

usage: grade_lut.py OUT.cube [SIZE]
"""
import sys

import numpy as np
from scipy.interpolate import PchipInterpolator

# --- colour space helpers (linear sRGB/Rec.709 primaries <-> OkLab) ---------
M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
               [0.2119034982, 0.6806995451, 0.1073969566],
               [0.0883024619, 0.2817188376, 0.6299787005]])
M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
               [1.9779984951, -2.4285922050, 0.4505937099],
               [0.0259040371, 0.7827717662, -0.8086757660]])
M1i, M2i = np.linalg.inv(M1), np.linalg.inv(M2)
GAMMA = 2.4  # BT.1886 display


def to_oklab(lin):
    lms = np.cbrt(np.maximum(lin @ M1.T, 0))
    return lms @ M2.T


def from_oklab(lab):
    lms = (lab @ M2i.T) ** 3
    return lms @ M1i.T


# --- look parameters --------------------------------------------------------
# tone curve, defined on display code values for a neutral grey (in, out)
LOOK = dict(
    tone=[(0.000, 0.018), (0.040, 0.034), (0.120, 0.085), (0.250, 0.200),
          (0.420, 0.400), (0.600, 0.618), (0.760, 0.785), (0.880, 0.880),
          (0.950, 0.922), (1.000, 0.950)],
    sat_mid=0.20, sat_low=0.88, sat_high_cut=0.32,
    skin=(52, 32, 0.04, 3.0),       # centre deg, half-width, chroma gain, hue shift
    green=(140, 45, -0.30, -10.0),
    blue=(255, 45, -0.10, -16.0),
    teal=(215, 0.026), warm=(68, 0.006),
    density=0.12,
)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def hue_window(h, centre, width):
    """1 at `centre` deg, 0 beyond +-width, smooth (raised cosine)."""
    d = np.abs((h - centre + 180) % 360 - 180)
    return np.where(d < width, 0.5 + 0.5 * np.cos(np.pi * d / width), 0.0)


def grade(rgb, look=LOOK):
    """rgb: (..., 3) Rec.709 code values in [0, 1] -> graded code values."""
    tone_v = PchipInterpolator(*zip(*look["tone"]))
    lin = np.clip(rgb, 0, 1) ** GAMMA
    lab = to_oklab(lin)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.hypot(a, b)
    h = np.degrees(np.arctan2(b, a)) % 360

    # tone on lightness (neutral: L = Y^(1/3), code value V = Y^(1/2.4) = L^1.25)
    v = np.clip(L, 0, 1) ** (3 / GAMMA)
    L2 = np.clip(tone_v(v), 0, 1) ** (GAMMA / 3)

    # chroma: follow the lightness change half way, then shape by lightness;
    # the mid-tone boost is a vibrance: it fades out for colours that are
    # already strong (the orange parka), so only the muted ones gain density
    C2 = C * np.sqrt(np.maximum(L2, 1e-4) / np.maximum(L, 1e-4))
    mids = smoothstep(0.12, 0.40, L2)
    C2 *= (look["sat_low"] + (1 - look["sat_low"]) * mids
           - look["sat_high_cut"] * smoothstep(0.80, 0.98, L2))
    C2 *= 1 + look["sat_mid"] * mids * (1 - 0.8 * smoothstep(0.07, 0.16, C))

    # hue-specific gains and shifts
    h2 = h.copy()
    wins = {}
    for key in ("skin", "green", "blue"):
        centre, width, gain, shift = look[key]
        w = hue_window(h, centre, width)
        wins[key] = w
        C2 *= 1 + gain * w
        h2 += shift * w
    a2, b2 = C2 * np.cos(np.radians(h2)), C2 * np.sin(np.radians(h2))

    # split tone: teal shadows, slightly warm highlights; faces get less teal.
    # The tint is a share of the lightness (same relative saturation at every
    # level), otherwise a fixed OkLab offset turns near-black pure cyan.
    th, tamt = look["teal"]
    wh, wamt = look["warm"]
    sh = (1 - smoothstep(0.08, 0.60, L2)) * (1 - 0.7 * wins["skin"] * smoothstep(0.02, 0.06, C))
    sh = sh * L2 / 0.5
    hi = smoothstep(0.60, 0.95, L2)
    a2 = a2 + tamt * sh * np.cos(np.radians(th)) + wamt * hi * np.cos(np.radians(wh))
    b2 = b2 + tamt * sh * np.sin(np.radians(th)) + wamt * hi * np.sin(np.radians(wh))

    # subtractive density: strong colour a little darker
    L2 = L2 - look["density"] * np.clip(C2, 0, 0.25) * smoothstep(0.2, 0.6, L2)

    out = from_oklab(np.stack([L2, a2, b2], -1))
    # gamut: measure how far along the line from the neutral of the same L to
    # the RGB cube boundary each colour sits (r = 1 on the boundary) and
    # compress softly from r = 0.85, so chroma never slams into a clipped channel
    grey = from_oklab(np.stack([L2, 0 * L2, 0 * L2], -1))
    d = out - grey
    room = np.where(d > 0, (1 - grey) / np.maximum(d, 1e-9),
                    np.where(d < 0, grey / np.maximum(-d, 1e-9), np.inf))
    r = 1 / np.maximum(np.min(room, axis=-1), 1e-9)
    knee = 0.85
    r2 = np.where(r > knee, knee + (1 - knee) * np.tanh((r - knee) / (1 - knee)), r)
    out = grey + d * (r2 / np.maximum(r, 1e-9))[..., None]
    return np.clip(out, 0, 1) ** (1 / GAMMA)


def write_cube(path, size=65):
    g = np.linspace(0, 1, size)
    bb, gg, rr = np.meshgrid(g, g, g, indexing="ij")   # R fastest in .cube order
    rgb = np.stack([rr, gg, bb], -1).reshape(-1, 3)
    out = grade(rgb)
    with open(path, "w") as f:
        f.write('TITLE "CineAlta-2010s print look"\n')
        f.write(f"LUT_3D_SIZE {size}\nDOMAIN_MIN 0.0 0.0 0.0\nDOMAIN_MAX 1.0 1.0 1.0\n")
        np.savetxt(f, out, fmt="%.6f")


if __name__ == "__main__":
    write_cube(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 65)
