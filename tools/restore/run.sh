#!/bin/sh
# Restore a handheld phone clip: constant 24 fps, stabilization,
# CineAlta-2010s style grade and finishing, cleaned and levelled audio.
#
# usage: tools/restore/run.sh SOURCE.mp4 OUT.mp4 [WORKDIR]
# needs: ffmpeg (with libx264), python3 with numpy, scipy, opencv-python-headless, soundfile
#
# The per-clip choices live in the scripts and were picked for the selfie
# clip they were made for: which synthesizer fills each dropped frame
# (fill_frames.py CHOICE, which also checks that the gaps match) and the
# stabilizer weights (stabilize.py).
set -eu
src=$1
out=$2
work=${3:-$(mktemp -d)}
here=$(dirname "$0")
mkdir -p "$work"

python3 "$here/fill_frames.py" "$src" "$work/filled.mkv"            # 410 frames + 10 synthesized -> 420 @ 24 fps
python3 "$here/stabilize.py" "$work/filled.mkv" "$work/stab.mkv"    # L1 camera path, 1080x1920, lossless
python3 "$here/grade_lut.py" "$work/look.cube" 65                   # the look as a 65^3 LUT

frames=$(ffprobe -v error -count_packets -select_streams v:0 -show_entries stream=nb_read_packets -of csv=p=0 "$work/filled.mkv")
ffmpeg -v error -y -i "$src" -vn -c:a pcm_f32le "$work/audio_src.wav"
python3 "$here/audio.py" "$work/audio_src.wav" "$work/audio.wav" "$(python3 -c "print($frames / 24)")"

python3 "$here/finish.py" "$work/stab.mkv" "$work/look.cube" "$out" "$work/audio.wav"
