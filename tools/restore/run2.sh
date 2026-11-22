#!/bin/sh
# v2: 2K (1440x2560), 60 fps, level horizon, synthesized edges.
#
# usage: tools/restore/run2.sh SOURCE.mp4 OUT.mp4 [WORKDIR]
# needs: ffmpeg (libx264), python3 with numpy, scipy, opencv-python-headless,
#        soundfile, torch (CPU is enough; bf16 uses AMX where present), spandrel
#
# Models (downloaded into WORKDIR/models and checked):
#   2x LiveActionV1 SPAN, jcj83429, OpenModelDB, CC BY-NC-SA 4.0
#   RIFE 4.25, hzwer/Practical-RIFE, MIT (Hugging Face mirror of the official zip)
set -eu
src=$1
out=$2
work=${3:-$(mktemp -d)}
here=$(dirname "$0")
mkdir -p "$work/models"

fetch() {  # url file sha256
    [ -f "$2" ] || curl -sSL -o "$2" "$1"
    echo "$3  $2" | sha256sum -c --quiet
}
span="$work/models/2xLiveActionV1_SPAN_490000.pth"
fetch "https://raw.githubusercontent.com/jcj83429/upscaling/f73a3a02874360ec6ced18f8bdd8e43b5d7bba57/2xLiveActionV1_SPAN/2xLiveActionV1_SPAN_490000.pth" \
    "$span" 8b166c75831ea7f694d9058ee9c8df8148af8cc1d2b57e69e6581b15cab572f7
fetch "https://huggingface.co/Bash2X/RIFE-Models/resolve/main/RIFE_v4.25.zip" \
    "$work/models/RIFE_v4.25.zip" e63d481b7ae5d4a4e6ad7ac5b410ff78f3bf7be3b51b2e38ca8152747abde5b4
unzip -q -o -j "$work/models/RIFE_v4.25.zip" train_log/flownet.pkl -d "$work/models"
echo "6615790efd627772917205db291f51cd392528a157ecbb2ecaeec3bff8eb6de2  $work/models/flownet.pkl" | sha256sum -c --quiet

python3 "$here/fill_frames.py" "$src" "$work/filled.mkv"                       # 420 frames @ 24 fps
python3 "$here/upscale.py" "$work/filled.mkv" "$span" "$work/sr.mkv"           # 1440x2560
python3 "$here/stabilize2.py" "$work/filled.mkv" "$work/sr.mkv" "$work/stab.mkv"
python3 "$here/interpolate.py" "$work/stab.mkv" "$work/models/flownet.pkl" "$work/stab60.mkv"
python3 "$here/grade_lut.py" "$work/look.cube" 65

frames=$(ffprobe -v error -count_packets -select_streams v:0 -show_entries stream=nb_read_packets -of csv=p=0 "$work/filled.mkv")
ffmpeg -v error -y -i "$src" -vn -c:a pcm_f32le "$work/audio_src.wav"
python3 "$here/audio.py" "$work/audio_src.wav" "$work/audio.wav" "$(python3 -c "print($frames / 24)")"

python3 "$here/finish.py" "$work/stab60.mkv" "$work/look.cube" "$out" "$work/audio.wav" \
    --crf 20 --sharpen 0.2 --grain 0.006
