#!/usr/bin/env python3
"""Split a soundtrack into dialogue and music/effects with Demucs, in chunks so a 2.5h movie fits in 8 GB RAM.

Usage (with the project venv):
    .venv/bin/python separate.py movie.mkv            # -> movie.no_vocals.flac + movie.vocals.flac
    .venv/bin/python separate.py movie.mkv --model htdemucs_ft   # slower, a bit cleaner

Then:
    python3 dub.py movie.mkv --background movie.no_vocals.flac --vocals movie.vocals.flac
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # a few ops aren't on Apple GPU yet

import numpy as np
import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model

SR = 44100


def read_chunk(path, start, length):
    out = subprocess.run(["ffmpeg", "-v", "error", "-ss", "%.3f" % start, "-t", "%.3f" % length, "-i", str(path),
                          "-map", "0:a:0", "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"],
                         stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).reshape(-1, 2).T.copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--model", default="htdemucs")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--chunk", type=float, default=600, help="seconds per chunk")
    ap.add_argument("--duration", type=float, help="only process the first N seconds (test)")
    args = ap.parse_args()

    total = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                  str(args.input)], stdout=subprocess.PIPE, check=True).stdout)
    if args.duration:
        total = min(total, args.duration)
    total_samples = int(round(total * SR))

    model = get_model(args.model)
    model.eval()
    voc_idx = model.sources.index("vocals")
    base = args.input.with_suffix("")
    out_bg = sf.SoundFile(str(base) + ".no_vocals.flac", "w", SR, 2, format="FLAC")
    out_voc = sf.SoundFile(str(base) + ".vocals.flac", "w", SR, 2, format="FLAC")

    pad = 5.0  # overlap each chunk by 5s on both sides, then trim, so chunk edges don't glitch
    written = 0
    t0 = time.time()
    pos = 0.0
    while written < total_samples:
        a = max(pos - pad, 0.0)
        wav = read_chunk(args.input, a, min(args.chunk, total - pos) + (pos - a) + pad)
        mix = torch.from_numpy(wav)
        ref = mix.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        with torch.no_grad():
            srcs = apply_model(model, ((mix - mean) / std)[None], device=args.device,
                               split=True, overlap=0.25, progress=False)[0]
        srcs = srcs * std + mean
        vocals = srcs[voc_idx]
        background = srcs.sum(0) - vocals

        lo = int(round((pos - a) * SR))
        n = min(int(round(args.chunk * SR)), total_samples - written, background.shape[1] - lo)
        out_bg.write(background[:, lo:lo + n].T.numpy())
        out_voc.write(vocals[:, lo:lo + n].T.numpy())
        written += n
        pos += args.chunk

        el = time.time() - t0
        done = written / total_samples
        print("separated %5.1f / %.1f min  (%.0f%%, ~%.0f min left)"
              % (written / SR / 60, total / 60, done * 100, el / done * (1 - done) / 60), flush=True)

    out_bg.close()
    out_voc.close()
    print("Done: %s.no_vocals.flac and %s.vocals.flac" % (base, base))


if __name__ == "__main__":
    main()
