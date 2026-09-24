# subdub

Turn a video's subtitles into a spoken dub with ElevenLabs, mixed over the original audio, and play it in VLC.

Each subtitle line is voiced and placed at its timestamp. The original soundtrack keeps playing underneath and dips while the dub speaks. The output is a new `.mkv` with:

- **Audio 1:** dub + original (default)
- **Audio 2:** untouched original (switch in VLC via *Audio → Audio Track*)
- the subtitles

## Requirements

- Python 3.9+ (`dub.py` uses only the standard library)
- `ffmpeg` / `ffprobe` on your PATH (`brew install ffmpeg`)
- An [ElevenLabs](https://elevenlabs.io) API key with text-to-speech permission

## Setup

```sh
cp .env.example .env   # then put your key in ELEVENLABS_API_KEY
```

## Usage

```sh
# see which audio/subtitle tracks a file has
python3 dub.py movie.mkv --list-tracks

# cheap test: only the first 3 minutes, then open in VLC
python3 dub.py movie.mkv --subs movie.srt --preview 180 --open

# full run (asks before spending credits unless -y)
python3 dub.py movie.mkv --subs movie.srt
```

Without `--subs`, it uses `movie.srt` next to the video, or the embedded subtitle track (`--sub-track N`).

It prints the character count and estimated credits before generating. Generated lines are cached in `.dubcache/`, so re-runs and remixes only pay for lines that changed.

### Timing

- Lines that run longer than their subtitle are sped up (up to `--max-speed 1.5`).
- Lines that would finish too early are read slower by ElevenLabs (down to `--min-voice-speed 0.7`), then stretched a little more locally (`--min-stretch 0.85`), aiming to fill `--fill 0.85` of the subtitle's time.

### Mix

| Option | Default | Effect |
|---|---|---|
| `--dub-volume` | 1.4 | dub loudness |
| `--original-volume` | 0.8 | original soundtrack loudness |
| `--duck` | 4 | how much the original dips while the dub speaks (1 = off) |
| `--voice` | George | ElevenLabs voice ID (or `ELEVENLABS_VOICE_ID` in `.env`) |
| `--model` | `eleven_flash_v2_5` | cheaper/faster; `eleven_multilingual_v2` for quality |

To test the pipeline without spending credits, `--engine say` uses macOS `say` instead of ElevenLabs.

## Keep the music, drop only the original voices (optional)

`separate.py` splits the soundtrack into dialogue and music/effects with [Demucs](https://github.com/facebookresearch/demucs), in chunks so a full movie fits in 8 GB of RAM.

```sh
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements-separate.txt
.venv/bin/python separate.py movie.mkv --device cpu   # -> movie.no_vocals.flac, movie.vocals.flac

python3 dub.py movie.mkv --subs movie.srt \
  --background movie.no_vocals.flac --vocals movie.vocals.flac --voice-volume 0.3
```

Music and effects stay at full volume. The original voices are mixed in at `--voice-volume` (0 removes them) and dip further under the dub.

On CPU it runs at about 2× real time. The Apple GPU (`--device mps`) failed on PyTorch 2.5.1; newer versions are untested.

## Limitations

- One voice for every character. Subtitles rarely say who is speaking.
- Subtitle credit lines ("Translated by…") are skipped, but other non-dialogue text in styled `.ass` files may get voiced. Prefer a dialogue-only track.
