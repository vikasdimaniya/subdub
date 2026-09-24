#!/usr/bin/env python3
"""Subtitle -> ElevenLabs voice dub, mixed on top of the original audio.

Usage:
    python3 dub.py movie.mkv                      # uses movie.srt next to it, or the embedded subtitle track
    python3 dub.py movie.mp4 --subs other.srt --preview 120 --open

Output: movie.dubbed.mkv with
    audio track 1 = original (ducked) + generated voice   (default)
    audio track 2 = untouched original
Only needs python3 (stdlib) and ffmpeg/ffprobe on PATH.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

SR = 24000  # generated clips: mono s16le @ 24kHz
HERE = Path(__file__).resolve().parent
VLC_BIN = "/Applications/VLC.app/Contents/MacOS/VLC"


# ---------------------------------------------------------------- helpers

def run(cmd, data=None):
    p = subprocess.run(cmd, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError("%s failed:\n%s" % (cmd[0], p.stderr.decode(errors="replace")[-2000:]))
    return p.stdout


def load_env():
    env_file = HERE / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def media_duration(path):
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)])
    return float(json.loads(out)["format"]["duration"])


# ---------------------------------------------------------------- subtitles

def load_subtitles(video, subs, track=0):
    """Return (is_external, SRT text). Any format ffmpeg reads (srt/vtt/ass/embedded) is normalised to SRT."""
    if subs is None:
        for ext in (".srt", ".vtt", ".ass", ".ssa"):
            cand = video.with_suffix(ext)
            if cand.exists():
                subs = cand
                break
    src = subs if subs is not None else video  # fall back to the first embedded subtitle track
    try:
        return subs is not None, run(["ffmpeg", "-v", "error", "-i", str(src), "-map", "0:s:%d" % track, "-f", "srt", "-"]).decode("utf-8", "replace")
    except RuntimeError as e:
        sys.exit("Could not read subtitles from %s (pass --subs file.srt).\n%s" % (src, e))


TS = r"(\d+):(\d+):(\d+)[,.](\d+)"


def to_sec(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")[:3]) / 1000


def clean(text):
    text = re.sub(r"<[^>]+>|\{[^}]*\}", "", text)          # html / ass tags
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)", "", text)       # [MUSIC] (laughs) sound descriptions
    text = re.sub(r"[♪♫#]", "", text)
    lines = []
    for ln in text.splitlines():
        ln = re.sub(r"^\s*-\s*", "", ln)                  # dialogue dashes
        ln = re.sub(r"^[A-Z][A-Z .'\-]{1,20}:\s*", "", ln)  # SPEAKER: labels
        if ln.strip():
            lines.append(ln.strip())
    return " ".join(lines).strip()


# translator / release-group credit lines that shouldn't be voiced
CREDITS = re.compile(r"(?i)\b(translated|subtitled|subbed|synced|timed|encoded|ripped) by\b|https?://|www\.|subtitle delay")


def parse_srt(srt):
    cues = []
    for block in re.split(r"\n\s*\n", srt.replace("\r", "")):
        m = re.search(TS + r"\s*-->\s*" + TS, block)
        if not m:
            continue
        text = clean(block[m.end():])
        if re.search(r"\w", text) and not CREDITS.search(text):
            cues.append({"start": to_sec(*m.groups()[:4]), "end": to_sec(*m.groups()[4:]), "text": text})
    cues.sort(key=lambda c: c["start"])
    return cues


# ---------------------------------------------------------------- TTS

CHARS_PER_SEC = 17.0  # measured speaking rate of ElevenLabs voices at speed 1.0


def voice_speed(cue, args):
    """Speaking speed so the line roughly fills its subtitle time (slow dramatic lines get a slower read)."""
    estimate = len(cue["text"]) / CHARS_PER_SEC
    target = max(args.fill * (cue["end"] - cue["start"]), 0.1)
    speed = max(args.min_voice_speed, min(1.0, estimate / target))
    return round(speed * 20) / 20  # 0.05 steps keep the cache stable


def elevenlabs_request(path, body=None):
    req = urllib.request.Request("https://api.elevenlabs.io" + path, method="POST" if body else "GET",
                                 data=json.dumps(body).encode() if body else None,
                                 headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"],
                                          "Content-Type": "application/json"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")
            if e.code in (429, 500, 502, 503) and attempt < 5:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError("ElevenLabs %s: %s" % (e.code, msg))
        except urllib.error.URLError:
            if attempt < 5:
                time.sleep(2 ** attempt)
                continue
            raise


def tts_elevenlabs(text, speed, args):
    body = {"text": text, "model_id": args.model}
    if args.lang:
        body["language_code"] = args.lang
    if speed != 1.0:
        body["voice_settings"] = dict(args.voice_settings, speed=speed)
    return elevenlabs_request("/v1/text-to-speech/%s?output_format=pcm_%d" % (args.voice, SR), body)


def tts_say(text, speed, args):
    """Free offline engine (macOS `say`) for testing the pipeline without spending credits."""
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "x.wav")
        run(["say", "-r", str(int(185 * speed)), "-o", out, "--data-format=LEI16@%d" % SR, text])
        with wave.open(out) as w:
            return w.readframes(w.getnframes())


def synth(cue, args, cache):
    speed = voice_speed(cue, args)
    key = "%s|%s|%s|%s|%s" % (args.engine, args.voice, args.model, args.lang, cue["text"])
    if speed != 1.0:
        key += "|speed=%.2f" % speed
    path = cache / (hashlib.sha1(key.encode()).hexdigest() + ".pcm")
    if not path.exists():
        pcm = (tts_say if args.engine == "say" else tts_elevenlabs)(cue["text"], speed, args)
        path.write_bytes(pcm)
    return path.read_bytes()


def retime(pcm, factor):
    return run(["ffmpeg", "-v", "error", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0",
                "-af", "atempo=%.3f" % factor, "-f", "s16le", "-ar", str(SR), "-ac", "1", "pipe:1"], pcm)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Dub a video from its subtitles with ElevenLabs.")
    ap.add_argument("video", type=Path)
    ap.add_argument("--subs", type=Path, help="subtitle file (default: <video>.srt or embedded track)")
    ap.add_argument("--sub-track", type=int, default=0, help="which subtitle track to read (0-based), see --list-tracks")
    ap.add_argument("--list-tracks", action="store_true", help="show the audio/subtitle tracks in the video and exit")
    ap.add_argument("--out", type=Path, help="output file (default: <video>.dubbed.mkv)")
    ap.add_argument("--voice", default=os.environ.get("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb"))
    ap.add_argument("--model", default="eleven_flash_v2_5", help="eleven_flash_v2_5 (fast/cheap) or eleven_multilingual_v2")
    ap.add_argument("--lang", help="ISO 639-1 language code of the subtitles, e.g. hi, en, es (optional)")
    ap.add_argument("--engine", choices=["elevenlabs", "say"], default="elevenlabs")
    ap.add_argument("--audio-track", type=int, default=0, help="which original audio track to mix under (0-based)")
    ap.add_argument("--original-volume", type=float, default=0.8)
    ap.add_argument("--background", type=Path, help="separated music/effects track (no dialogue), e.g. Demucs 'no_vocals'")
    ap.add_argument("--vocals", type=Path, help="separated Japanese dialogue track, mixed in quietly (with --background)")
    ap.add_argument("--voice-volume", type=float, default=0.3, help="level of the --vocals track (0 = remove Japanese voice)")
    ap.add_argument("--dub-volume", type=float, default=1.4)
    ap.add_argument("--duck", type=float, default=4, help="how hard to duck the original while the dub speaks (1 = off)")
    ap.add_argument("--max-speed", type=float, default=1.5, help="max speed-up for lines that overrun their slot")
    ap.add_argument("--fill", type=float, default=0.85, help="aim for each line to fill this fraction of its subtitle time")
    ap.add_argument("--min-voice-speed", type=float, default=0.7, help="slowest ElevenLabs speaking speed (0.7-1.0, 1 = never slow down)")
    ap.add_argument("--min-stretch", type=float, default=0.85, help="extra local slow-down after generation (1 = off)")
    ap.add_argument("--workers", type=int, default=3, help="parallel TTS requests (match your ElevenLabs plan)")
    ap.add_argument("--preview", type=float, help="only process the first N seconds (cheap test run)")
    ap.add_argument("--yes", "-y", action="store_true", help="don't ask before spending credits")
    ap.add_argument("--open", action="store_true", help="open the result in VLC")
    args = ap.parse_args()

    load_env()
    if args.list_tracks:
        print(run(["ffprobe", "-v", "error", "-show_entries",
                   "stream=index,codec_type,codec_name:stream_tags=language,title",
                   "-of", "compact=p=0:nk=0", str(args.video)]).decode())
        return
    if args.engine == "elevenlabs" and not os.environ.get("ELEVENLABS_API_KEY"):
        sys.exit("Set ELEVENLABS_API_KEY (env var or .env file next to dub.py).")
    video = args.video.resolve()
    out = args.out or video.with_name(video.stem + ".dubbed.mkv")
    cache = HERE / ".dubcache"
    cache.mkdir(exist_ok=True)
    args.voice_settings = {}
    if args.engine == "elevenlabs":  # keep the voice's own stability/similarity when we override speed
        try:
            args.voice_settings = json.loads(elevenlabs_request("/v1/voices/%s/settings" % args.voice))
            args.voice_settings.pop("speed", None)
        except RuntimeError:
            pass  # key without voices_read permission: send speed only

    duration = media_duration(video)
    if args.preview:
        duration = min(duration, args.preview)
    external_subs, srt = load_subtitles(video, args.subs, args.sub_track)
    cues = [c for c in parse_srt(srt) if c["start"] < duration]
    if not cues:
        sys.exit("No subtitle lines found.")

    chars = sum(len(c["text"]) for c in cues)
    per_char = 0.5 if "flash" in args.model or "turbo" in args.model else 1.0
    print("%d lines, %d characters to voice with %s: ~%d credits (lines already cached are free)."
          % (len(cues), chars, args.model, chars * per_char))
    for c in cues[:3]:
        print("  %7.1fs  %s" % (c["start"], c["text"]))
    if args.engine == "elevenlabs" and not args.yes and input("Continue? [y/N] ").strip().lower() != "y":
        return

    # 1. generate all clips in parallel (cached on disk, so re-runs are free)
    clips = [None] * len(cues)
    done = 0
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(synth, c, args, cache): i for i, c in enumerate(cues)}
        for f in concurrent.futures.as_completed(futs):
            clips[futs[f]] = f.result()
            done += 1
            print("\rGenerating voice: %d/%d" % (done, len(cues)), end="", flush=True)
    print()

    # 2. lay clips onto one timeline: speed up lines that overrun their slot, stretch ones that end too early
    track = bytearray(int(duration * SR) * 2)
    cursor = 0.0
    squeezed = stretched = 0
    for i, (cue, pcm) in enumerate(zip(cues, clips)):
        start = max(cue["start"], cursor)
        next_start = cues[i + 1]["start"] if i + 1 < len(cues) else duration
        slot = max(min(next_start, cue["end"] + 1.0) - start, 0.3)
        length = len(pcm) / 2 / SR
        target = min(args.fill * (cue["end"] - cue["start"]), slot)
        if length > slot:
            pcm = retime(pcm, min(length / slot, args.max_speed))
            squeezed += 1
        elif length < target * 0.95 and args.min_stretch < 1:
            pcm = retime(pcm, max(length / target, args.min_stretch))
            stretched += 1
        a = int(start * SR) * 2
        pcm = pcm[: max(len(track) - a, 0)]
        track[a:a + len(pcm)] = pcm
        cursor = start + len(pcm) / 2 / SR
    print("%d lines sped up, %d slowed down to fit their timing." % (squeezed, stretched))

    # 3. mix: duck original under the dub, keep untouched original as a second track
    with tempfile.TemporaryDirectory() as d:
        dub_wav = os.path.join(d, "dub.wav")
        with wave.open(dub_wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(track)

        subs_srt = os.path.join(d, "subs.srt")
        Path(subs_srt).write_text(srt, encoding="utf-8")

        fmt = "aresample=48000,aformat=sample_rates=48000:channel_layouts=stereo"
        duck = "sidechaincompress=threshold=0.05:ratio=%s:attack=20:release=400" % args.duck
        extra_inputs = []
        if args.background:
            # separated stems: music/effects stay at full volume, only the Japanese voice is lowered
            extra_inputs += ["-i", str(args.background)]
            graph = "[3:a]{fmt},volume={ov}[bg];[1:a]{fmt},volume={dv}".format(
                fmt=fmt, ov=args.original_volume, dv=args.dub_volume)
            if args.vocals:
                extra_inputs += ["-i", str(args.vocals)]
                graph += (",asplit=2[sc][dub];[4:a]{fmt},volume={vv}[voc];[voc][sc]{duck}[vd];"
                          "[bg][vd][dub]amix=inputs=3:duration=first:normalize=0[mix]").format(
                    fmt=fmt, vv=args.original_volume * args.voice_volume, duck=duck)
            else:
                graph += "[dub];[bg][dub]amix=inputs=2:duration=first:normalize=0[mix]"
        else:
            graph = (
                "[0:a:{t}]{fmt},volume={ov}[orig];"
                "[1:a]{fmt},volume={dv},asplit=2[sc][dub];"
                "[orig][sc]{duck}[ducked];"
                "[ducked][dub]amix=inputs=2:duration=first:normalize=0[mix]"
            ).format(t=args.audio_track, fmt=fmt, ov=args.original_volume, dv=args.dub_volume, duck=duck)

        def mux(with_subs):
            cmd = ["ffmpeg", "-v", "error", "-stats", "-y"]
            if args.preview:
                cmd += ["-t", str(duration)]
            cmd += ["-i", str(video), "-i", dub_wav, "-i", subs_srt] + extra_inputs + ["-filter_complex", graph,
                    "-map", "0:v?", "-map", "[mix]", "-map", "0:a"]
            if external_subs:
                cmd += ["-map", "2:s"]
            if with_subs:
                cmd += ["-map", "0:s?"]
            cmd += ["-c", "copy", "-c:a:0", "aac", "-b:a:0", "192k",
                    "-metadata:s:a:0", "title=Dub (ElevenLabs)", "-metadata:s:a:1", "title=Original",
                    "-disposition:a", "0", "-disposition:a:0", "default"]
            if args.preview:
                cmd += ["-t", str(duration)]
            cmd.append(str(out))
            print("Mixing -> %s" % out)
            return subprocess.run(cmd).returncode == 0

        if not mux(True) and not mux(False):  # some subtitle codecs (e.g. mp4 mov_text) can't be copied
            sys.exit("ffmpeg mixing failed.")

    print("Done: %s" % out)
    if args.open:
        subprocess.Popen([VLC_BIN if os.path.exists(VLC_BIN) else "vlc", str(out)])


if __name__ == "__main__":
    main()
