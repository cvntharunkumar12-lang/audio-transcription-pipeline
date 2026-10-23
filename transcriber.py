"""
Audio transcription pipeline.

Pipeline stages
  1. Ingest      - accept any audio/video file, validate it exists and is readable
  2. Normalise   - ffmpeg converts to 16 kHz mono PCM WAV (the format Whisper expects)
  3. Chunk       - files longer than LONG_AUDIO_SECONDS are split into fixed windows
  4. Transcribe  - faster-whisper (open source, CTranslate2 backend), VAD filtered
  5. Post-process- clean text, apply chunk offsets, build JSON / SRT / TXT, keywords

Usage
  python transcriber.py input.mp3 --model base --out-dir output
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
import wave
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SUPPORTED_HINT = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma", ".mp4", ".webm", ".mkv"}
TARGET_SAMPLE_RATE = 16000
LONG_AUDIO_SECONDS = 30 * 60   # above this we chunk
CHUNK_SECONDS = 10 * 60        # size of each chunk
STOPWORDS = set(
    "the a an and or but if of to in on at for with is are was were be been it this that these those "
    "i you he she we they me my your our their as by from not so do does did have has had will would can could "
    "there here what which who whom about into than then just also very".split()
)


class TranscriptionError(RuntimeError):
    pass


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str


# ---------------------------------------------------------------- ingest / normalise
def _require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise TranscriptionError(f"'{name}' not found on PATH. Install ffmpeg (includes ffprobe).")


def probe_duration(path: Path) -> float:
    """Return duration in seconds using ffprobe."""
    _require_tool("ffprobe")
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        raise TranscriptionError(f"Cannot read audio file (corrupt or unsupported?): {proc.stderr.strip()}")
    return float(proc.stdout.strip())


def normalise_audio(src: Path, dst: Path) -> Path:
    """Convert any input to 16 kHz mono 16-bit PCM WAV."""
    _require_tool("ffmpeg")
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE),
         "-c:a", "pcm_s16le", str(dst)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise TranscriptionError(f"ffmpeg conversion failed: {proc.stderr[-400:]}")
    return dst


def split_into_chunks(wav: Path, out_dir: Path, chunk_seconds: int) -> list[Path]:
    """Split a WAV into fixed-length chunks (stream copy, no re-encode)."""
    pattern = out_dir / "chunk_%04d.wav"
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav), "-f", "segment", "-segment_time", str(chunk_seconds),
         "-c", "copy", str(pattern)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise TranscriptionError(f"ffmpeg split failed: {proc.stderr[-400:]}")
    return sorted(out_dir.glob("chunk_*.wav"))


# ---------------------------------------------------------------- transcription
_MODEL_CACHE: dict = {}


def load_model(model_size: str = "base", device: str = "auto", compute_type: str = "int8"):
    """Load (and cache) the model once per process - loading is the expensive part."""
    key = (model_size, device, compute_type)
    if key not in _MODEL_CACHE:
        from faster_whisper import WhisperModel
        _MODEL_CACHE[key] = WhisperModel(model_size, device=device, compute_type=compute_type)
    return _MODEL_CACHE[key]


def load_wav_samples(wav: Path) -> np.ndarray:
    """Read our normalised 16 kHz mono PCM WAV into a float32 array in [-1, 1].

    Passing samples directly means faster-whisper never has to decode the file
    itself, so the pipeline does not depend on the PyAV ('av') package version.
    """
    with wave.open(str(wav), "rb") as wf:
        if wf.getframerate() != TARGET_SAMPLE_RATE or wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise TranscriptionError("Normalised audio is not 16 kHz mono 16-bit PCM")
        frames = wf.readframes(wf.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def _transcribe_file(model, wav: Path, offset: float, start_id: int, language: str | None):
    segments_iter, info = model.transcribe(
        load_wav_samples(wav), language=language, vad_filter=True, beam_size=5,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    out: list[Segment] = []
    for i, seg in enumerate(segments_iter, start=start_id):  # generator -> low memory
        text = seg.text.strip()
        if text:
            out.append(Segment(i, round(seg.start + offset, 2), round(seg.end + offset, 2), text))
    return out, info


# ---------------------------------------------------------------- post-processing
def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text


def to_timestamp(seconds: float, srt: bool = False) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    sep = "," if srt else "."
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(segments: list[Segment]) -> str:
    blocks = [f"{s.id}\n{to_timestamp(s.start, True)} --> {to_timestamp(s.end, True)}\n{s.text}\n" for s in segments]
    return "\n".join(blocks)


def top_keywords(text: str, n: int = 10) -> list[str]:
    words = re.findall(r"[a-zA-Z']{3,}", text.lower())
    counts = Counter(w for w in words if w not in STOPWORDS)
    return [w for w, _ in counts.most_common(n)]


def build_result(segments: list[Segment], info, duration: float, source: str) -> dict:
    for s in segments:
        s.text = clean_text(s.text)
    full_text = " ".join(s.text for s in segments)
    return {
        "source_file": source,
        "language": getattr(info, "language", None),
        "language_probability": round(getattr(info, "language_probability", 0.0) or 0.0, 3),
        "duration_seconds": round(duration, 2),
        "word_count": len(full_text.split()),
        "keywords": top_keywords(full_text),
        "full_text": full_text,
        "segments": [asdict(s) for s in segments],
    }


# ---------------------------------------------------------------- public entry point
def transcribe(path: str | Path, model_size: str = "base", language: str | None = None,
               device: str = "auto", compute_type: str = "int8") -> dict:
    src = Path(path)
    if not src.is_file():
        raise TranscriptionError(f"File not found: {src}")
    if src.suffix.lower() not in SUPPORTED_HINT:
        # not fatal: ffmpeg can read many more formats, ffprobe below is the real check
        pass
    duration = probe_duration(src)
    model = load_model(model_size, device, compute_type)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        wav = normalise_audio(src, tmp_dir / "normalised.wav")

        segments: list[Segment] = []
        info = None
        if duration > LONG_AUDIO_SECONDS:
            chunk_dir = tmp_dir / "chunks"
            chunk_dir.mkdir()
            for idx, chunk in enumerate(split_into_chunks(wav, chunk_dir, CHUNK_SECONDS)):
                part, info = _transcribe_file(model, chunk, idx * CHUNK_SECONDS, len(segments) + 1, language)
                segments.extend(part)
        else:
            segments, info = _transcribe_file(model, wav, 0.0, 1, language)

    return build_result(segments, info, duration, src.name)


def save_outputs(result: dict, out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(result["source_file"]).stem
    segs = [Segment(**s) for s in result["segments"]]
    paths = {
        "json": out / f"{stem}.json",
        "srt": out / f"{stem}.srt",
        "txt": out / f"{stem}.txt",
    }
    paths["json"].write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["srt"].write_text(to_srt(segs), encoding="utf-8")
    paths["txt"].write_text(
        "\n".join(f"[{to_timestamp(s.start)} -> {to_timestamp(s.end)}] {s.text}" for s in segs), encoding="utf-8"
    )
    return {k: str(v) for k, v in paths.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe an audio file with per-segment timestamps.")
    ap.add_argument("audio")
    ap.add_argument("--model", default="base", help="tiny | base | small | medium | large-v3")
    ap.add_argument("--language", default=None, help="e.g. en, hi, te (auto-detect if omitted)")
    ap.add_argument("--out-dir", default="output")
    args = ap.parse_args()
    try:
        result = transcribe(args.audio, args.model, args.language)
    except TranscriptionError as e:
        raise SystemExit(f"Error: {e}")
    print(json.dumps({k: v for k, v in save_outputs(result, args.out_dir).items()}, indent=2))
    for s in result["segments"]:
        print(f"[{to_timestamp(s['start'])} -> {to_timestamp(s['end'])}] {s['text']}")


if __name__ == "__main__":
    main()
