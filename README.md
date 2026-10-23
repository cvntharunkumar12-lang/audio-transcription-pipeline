# Software Engineer Assessment – Transcription Pipeline
**Company:** Volga Partners  |  **Task:** Audio → Text pipeline with downstream processing

---

## 1. Problem Statement

Build a simple transcription pipeline that converts audio input into text and processes the result for downstream use.

**Allowed:** any open-source speech-to-text library or API, any programming language, mock data where needed.
**Focus:** engineering decisions, not training a model from scratch. The evaluators want to see how you think about and solve the problem.

### Part 1 – Questions on the form
| # | Question |
|---|----------|
| 1 | Implement a service or script that accepts an audio file (e.g., WAV or MP3) |
| 2 | Implement a service or script that transcribes spoken language into text |
| 3 | Implement a service or script that returns the transcription with timestamps per segment |
| 4 | How do you handle different audio formats? |
| 5 | How do you deal with long audio files? |

---

## 2. Solution Overview

Language: **Python 3.10+**. STT engine: **faster-whisper** (open-source Whisper reimplementation on CTranslate2: 4x faster than the original and low memory, runs on CPU with int8). Audio tooling: **ffmpeg / ffprobe**.

```
 audio file ──► validate ──► normalise ──► (chunk if long) ──► transcribe ──► post-process ──► JSON / SRT / TXT
  (any fmt)     ffprobe     ffmpeg 16k     10-min windows     faster-whisper   clean, offsets,
                            mono WAV                          + VAD            keywords, stats
```

### Files
| File | Purpose |
|------|---------|
| `transcriber.py` | Core library + CLI (all pipeline logic) |
| `api.py` | Optional FastAPI service exposing `POST /transcribe` |
| `requirements.txt` | Python dependencies |
| `run_tests.bat` | Windows script: converts the sample to WAV/MP3/OGG, transcribes each, then tests a corrupt file |
| `test_long.py` | Long-file test: loops the sample, forces chunking, checks timestamps stay continuous |
| `sample.m4a` | Recorded 45 s test clip (mock data) |
| `output/` | Example JSON / SRT / TXT produced by the pipeline |

---

## 3. Answers to the Form Questions

### Q1. Accepting an audio file
- **CLI:** `python transcriber.py input.mp3` takes a path.
- **API:** `POST /transcribe` takes a multipart upload; the file is streamed to a temp file (not held in RAM), size-capped at 500 MB, and deleted after processing.
- **Validation:** file existence check, then `ffprobe` confirms the file is actually decodable audio. Corrupt or non-audio files return a clear error (`TranscriptionError` / HTTP 400) instead of crashing. The extension is not trusted; content is.

### Q2. Transcribing speech to text
- `faster-whisper` loaded **once** and cached (`load_model`), since model loading dominates latency; the service warms it up at startup.
- `int8` quantisation by default so it runs on a laptop CPU; switch to `device="cuda", compute_type="float16"` on GPU.
- Language is auto-detected (confidence is reported) or can be forced with `--language`, which improves accuracy and speed.
- **VAD filter** (voice activity detection) skips silence, which cuts compute and reduces Whisper's known hallucinations on silent audio.
- The normalised WAV is read with Python's built-in `wave` module and passed to the model as a float32 NumPy array. This means faster-whisper never decodes the file itself, so the pipeline does **not** depend on the PyAV (`av`) package. This was a real fix: with Python 3.13, the newest `av` (19.x) crashed faster-whisper 1.2.1 (`open() got an unexpected keyword argument 'metadata_errors'`), and older `av` versions have no Python 3.13 wheel.
- Model size is a configurable accuracy/speed trade-off: `tiny` → `base` → `small` → `medium` → `large-v3`.

### Q3. Timestamps per segment
Each segment returned by the model includes `start` and `end` (seconds). Output shape:

```json
{
  "source_file": "sample.mp3",
  "language": "en",
  "language_probability": 0.99,
  "duration_seconds": 42.5,
  "word_count": 118,
  "keywords": ["pipeline", "audio", "..."],
  "full_text": "Hello and welcome ...",
  "segments": [
    {"id": 1, "start": 0.0,  "end": 4.2, "text": "Hello and welcome to the demo."},
    {"id": 2, "start": 4.2,  "end": 9.8, "text": "Today we build a transcription pipeline."}
  ]
}
```
Also exported as **SRT** (subtitles) and a human-readable **TXT** with `[HH:MM:SS.mmm -> HH:MM:SS.mmm]` prefixes.

**Downstream processing** (the "processes the result for downstream use" requirement): whitespace/punctuation cleanup, global timestamp correction across chunks, word count, language metadata, top keywords (stop-word filtered frequency), and multiple output formats (JSON for machines, SRT for video, TXT for people). The JSON is ready to feed into summarisation, search indexing, or analytics.

### Q4. How do I handle different audio formats?
1. **Don't branch per format.** Every input (WAV, MP3, M4A, FLAC, OGG, OPUS, AAC, even MP4/WEBM video) goes through **one ffmpeg normalisation step** to **16 kHz, mono, 16-bit PCM WAV**, the exact input Whisper models expect.
2. This also normalises different sample rates, stereo/multi-channel audio and bit depths, so the transcriber only ever sees one format.
3. Video containers are handled with `-vn` (strip video).
4. Detection is by content via `ffprobe`, not by file extension, so a mis-named file still works and a corrupt one fails fast with a readable message.
5. **Trade-off:** conversion costs a few seconds and temp disk space; accepted for robustness and a simpler codebase. Adding a new format needs zero code changes if ffmpeg supports it.

### Q5. How do I deal with long audio files?
| Concern | Approach |
|---------|----------|
| **Memory** | Segments are consumed from a **generator** as they are produced, so the whole transcript is never required in memory at once. Uploads are streamed to disk. |
| **Chunking** | Files over 30 minutes are split by ffmpeg (stream copy, no re-encode) into 10-minute chunks. Each chunk is transcribed independently and its timestamps are **shifted by the chunk offset** so timestamps stay globally correct. |
| **Silence** | VAD removes silent stretches, which are common in long recordings. |
| **Failure isolation** | A failure affects one chunk, not the whole job, and chunks make retry/resume possible. |
| **Parallelism** | Chunks are independent, so they can be run in parallel workers (multiprocessing, or a job queue like Celery/RQ) for near-linear speed-up. |
| **API usage** | For very long files in production, return a `job_id` immediately (async), process in a background worker, and let the client poll `GET /jobs/{id}` or receive a webhook, to avoid HTTP timeouts. |

**Known limitation (confirmed in testing):** fixed-length chunk boundaries can cut a word in half or drop a few words. In the long-file test, "...it uses the faster [whisper library to] convert my voice..." lost the words in brackets at the 20 s boundary. Mitigations: (a) add 1–2 s overlap and de-duplicate the overlapping segments, or (b) cut at silence points detected by VAD (`ffmpeg silencedetect`). Not implemented to keep the solution simple; noted as a next step.

---

## 4. Setup & Run

```bash
# 1. Install ffmpeg
#   Windows: winget install Gyan.FFmpeg      macOS: brew install ffmpeg      Linux: sudo apt install ffmpeg
# 2. Python deps
python -m venv venv && venv\Scripts\activate      # (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt

# 3a. CLI
python transcriber.py sample.mp3 --model base --out-dir output
python transcriber.py meeting.m4a --model small --language en

# 3b. REST service
uvicorn api:app --reload
curl -F "file=@sample.mp3" http://127.0.0.1:8000/transcribe
```
The first run downloads the model weights (~140 MB for `base`).

**Testing with mock data** (allowed by the task): record a 10–30 s clip on your phone, or generate one with a TTS tool, then convert it to several formats to test format handling:
```bash
ffmpeg -i sample.mp3 sample.wav
ffmpeg -i sample.mp3 sample.flac
ffmpeg -i sample.mp3 sample.ogg
```
For long-file testing, concatenate a clip repeatedly (`ffmpeg -stream_loop 60 -i sample.mp3 -c copy long.mp3`) and confirm timestamps keep increasing across the 10-minute boundaries.

---

## 4b. Verified Test Results

Tested on Windows, Python 3.13, ffmpeg 9.0.2, faster-whisper 1.2.1, model `base` (CPU, int8), with a 45.7 s recording.

| Test | Result |
|------|--------|
| m4a input | Correct transcript, 9 segments with timestamps (e.g. `[00:00:00.750 -> 00:00:05.830] Hello, this is a test of my audio transcription pipeline.`) |
| WAV, MP3, OGG (converted from the m4a) | All produced essentially the same transcript, so one normalisation step handles different formats |
| Corrupt file (`fake.mp3`, text renamed as MP3) | Clean one-line `Error: Cannot read audio file (corrupt or unsupported?)`, no crash |
| Long file (137 s looped sample, chunk size forced to 20 s via `test_long.py`) | 23 segments, timestamps continuous to 00:02:16, ids sequential, `Timestamps always increasing: True` |
| Accuracy | Mostly correct; the small `base` model makes occasional word errors (e.g. "JSON" -> "Jason"). `--model small` or larger improves this |

Run the tests yourself: `run_tests.bat` and `python test_long.py`.

---

## 5. Engineering Decisions & Trade-offs

| Decision | Why | Alternative |
|----------|-----|-------------|
| faster-whisper | Open source, accurate, offline, CPU-friendly, no per-minute cost, privacy-preserving | Cloud APIs (OpenAI, Google STT, AssemblyAI): easier, but cost and data leaves your infrastructure |
| ffmpeg normalisation | One code path for all formats | Per-format Python decoders (pydub, soundfile): more code, more edge cases |
| int8 on CPU | Runs anywhere | float16 on GPU for throughput |
| Fixed-size chunking | Simple, predictable, parallelisable | Silence-aware or overlap chunking for better boundaries |
| Model cached per process | Avoids multi-second reload | n/a |
| Output JSON + SRT + TXT | Serves machines, video tools and humans | Single format |

## 6. Error Handling
- Missing file / unreadable or corrupt audio → clear message (CLI exit, HTTP 400)
- ffmpeg/ffprobe not installed → explicit install hint
- Oversized upload → HTTP 413
- Silent audio → returns empty segment list rather than hallucinated text (VAD)
- Temp files are always cleaned up (`TemporaryDirectory`, `finally`)

## 7. Possible Improvements
- Overlap/silence-aware chunking; parallel chunk workers
- Speaker diarisation (pyannote) for multi-speaker audio
- Word-level timestamps (`word_timestamps=True`)
- Async job queue + persistent storage, auth and rate limiting
- Dockerfile for reproducible deployment; unit tests for timestamp offsets and SRT formatting
- Evaluation with Word Error Rate (WER) against reference transcripts
- Summarisation / entity extraction as further downstream steps

---

## 8. Ready-to-paste Short Answers for the Google Form

**Q1 (accepts audio file):** Python script/FastAPI service. Takes a file path (CLI) or multipart upload (`POST /transcribe`). The file is streamed to a temp file, validated with ffprobe, and cleaned up afterwards. Code: `transcriber.py`, `api.py` (share your GitHub/Drive link).

**Q2 (transcribes):** Uses open-source faster-whisper (Whisper on CTranslate2), int8 on CPU, with VAD filtering and automatic language detection. The model is loaded once and cached.

**Q3 (timestamps):** Each segment is returned with `start`/`end` seconds and text in JSON, plus SRT and TXT exports. Timestamps are corrected to be global across chunks.

**Q4 (formats):** Every input goes through a single ffmpeg step that converts it to 16 kHz mono PCM WAV, so WAV/MP3/M4A/FLAC/OGG and even video files are handled identically. Detection uses ffprobe (content, not extension) so corrupt files fail fast with a clear error.

**Q5 (long files):** Files over 30 min are split by ffmpeg into 10-min chunks (stream copy). Each is transcribed with VAD, segments are streamed from a generator to keep memory low, and timestamps are offset by chunk start. Chunks are independent, so they can be processed in parallel and retried individually. For production, run as an async job with polling/webhook. Limitation (seen in my own test): chunk boundaries can drop a few words; fixable with a small overlap and de-duplication, or by cutting at silence.
