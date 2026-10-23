"""Long-file test: loops the sample 3x (~2.3 min) and forces chunking into 20 s pieces.
Shows that timestamps keep increasing across chunk boundaries."""
import subprocess
from pathlib import Path

import transcriber as t

SRC = "sample.m4a"
LONG = "long_test.mp3"

subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-stream_loop", "2", "-i", SRC, LONG], check=True)

# Lower the thresholds just for this test (real values: 30 min / 10 min)
t.LONG_AUDIO_SECONDS = 60
t.CHUNK_SECONDS = 20

result = t.transcribe(LONG, model_size="base")
print(f"Duration: {result['duration_seconds']} s, segments: {len(result['segments'])}")
prev = -1.0
ok = True
for s in result["segments"]:
    print(f"{s['id']:>3} [{t.to_timestamp(s['start'])} -> {t.to_timestamp(s['end'])}] {s['text']}")
    if s["start"] < prev:
        ok = False
    prev = s["start"]
ids = [s["id"] for s in result["segments"]]
print("\nTimestamps always increasing:", ok)
print("Segment ids unique and sequential:", ids == list(range(1, len(ids) + 1)))
