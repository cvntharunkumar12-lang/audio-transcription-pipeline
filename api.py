"""
Optional REST service wrapper.

Run:   uvicorn api:app --reload
Test:  curl -F "file=@sample.mp3" http://127.0.0.1:8000/transcribe
"""
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from transcriber import TranscriptionError, load_model, transcribe

app = FastAPI(title="Transcription Service")
MAX_UPLOAD_MB = 500


@app.on_event("startup")
def warm_up() -> None:
    load_model("base")  # load once, not per request


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe_endpoint(file: UploadFile = File(...), model: str = "base", language: str | None = None):
    suffix = Path(file.filename or "audio").suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)  # stream to disk, never hold in RAM
        tmp_path = Path(tmp.name)
    try:
        if tmp_path.stat().st_size > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(413, f"File larger than {MAX_UPLOAD_MB} MB")
        return transcribe(tmp_path, model_size=model, language=language)
    except TranscriptionError as e:
        raise HTTPException(400, str(e))
    finally:
        tmp_path.unlink(missing_ok=True)
