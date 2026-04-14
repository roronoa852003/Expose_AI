import io
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from inference.video_infer import video_fake_probability, image_fake_probability
from inference.audio_infer import audio_fake_probability
from utils.metadata_infer import metadata_fake_probability
from utils.audio_explain import plot_spectrogram
from inference.audio_infer import extract_audio
from fusion.fusion import fuse
from llm.llm_auditor import audit_decision



class LlmAuditResult(BaseModel):
    consistency: str
    confidence_level: str
    explanation: str
    warnings: Optional[list[str]] = None


class AnalysisResponse(BaseModel):
    video_prob: Optional[float]
    audio_prob: Optional[float]
    meta_prob: Optional[float]
    final_score: float
    label: str
    detected_type: str
    reason: str
    override_triggered: bool
    llm_audit: Optional[LlmAuditResult] = None


app = FastAPI(title="Multimodal Deepfake Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> dict:
    return {
        "message": "Welcome to the Multimodal Deepfake Detector API",
        "health_check": "/health",
        "documentation": "/docs"
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_video(file: UploadFile = File(...)) -> AnalysisResponse:
    ALLOWED_MIME_TYPES = {
        "video/mp4", "video/x-msvideo", "video/quicktime", "video/webm",
        "audio/mpeg", "audio/wav", "audio/x-wav", "audio/ogg", "audio/mp4", "audio/x-m4a",
        "image/jpeg", "image/png", "image/webp",
        "application/octet-stream"
    }

    if file.content_type and file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a supported media file.")

    content_type = file.content_type or "application/octet-stream"
    filename = (file.filename or "").lower()
    image_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    audio_exts = (".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac")
    video_exts = (".mp4", ".mov", ".avi", ".webm", ".mkv", ".mpeg", ".mpg")

    is_audio = content_type.startswith("audio/")
    is_image = content_type.startswith("image/")
    is_video = content_type.startswith("video/")

    if content_type == "application/octet-stream":
        if filename.endswith(image_exts):
            is_image = True
            is_audio = False
            is_video = False
        elif filename.endswith(audio_exts):
            is_audio = True
            is_image = False
            is_video = False
        elif filename.endswith(video_exts):
            is_video = True
            is_audio = False
            is_image = False
        else:
            # Keep video as safest fallback for unknown binary uploads.
            is_video = True

    ext = ".mp4" if is_video else (".wav" if is_audio else ".jpg")
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        file_path = tmp.name

    video_prob = None
    audio_prob = None
    meta_prob = None

    try:
        source_type = "video"
        if is_image:
            source_type = "image"
            video_prob = image_fake_probability(file_path)
            meta_prob = metadata_fake_probability(file_path, media_type="image")
        elif is_audio:
            source_type = "audio"
            audio_prob = audio_fake_probability(file_path)
            meta_prob = metadata_fake_probability(file_path, media_type="audio")
        else:
            # Run heavy video/audio inference concurrently for faster HD processing.
            with ThreadPoolExecutor(max_workers=2) as pool:
                fut_video = pool.submit(video_fake_probability, file_path)
                fut_audio = pool.submit(audio_fake_probability, file_path)
                video_prob = fut_video.result()
                audio_prob = fut_audio.result()
            meta_prob = metadata_fake_probability(file_path, media_type="video")

        final_score, label, dtype, reason_text = fuse(
            video_prob, audio_prob, meta_prob, source_type=source_type
        )

        override_triggered = dtype in {
            "AUDIO_DEEPFAKE",
            "VIDEO_DEEPFAKE",
            "METADATA_DEEPFAKE",
        }

        llm_result: Optional[LlmAuditResult] = None
        try:
            raw = audit_decision(
                video_prob,
                audio_prob,
                meta_prob,
                label,
                dtype,
                override_triggered,
            )
            llm_result = LlmAuditResult(**raw)
        except Exception:
            llm_result = None

        return AnalysisResponse(
            video_prob=video_prob,
            audio_prob=audio_prob,
            meta_prob=meta_prob,
            final_score=final_score,
            label=label,
            detected_type=dtype,
            reason=reason_text,
            override_triggered=override_triggered,
            llm_audit=llm_result,
        )
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"CRITICAL ERROR DURING ANALYSIS:\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}\n\nTraceback:\n{error_details}")
    finally:
        # Cleanup temp file
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/api/spectrogram")
async def get_spectrogram(file: UploadFile = File(...)):
    """Extract audio from the uploaded video and return a mel spectrogram PNG."""
    content_type = file.content_type or "application/octet-stream"
    if content_type.startswith("image/"):
        raise HTTPException(status_code=204, detail="No usable audio track found")

    ext = ".mp4"
    if content_type.startswith("audio/"):
        ext = ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        video_path = tmp.name

    audio_path = extract_audio(video_path)
    if audio_path is None:
        raise HTTPException(status_code=204, detail="No usable audio track found")

    try:
        fig = plot_spectrogram(audio_path)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=120,
                    facecolor="#0a0a0a", edgecolor="none")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Spectrogram generation failed: {e}")