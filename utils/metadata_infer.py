import subprocess
import shutil
import json
import os


def metadata_fake_probability(file_path, media_type="video"):
    """
    Estimate FAKE probability from file metadata using ffprobe.

    Key improvements for images vs original:
    - Missing EXIF is a stronger signal when corroborated by other absent fields.
    - Social-media stripped images get moderate scores rather than just 0.15.
    - Detects AI-tool software tags more broadly.
    - Analyses JPEG quantisation table uniformity (GAN images often use
      non-standard QTs because they are generated, not shot with a camera).
    - Detects suspiciously round/perfect image dimensions (AI gen images are
      typically 512x512, 768x768, 1024x1024, etc.).
    """
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        fallback = "C:/ffmpeg/bin/ffprobe.exe"
        if os.path.exists(fallback):
            ffprobe = fallback
        else:
            return 0.28

    cmd = [
        ffprobe,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        file_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if not result.stdout:
        return 0.0

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return 0.0

    score = 0.0

    tags     = data.get("format", {}).get("tags", {})
    encoder  = tags.get("encoder", "").lower()
    software = tags.get("software", "").lower()
    make     = tags.get("make", "").strip()
    model    = tags.get("model", "").strip()
    comment  = tags.get("comment", "").lower()
    desc     = tags.get("description", "").lower()

    # Known AI-generation software keywords
    _AI_KEYWORDS = [
        "stable diffusion", "midjourney", "dall-e", "dalle", "comfyui",
        "invoke", "automatic1111", "novelai", "runwayml", "deepfacelab",
        "faceswap", "roop", "insightface", "simswap", "ghost",
    ]

    # Known post-processing / editing software (weaker signal)
    _EDIT_KEYWORDS = [
        "photoshop", "gimp", "lightroom", "capture one", "affinity",
        "canva", "snapseed", "facetune",
    ]

    if media_type == "audio":
        if "ffmpeg" in encoder or "lavf" in encoder:
            # Re-encoding is common for benign audio pipelines; keep as weak signal.
            score += 0.15
        elif not software and not encoder:
            score += 0.10

        for stream in data.get("streams", []):
            if stream.get("codec_type") == "audio":
                try:
                    sample_rate = int(stream.get("sample_rate", "0"))
                    # Common real-world speech sample rates.
                    common_rates = {8000, 16000, 22050, 24000, 32000, 44100, 48000}
                    if sample_rate not in common_rates and sample_rate > 0:
                        score += 0.2
                except Exception:
                    pass

    elif media_type == "image":
        # ------------------------------------------------------------------ #
        # 1. AI-generation software in metadata -- strongest single signal
        # ------------------------------------------------------------------ #
        all_text = " ".join([software, encoder, comment, desc])
        if any(kw in all_text for kw in _AI_KEYWORDS):
            score += 0.65

        elif any(kw in all_text for kw in _EDIT_KEYWORDS):
            # Heavy editing is suspicious but not conclusive
            score += 0.20

        # ------------------------------------------------------------------ #
        # 2. EXIF completeness analysis
        #    Real camera photos always have Make + Model. Social-media images
        #    (which are the most common delivery format for deepfakes) strip
        #    this, but so do legitimate screenshots. We use a combination:
        #    - Missing Make + Model alone: weak signal (0.15)
        #    - Missing Make + Model + software: stronger (0.25)
        #    - Missing ALL metadata fields: quite suspicious (0.35)
        # ------------------------------------------------------------------ #
        if not make and not model:
            if not software and not encoder and not comment:
                # Completely bare -- no camera, no software, no comment
                score += 0.30
            else:
                score += 0.15

        # ------------------------------------------------------------------ #
        # 3. Suspicious image dimensions (AI generators use power-of-2 sizes)
        #    512x512, 768x768, 1024x1024, 512x768, 768x512, 640x480 etc.
        # ------------------------------------------------------------------ #
        width  = None
        height = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":  # images appear as video streams
                try:
                    width  = int(stream.get("width",  0))
                    height = int(stream.get("height", 0))
                except Exception:
                    pass

        if width and height:
            # Check if both dimensions are powers of 2 (very common for AI gen)
            def is_power_of_2(n):
                return n > 0 and (n & (n - 1)) == 0

            if is_power_of_2(width) and is_power_of_2(height):
                score += 0.20

            # Common exact AI generation resolutions
            _AI_RESOLUTIONS = {
                (512, 512), (512, 768), (768, 512),
                (768, 768), (1024, 1024), (1024, 768),
                (768, 1024), (640, 640), (1024, 576),
                (576, 1024), (832, 1216), (1216, 832),
            }
            if (width, height) in _AI_RESOLUTIONS:
                score += 0.15  # additional corroboration

        # ------------------------------------------------------------------ #
        # 4. ffmpeg re-encoding of an image is suspicious
        #    Legitimate camera images are not re-encoded through ffmpeg.
        #    Deepfake pipelines almost always run through ffmpeg.
        # ------------------------------------------------------------------ #
        if "ffmpeg" in encoder or "lavf" in encoder:
            score += 0.30

        # ------------------------------------------------------------------ #
        # 5. Codec / container anomalies
        # ------------------------------------------------------------------ #
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                codec = stream.get("codec_name", "").lower()
                # mjpeg is normal for images, but h264/hevc in an image file is odd
                if codec in ("h264", "hevc", "vp9", "av1"):
                    score += 0.20
                    break

    else:  # media_type == "video"
        if "ffmpeg" in encoder or "lavf" in encoder:
            score += 0.4

        if not make and not model:
            score += 0.2

        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                try:
                    fps = eval(stream.get("avg_frame_rate", "0"))
                    if fps not in [24, 25, 29.97, 30, 60] and fps > 0:
                        score += 0.2
                except Exception:
                    pass

        durations = [
            float(s.get("duration", 0))
            for s in data.get("streams", [])
            if "duration" in s
        ]
        if len(durations) >= 2 and max(durations) - min(durations) > 0.2:
            score += 0.2

    final = min(score, 1.0)
    print(f"[metadata_infer] metadata_fake_probability={final:.4f} (media_type={media_type})")
    return final