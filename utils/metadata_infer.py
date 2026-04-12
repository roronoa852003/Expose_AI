import subprocess
import shutil
import json
import os

# CHANGE THIS PATH IF YOUR FFmpeg IS INSTALLED ELSEWHERE
def metadata_fake_probability(file_path, media_type="video"):
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        # Fallback for Windows local development
        fallback = "C:/ffmpeg/bin/ffprobe.exe"
        if os.path.exists(fallback):
            ffprobe = fallback
        else:
            # On Render/Linux, ffprobe must be in PATH
            return 0.28
    
    cmd = [
        ffprobe,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        file_path
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if not result.stdout:
        return 0.0

    data = json.loads(result.stdout)
    score = 0.0

    tags = data.get("format", {}).get("tags", {})
    encoder = tags.get("encoder", "").lower()
    software = tags.get("software", "").lower()
    make = tags.get("make", "")
    model = tags.get("model", "")

    if media_type == "audio":
        if "ffmpeg" in encoder or "lavf" in encoder:
            score += 0.4
        elif not software and not encoder:
            score += 0.2
            
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "audio":
                try:
                    sample_rate = int(stream.get("sample_rate", "0"))
                    # AI tools often generate lower sample rates like 16k or 24k
                    if sample_rate not in [44100, 48000] and sample_rate > 0:
                        score += 0.2
                except Exception:
                    pass

    elif media_type == "image":
        if not make and not model and not software:
            score += 0.4 # Missing common camera EXIF
        
        # Software hints of generative AI
        generative_software = ["stable diffusion", "midjourney", "dall-e", "comfyui", "photoshop"]
        if any(tool in software for tool in generative_software) or any(tool in encoder for tool in generative_software):
            score += 0.6

    else: # media_type == "video"
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

    return min(score, 1.0)