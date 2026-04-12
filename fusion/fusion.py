def fuse(video_prob, audio_prob, meta_prob):
    """
    Rule-guided multimodal fusion with safeguard logic.
    """

    # ---------------- Thresholds ----------------
    AUDIO_TH = 0.50         # strong audio spoof
    VIDEO_STRONG_TH = 0.50   # strong visual manipulation
    VIDEO_SUS_TH = 0.60      # subtle visual manipulation
    META_STRONG_TH = 0.85    # strong synthetic pipeline
    META_SUS_TH = 0.50       # re-encoding / suspicious

    # ---------------- HARD OVERRIDES ----------------

    # Audio-only deepfake
    if audio_prob is not None and audio_prob >= AUDIO_TH:
        return (
            audio_prob,
            "DEEPFAKE",
            "AUDIO_DEEPFAKE",
            "Audio fake probability exceeded threshold (≥ 60%)"
        )

    # Strong video manipulation
    if video_prob is not None and video_prob >= VIDEO_STRONG_TH:
        return (
            video_prob,
            "DEEPFAKE",
            "VIDEO_DEEPFAKE",
            "Visual manipulation confidence exceeded threshold (≥ 50%)"
        )

    # Subtle video + suspicious metadata
    if video_prob is not None and meta_prob is not None and video_prob >= VIDEO_SUS_TH and meta_prob >= META_SUS_TH:
        return (
            video_prob,
            "DEEPFAKE",
            "VIDEO_DEEPFAKE",
            "Moderate visual anomalies combined with suspicious metadata"
        )

    # Metadata-only (very rare, strong case)
    if meta_prob is not None and meta_prob >= META_STRONG_TH:
        return (
            meta_prob,
            "DEEPFAKE",
            "METADATA_DEEPFAKE",
            "Strong metadata anomalies indicate synthetic processing"
        )

    # ---------------- SOFT FUSION ----------------

    weights = {"video": 0.4, "audio": 0.4, "meta": 0.2}

    score = 0.0
    total = 0.0

    if video_prob is not None:
        score += video_prob * weights["video"]
        total += weights["video"]

    if audio_prob is not None:
        score += audio_prob * weights["audio"]
        total += weights["audio"]

    if meta_prob is not None:
        score += meta_prob * weights["meta"]
        total += weights["meta"]

    if total == 0:
         return (
            0.0,
            "REAL",
            "UNKNOWN",
            "No valid modalities analyzed."
        )

    final_score = score / total

    if final_score > 0.5:
        detected_type = "MULTIMODAL_DEEPFAKE"
        if video_prob is None:
            detected_type = "AUDIO_DEEPFAKE"
        elif audio_prob is None:
            detected_type = "VIDEO_DEEPFAKE"
            
        return (
            final_score,
            "DEEPFAKE",
            detected_type,
            "Combined multimodal evidence indicates manipulation"
        )

    return (
        final_score,
        "REAL",
        "NO_MANIPULATION",
        "No modality exceeded manipulation thresholds"
    )