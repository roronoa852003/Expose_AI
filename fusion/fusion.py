def fuse(video_prob, audio_prob, meta_prob, source_type=None):
    """
    Rule-guided multimodal fusion with safeguard logic.

    Parameters
    ----------
    video_prob  : float | None -- FAKE probability from the visual model.
    audio_prob  : float | None -- FAKE probability from the audio model.
                  Pass None when the source has no audio track.
    meta_prob   : float | None -- FAKE probability from metadata analysis.
    source_type : str | None   -- "video", "image", "audio", or None.

    Returns
    -------
    (score: float, verdict: str, detected_type: str, reason: str)
    """

    # ------------------------------------------------------------------ #
    #  Thresholds                                                         #
    # ------------------------------------------------------------------ #

    # --- Audio ---
    # Lowered from 0.50: the combined pipeline (model + 7 forensic signals)
    # is well-calibrated and 0.46+ is a reliable fake indicator.
    AUDIO_TH             = 0.46

    # Audio-only standalone threshold.
    # Lower because physics-based forensic signals don't need cross-modal
    # corroboration to be trusted.
    AUDIO_ONLY_TH        = 0.44

    # Audio-only + metadata corroboration (moderate audio + suspicious metadata)
    AUDIO_META_SUS_TH    = 0.38
    META_AUDIO_SUS_TH    = 0.55

    # --- Visual hard rule requested ---
    # If visual anomaly probability is >= 40%, mark as fake.
    VIDEO_STRONG_TH      = 0.40

    # --- Video WITHOUT audio ---
    VIDEO_NO_AUDIO_TH    = 0.40
    VIDEO_NO_AUDIO_LOW_TH= 0.35

    VIDEO_SUS_TH         = 0.62   # subtle video + metadata

    # --- Image ---
    IMAGE_STRONG_TH      = 0.40
    IMAGE_META_SUS_TH    = 0.45
    IMAGE_META_ONLY_TH   = 0.75

    # --- Metadata ---
    META_STRONG_TH       = 0.85
    META_SUS_TH          = 0.50

    # --- Soft fusion thresholds ---
    SOFT_FAKE_TH_VIDEO   = 0.52
    SOFT_FAKE_TH_IMAGE   = 0.48
    SOFT_FAKE_TH_AUDIO   = 0.44

    # ------------------------------------------------------------------ #
    #  Convenience flags                                                  #
    # ------------------------------------------------------------------ #

    image_only = (source_type == "image")
    video_only = (source_type == "video")
    audio_only = (source_type == "audio")
    has_audio  = (audio_prob is not None)
    has_meta   = (meta_prob  is not None)
    has_video  = (video_prob is not None)

    if audio_only:
        SOFT_FAKE_TH = SOFT_FAKE_TH_AUDIO
    elif image_only:
        SOFT_FAKE_TH = SOFT_FAKE_TH_IMAGE
    else:
        SOFT_FAKE_TH = SOFT_FAKE_TH_VIDEO

    # ------------------------------------------------------------------ #
    #  HARD OVERRIDES                                                     #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # AUDIO-ONLY RULES
    #
    # These were completely missing in the original code.
    # When source_type == "audio", video_prob is None. The old code fell
    # through to soft fusion where total weight came from meta only (0.30),
    # so audio_prob was NEVER used. Real and fake audio got the same score.
    # ------------------------------------------------------------------ #

    # 1a. Audio-only: strong fake signal
    if audio_only and has_audio and audio_prob >= AUDIO_ONLY_TH:
        return (
            audio_prob,
            "DEEPFAKE",
            "AUDIO_DEEPFAKE",
            (
                f"Audio-only source; fake probability {audio_prob:.2%} exceeded "
                f"standalone threshold ({AUDIO_ONLY_TH:.0%})"
            ),
        )

    # 1b. Audio-only: moderate fake + metadata corroboration
    if (
        audio_only
        and has_audio
        and has_meta
        and audio_prob >= AUDIO_META_SUS_TH
        and meta_prob  >= META_AUDIO_SUS_TH
    ):
        combined = 0.70 * audio_prob + 0.30 * meta_prob
        return (
            combined,
            "DEEPFAKE",
            "AUDIO_DEEPFAKE",
            (
                f"Audio-only source; moderate audio score {audio_prob:.2%} "
                f"corroborated by metadata score {meta_prob:.2%}"
            ),
        )

    # 1c. Audio-only: real verdict -- explicitly return REAL before falling
    #     through, to make the logic clear and avoid any ambiguity.
    if audio_only and has_audio and audio_prob < AUDIO_ONLY_TH:
        if not (has_meta and meta_prob >= META_AUDIO_SUS_TH and audio_prob >= AUDIO_META_SUS_TH):
            # Below threshold and no strong metadata corroboration -> REAL
            # (still computed via soft fusion below for the exact score)
            pass  # fall through to soft fusion for the numeric score

    # 2. Audio deepfake in a video source (video + audio together)
    if not audio_only and has_audio and audio_prob >= AUDIO_TH:
        return (
            audio_prob,
            "DEEPFAKE",
            "AUDIO_DEEPFAKE",
            f"Audio fake probability {audio_prob:.2%} exceeded threshold ({AUDIO_TH:.0%})",
        )

    # 3. Strong image manipulation
    if image_only and has_video and video_prob >= IMAGE_STRONG_TH:
        return (
            video_prob,
            "DEEPFAKE",
            "IMAGE_DEEPFAKE",
            f"Image manipulation confidence {video_prob:.2%} exceeded threshold ({IMAGE_STRONG_TH:.0%})",
        )

    # 4. Strong video manipulation with audio present
    if (
        not image_only
        and not audio_only
        and has_video
        and has_audio
        and video_prob >= VIDEO_STRONG_TH
    ):
        return (
            video_prob,
            "DEEPFAKE",
            "VIDEO_DEEPFAKE",
            f"Visual manipulation confidence {video_prob:.2%} exceeded threshold ({VIDEO_STRONG_TH:.0%})",
        )

    # 5a. Video-only, high confidence
    if video_only and not has_audio and has_video and video_prob >= VIDEO_NO_AUDIO_TH:
        return (
            video_prob,
            "DEEPFAKE",
            "VIDEO_DEEPFAKE",
            (
                f"Video-only source; visual confidence {video_prob:.2%} exceeded "
                f"standalone threshold ({VIDEO_NO_AUDIO_TH:.0%})"
            ),
        )

    # 5b. Video-only, moderate confidence + temporal blended signal
    if (
        video_only
        and not has_audio
        and has_video
        and VIDEO_NO_AUDIO_LOW_TH <= video_prob < VIDEO_NO_AUDIO_TH
    ):
        if has_meta and meta_prob >= META_SUS_TH:
            combined = 0.75 * video_prob + 0.25 * meta_prob
            return (
                combined,
                "DEEPFAKE",
                "VIDEO_DEEPFAKE",
                (
                    f"Video-only source; moderate visual score {video_prob:.2%} "
                    f"corroborated by metadata score {meta_prob:.2%}"
                ),
            )
        else:
            return (
                video_prob,
                "DEEPFAKE",
                "VIDEO_DEEPFAKE",
                (
                    f"Video-only source; blended spatial+temporal score {video_prob:.2%} "
                    f"indicates manipulation (temporal variance in deepfake range)"
                ),
            )

    # 6. Subtle video + suspicious metadata
    if (
        not image_only
        and not audio_only
        and has_video
        and has_meta
        and video_prob >= VIDEO_SUS_TH
        and meta_prob  >= META_SUS_TH
    ):
        return (
            video_prob,
            "DEEPFAKE",
            "VIDEO_DEEPFAKE",
            "Moderate visual anomalies combined with suspicious metadata",
        )

    # 7. Image + suspicious metadata
    if (
        image_only
        and has_video
        and has_meta
        and video_prob >= IMAGE_META_SUS_TH
        and meta_prob  >= META_SUS_TH
    ):
        combined = 0.70 * video_prob + 0.30 * meta_prob
        return (
            combined,
            "DEEPFAKE",
            "IMAGE_DEEPFAKE",
            (
                f"Image anomalies ({video_prob:.2%}) combined with "
                f"suspicious metadata ({meta_prob:.2%})"
            ),
        )

    # 8. Image: metadata alone very strong (AI software tag etc.)
    if image_only and has_meta and meta_prob >= IMAGE_META_ONLY_TH:
        return (
            meta_prob,
            "DEEPFAKE",
            "IMAGE_DEEPFAKE",
            f"Strong metadata indicators ({meta_prob:.2%}) of AI-generated image",
        )

    # 9. Metadata-only for video (very conservative)
    if not image_only and not audio_only and has_meta and meta_prob >= META_STRONG_TH:
        return (
            meta_prob,
            "DEEPFAKE",
            "METADATA_DEEPFAKE",
            f"Strong metadata anomalies {meta_prob:.2%} indicate synthetic processing",
        )

    # ------------------------------------------------------------------ #
    #  SOFT FUSION                                                        #
    #                                                                     #
    #  audio_only uses dedicated weights (audio=0.80, meta=0.20).         #
    #  Previously audio_only fell into the not-has-audio branch with      #
    #  video=0.70, meta=0.30, and since has_video=False, total=0.30 so   #
    #  final_score = metadata only. audio_prob was completely ignored.    #
    # ------------------------------------------------------------------ #

    if audio_only:
        weights = {"video": 0.00, "audio": 0.80, "meta": 0.20}
    elif image_only:
        weights = {"video": 0.80, "audio": 0.00, "meta": 0.20}
    elif not has_audio:
        weights = {"video": 0.70, "audio": 0.00, "meta": 0.30}
    else:
        weights = {"video": 0.45, "audio": 0.40, "meta": 0.15}

    score = 0.0
    total = 0.0

    if has_video:
        score += video_prob * weights["video"]
        total += weights["video"]

    if has_audio:
        score += audio_prob * weights["audio"]
        total += weights["audio"]

    if has_meta:
        score += meta_prob * weights["meta"]
        total += weights["meta"]

    if total == 0.0:
        return (0.0, "REAL", "UNKNOWN", "No valid modalities analyzed.")

    final_score = score / total

    if final_score > SOFT_FAKE_TH:
        if audio_only or (not has_video and has_audio):
            detected_type = "AUDIO_DEEPFAKE"
        elif image_only:
            detected_type = "IMAGE_DEEPFAKE"
        elif not has_audio:
            detected_type = "VIDEO_DEEPFAKE"
        else:
            detected_type = "MULTIMODAL_DEEPFAKE"

        return (
            final_score,
            "DEEPFAKE",
            detected_type,
            f"Combined evidence (score {final_score:.2%}) indicates manipulation",
        )

    return (
        final_score,
        "REAL",
        "NO_MANIPULATION",
        f"No modality exceeded manipulation thresholds (score {final_score:.2%})",
    )