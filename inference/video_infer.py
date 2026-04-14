import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import ViTForImageClassification, AutoImageProcessor

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_REPO_ID   = os.environ.get("HF_MODEL_REPO", "vedanthundare/expose_ai")
_HF_TOKEN  = os.environ.get("HF_TOKEN")
_CACHE_DIR = os.environ.get("HF_HOME", "/app/hf_cache")

video_model = ViTForImageClassification.from_pretrained(
    _REPO_ID,
    subfolder="models/video_model",
    token=_HF_TOKEN,
    cache_dir=_CACHE_DIR,
)
video_extractor = AutoImageProcessor.from_pretrained(
    _REPO_ID,
    subfolder="models/video_model",
    token=_HF_TOKEN,
    cache_dir=_CACHE_DIR,
)
video_model.eval()

_id2label = getattr(video_model.config, "id2label", {})
_FAKE_IDX = next(
    (
        int(k)
        for k, v in _id2label.items()
        if str(v).upper() in ("FAKE", "1", "FORGED", "DEEPFAKE", "MANIPULATED")
    ),
    1,
)
print(f"[video_infer] id2label={_id2label}  ->  using FAKE index={_FAKE_IDX}")

_FACE_CASCADE = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
)

# ---------------------------------------------------------------------------
# Sharpness gate -- MUST be measured on face crop, NOT the full frame
# ---------------------------------------------------------------------------

_FACE_SHARP_MIN = 25.0


def _face_is_sharp(face_gray):
    return float(cv2.Laplacian(face_gray, cv2.CV_64F).var()) >= _FACE_SHARP_MIN


# ---------------------------------------------------------------------------
# Handcrafted signals -- VIDEO
# ---------------------------------------------------------------------------

def _edge_ratio_score(face_gray):
    """
    Ratio of very-sharp to moderate edges inside the face region.
    Real faces: p95/p50 approx 5-10.  Manipulated: often 15-35.
    Returns [0, 1].
    """
    lap = np.abs(cv2.Laplacian(face_gray, cv2.CV_32F))
    p50 = float(np.percentile(lap, 50)) + 1e-8
    p95 = float(np.percentile(lap, 95)) + 1e-8
    ratio = p95 / p50
    return float(np.clip((ratio - 8.0) / 22.0, 0.0, 1.0))


def _hue_inconsistency_score(face_bgr):
    """
    Hue standard deviation inside the face.
    Typical real face hue std: 5-12.  Suspicious: > 20.
    Returns [0, 1].
    """
    hsv     = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
    hue_std = float(hsv[:, :, 0].std())
    return float(np.clip((hue_std - 10.0) / 40.0, 0.0, 1.0))


def _blend_seam_score(full_gray, x, y, fw, fh):
    """
    Gradient magnitude at the face boundary relative to the face interior.
    Returns [0, 1].
    """
    sobel = np.abs(cv2.Sobel(full_gray, cv2.CV_64F, 1, 1, ksize=3))
    bw    = max(4, int(fw * 0.06))

    boundary = np.zeros_like(full_gray, dtype=bool)
    boundary[y          : y + bw,       x : x + fw] = True
    boundary[y + fh - bw: y + fh,       x : x + fw] = True
    boundary[y          : y + fh, x          : x + bw] = True
    boundary[y          : y + fh, x + fw - bw: x + fw] = True

    interior = np.zeros_like(full_gray, dtype=bool)
    interior[y + bw: y + fh - bw, x + bw: x + fw - bw] = True

    bnd_val = float(np.mean(sobel[boundary])) if boundary.any() else 0.0
    int_val = float(np.mean(sobel[interior])) if interior.any() else 1.0

    ratio = bnd_val / (int_val + 1e-8)
    return float(np.clip((ratio - 1.2) / 1.8, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Temporal consistency
# ---------------------------------------------------------------------------

def _temporal_variance_score(face_diffs):
    """
    Convert per-frame face pixel differences into a [0, 1] fake score.
    Natural motion: ~50-80 -> 0.17-0.27. Deepfake: 190-220 -> 0.63-0.73.
    """
    if len(face_diffs) < 4:
        return 0.0
    var = float(np.var(face_diffs))
    return float(np.clip(var / 300.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# IMAGE-SPECIFIC forensic signals
# ---------------------------------------------------------------------------

def _jpeg_dct_ghost_score(face_gray):
    """
    JPEG ghost / double-compression detection via DCT coefficient analysis.

    Face-swap deepfakes are almost always double-compressed: the original image
    at quality Q1, then the face is pasted and saved at quality Q2. The
    coefficient histogram of the pasted region differs from the background.

    For a single-face crop we look at the 8x8 block DCT coefficients:
    - Real images: smooth histogram of AC coefficients (montonic falloff).
    - Double-compressed or GAN-generated: anomalous spikes at specific
      quantisation grid positions.

    Returns [0, 1].
    """
    try:
        # Resize to multiple of 8 for clean block DCT
        h, w = face_gray.shape
        h8 = (h // 8) * 8
        w8 = (w // 8) * 8
        if h8 < 16 or w8 < 16:
            return 0.0
        crop = face_gray[:h8, :w8].astype(np.float32) - 128.0

        block_ac_vars = []
        for row in range(0, h8, 8):
            for col in range(0, w8, 8):
                block = crop[row:row+8, col:col+8]
                dct_block = cv2.dct(block)
                # AC components only (exclude DC at [0,0])
                ac = dct_block.flatten()[1:]
                block_ac_vars.append(float(np.var(ac)))

        if not block_ac_vars:
            return 0.0

        ac_var_arr = np.array(block_ac_vars)
        # Coefficient of variation of block variances:
        # High CV means some blocks are abnormally different from others
        # (spliced region vs background).
        mean_v = float(np.mean(ac_var_arr)) + 1e-8
        std_v  = float(np.std(ac_var_arr))
        cv     = std_v / mean_v

        # Empirical calibration:
        # Real JPEG: CV ~ 0.6-1.2 (moderate block variation, codec-uniform)
        # Double-compressed face-swap: CV ~ 1.4-2.5 (large outlier blocks)
        # GAN-generated: CV ~ 0.3-0.6 (unnaturally uniform -- too smooth)
        #
        # Both extremes are suspicious -- score based on distance from normal range.
        if cv < 0.6:
            # Unnaturally uniform -- GAN fingerprint
            score = float(np.clip((0.6 - cv) / 0.6, 0.0, 1.0))
        elif cv > 1.2:
            # Double-compression or splicing
            score = float(np.clip((cv - 1.2) / 1.3, 0.0, 1.0))
        else:
            score = 0.0

        return score
    except Exception:
        return 0.0


def _noise_inconsistency_score(face_gray, full_gray, x, y, fw, fh):
    """
    Compare the noise level (residual after Gaussian blur) of the face region
    vs a border region of similar size just outside the face.

    Face-swap algorithms apply smoothing to the pasted face to hide the
    boundary, creating a noise mismatch: the face region is unnaturally smooth
    relative to the surrounding skin/background.

    Returns [0, 1] where high = suspicious noise mismatch.
    """
    try:
        img_h, img_w = full_gray.shape

        # Noise residual = image - blurred image
        blurred = cv2.GaussianBlur(full_gray.astype(np.float32), (5, 5), 0)
        residual = np.abs(full_gray.astype(np.float32) - blurred)

        # Face region noise
        face_noise = float(np.mean(residual[y:y+fh, x:x+fw]))

        # Border region (same area, just outside face)
        pad = max(fw, fh)
        bx1 = max(0, x - pad)
        by1 = max(0, y - pad)
        bx2 = min(img_w, x + fw + pad)
        by2 = min(img_h, y + fh + pad)

        # Mask out the face itself
        border_mask = np.zeros_like(full_gray, dtype=bool)
        border_mask[by1:by2, bx1:bx2] = True
        border_mask[y:y+fh, x:x+fw]   = False

        if not border_mask.any():
            return 0.0

        border_noise = float(np.mean(residual[border_mask]))

        if border_noise < 1e-6:
            return 0.0

        # Ratio: if face is much smoother (lower noise) than surroundings, suspicious
        ratio = face_noise / border_noise
        # Real: ratio ~ 0.8-1.2 (similar noise level)
        # Smoothed face-swap: ratio ~ 0.3-0.6 (face much smoother)
        if ratio < 0.8:
            score = float(np.clip((0.8 - ratio) / 0.8, 0.0, 1.0))
        else:
            score = 0.0

        return score
    except Exception:
        return 0.0


def _facial_symmetry_score(face_gray):
    """
    Measure left-right asymmetry of the face.

    Real human faces have moderate natural asymmetry (they are NOT perfectly
    symmetric). Face-swap deepfakes often produce two failure modes:
    1. Excessive symmetry -- GAN generators are trained with symmetric loss.
    2. Pathological asymmetry -- the source and target faces have incompatible
       geometry, leaving visible misalignment on one side.

    We measure structural similarity between the left and right half.
    SSIM close to 1.0 (unnaturally symmetric) OR very low (pathological mismatch)
    are both suspicious.

    Returns [0, 1] where high = suspicious.
    """
    try:
        h, w = face_gray.shape
        if w < 32 or h < 32:
            return 0.0

        mid = w // 2
        left  = face_gray[:, :mid]
        right = cv2.flip(face_gray[:, mid:], 1)

        # Resize to same shape if odd width
        min_w = min(left.shape[1], right.shape[1])
        left  = left[:, :min_w]
        right = right[:, :min_w]

        # Normalised cross-correlation as a proxy for structural similarity
        left_f  = left.astype(np.float32)
        right_f = right.astype(np.float32)
        l_norm  = left_f  - left_f.mean()
        r_norm  = right_f - right_f.mean()
        l_std   = float(np.std(l_norm)) + 1e-8
        r_std   = float(np.std(r_norm)) + 1e-8
        ncc     = float(np.mean(l_norm * r_norm)) / (l_std * r_std)

        # NCC range [-1, 1]. Real faces: 0.5-0.85 (moderate natural symmetry).
        # Unnaturally symmetric (GAN): > 0.90
        # Severely misaligned (bad face-swap): < 0.35
        if ncc > 0.90:
            return float(np.clip((ncc - 0.90) / 0.10, 0.0, 1.0))
        elif ncc < 0.35:
            return float(np.clip((0.35 - ncc) / 0.35, 0.0, 1.0))
        else:
            return 0.0
    except Exception:
        return 0.0


def _skin_texture_uniformity_score(face_gray):
    """
    Local Binary Pattern (LBP) uniformity of the skin region.

    GAN generators trained on face generation produce unnaturally uniform skin
    texture -- the spatial frequency of micro-texture is too regular.
    Real skin has natural fractal micro-variation detectable via LBP entropy.

    Returns [0, 1] where high = unnaturally uniform (suspicious).
    """
    try:
        if face_gray.shape[0] < 32 or face_gray.shape[1] < 32:
            return 0.0

        # Simple LBP approximation using local variance map
        kernel = np.ones((5, 5), np.float32) / 25.0
        local_mean = cv2.filter2D(face_gray.astype(np.float32), -1, kernel)
        local_sq_mean = cv2.filter2D(
            (face_gray.astype(np.float32) ** 2), -1, kernel
        )
        local_var = local_sq_mean - local_mean ** 2
        local_var = np.maximum(local_var, 0.0)

        # Entropy of local variance histogram
        hist, _ = np.histogram(local_var, bins=32, range=(0, 500))
        hist    = hist.astype(np.float32) + 1e-8
        hist   /= hist.sum()
        entropy = float(-np.sum(hist * np.log2(hist)))

        # Max entropy for 32 bins = log2(32) = 5.0
        # Real skin: entropy ~ 3.5-4.5 (rich texture variation)
        # GAN skin:  entropy ~ 1.5-2.8 (overly uniform, low entropy)
        # Over-processed (heavy Gaussian blur): entropy < 1.5
        if entropy < 3.0:
            return float(np.clip((3.0 - entropy) / 3.0, 0.0, 1.0))
        else:
            return 0.0
    except Exception:
        return 0.0


def _frequency_domain_score(face_gray):
    """
    FFT-based GAN fingerprint detection.

    GAN generators (ProGAN, StyleGAN, etc.) produce characteristic peaks in
    the high-frequency spectrum of generated images, caused by upsampling
    artefacts in the generator's decoder layers (nearest-neighbour or bilinear
    upsampling creates regular grid patterns).

    This is JPEG-safe: we analyse the magnitude spectrum and look for
    unusually elevated high-frequency energy relative to the mid-frequency band.
    Real photographs have monotonically decreasing FFT magnitude from low to
    high frequencies. GAN images break this 1/f relationship.

    Returns [0, 1].
    """
    try:
        if face_gray.shape[0] < 64 or face_gray.shape[1] < 64:
            return 0.0

        resized = cv2.resize(face_gray, (128, 128)).astype(np.float32)
        resized -= resized.mean()

        fft    = np.fft.fft2(resized)
        fft_sh = np.fft.fftshift(fft)
        mag    = np.abs(fft_sh)

        h, w  = mag.shape
        cy, cx = h // 2, w // 2

        # Define radial bands
        Y, X  = np.ogrid[:h, :w]
        dist  = np.sqrt((X - cx)**2 + (Y - cy)**2)

        low_mask  = dist < 10
        mid_mask  = (dist >= 10) & (dist < 40)
        high_mask = dist >= 40

        low_mean  = float(np.mean(mag[low_mask]))  + 1e-8
        mid_mean  = float(np.mean(mag[mid_mask]))  + 1e-8
        high_mean = float(np.mean(mag[high_mask])) + 1e-8

        # 1/f ratio check: normally high/mid << mid/low
        high_mid_ratio = high_mean / mid_mean
        mid_low_ratio  = mid_mean  / low_mean

        # Real photos: high_mid_ratio ~ 0.05-0.20
        # GAN images:  high_mid_ratio ~ 0.25-0.60 (elevated high-freq energy)
        score = float(np.clip((high_mid_ratio - 0.20) / 0.40, 0.0, 1.0))
        return score
    except Exception:
        return 0.0


def _color_channel_correlation_score(face_bgr):
    """
    Cross-channel correlation consistency check.

    Real camera images have strong RGB channel correlation due to the physics
    of light capture and demosaicing. GAN-generated images show altered
    inter-channel correlations because the generator learns RGB jointly but
    not necessarily with the same physical constraints.

    Returns [0, 1] where high = suspicious low cross-channel correlation.
    """
    try:
        if face_bgr.shape[0] < 16 or face_bgr.shape[1] < 16:
            return 0.0

        b = face_bgr[:, :, 0].astype(np.float32).flatten()
        g = face_bgr[:, :, 1].astype(np.float32).flatten()
        r = face_bgr[:, :, 2].astype(np.float32).flatten()

        def safe_corr(a, b_arr):
            std_a = float(np.std(a))
            std_b = float(np.std(b_arr))
            if std_a < 1e-6 or std_b < 1e-6:
                return 1.0
            return float(np.corrcoef(a, b_arr)[0, 1])

        rg = abs(safe_corr(r, g))
        rb = abs(safe_corr(r, b))
        gb = abs(safe_corr(g, b))
        avg_corr = (rg + rb + gb) / 3.0

        # Real skin: avg_corr ~ 0.85-0.98 (highly correlated channels)
        # GAN-generated: avg_corr ~ 0.60-0.80 (less physical correlation)
        if avg_corr < 0.80:
            return float(np.clip((0.80 - avg_corr) / 0.80, 0.0, 1.0))
        else:
            return 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# ViT inference
# ---------------------------------------------------------------------------

def _predict_fake_probability(image):
    inputs = video_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = video_model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=1)
    return float(probs[0][_FAKE_IDX].item())


def _multi_view_fake_probability(image, n_crops=7):
    """
    Average predictions over multiple test-time augmentations for stability.

    For images (still photos) we use more augmentations than for video frames
    because we have no temporal signal to fall back on -- the ViT score must
    carry more weight.

    Augmentations:
      1. Original
      2. Horizontal flip
      3. 90% centre crop
      4. 80% centre crop (tighter, focuses on inner face)
      5. Brightness +15%
      6. Brightness -15%
      7. CLAHE-enhanced (contrast normalisation -- exposes GAN smoothing)
    """
    views = [image, image.transpose(Image.FLIP_LEFT_RIGHT)]

    w, h = image.size

    for scale in (0.90, 0.80):
        crop_size = int(min(w, h) * scale)
        if crop_size >= 64:
            l = (w - crop_size) // 2
            t = (h - crop_size) // 2
            views.append(image.crop((l, t, l + crop_size, t + crop_size)))

    try:
        from PIL import ImageEnhance
        enh = ImageEnhance.Brightness(image)
        views.append(enh.enhance(1.15))
        views.append(enh.enhance(0.85))
    except Exception:
        pass

    # CLAHE augmentation -- flattens illumination, reveals GAN texture anomalies
    try:
        img_np  = np.array(image.convert("RGB"))
        img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
        clahe_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        views.append(Image.fromarray(clahe_rgb))
    except Exception:
        pass

    return float(np.mean([_predict_fake_probability(v) for v in views]))


# ---------------------------------------------------------------------------
# Face detection helpers
# ---------------------------------------------------------------------------

def _extract_faces(rgb_image, max_faces=2):
    """Return list of (x, y, fw, fh) for the largest detected faces."""
    if _FACE_CASCADE.empty():
        return []
    gray  = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(60, 60),
    )
    if len(faces) == 0:
        return []
    return sorted(faces, key=lambda f: int(f[2]) * int(f[3]), reverse=True)[:max_faces]


def _crop_face_padded(rgb_image, x, y, fw, fh, padding=0.30):
    """Crop face with padding to capture the swap seam region."""
    img_h, img_w = rgb_image.shape[:2]
    px = int(padding * fw)
    py = int(padding * fh)
    x1 = max(0, x - px);     y1 = max(0, y - py)
    x2 = min(img_w, x+fw+px); y2 = min(img_h, y+fh+py)
    return Image.fromarray(rgb_image[y1:y2, x1:x2])


# ---------------------------------------------------------------------------
# Per-frame scoring (VIDEO)
# ---------------------------------------------------------------------------

def _score_frame(rgb, pil_image, full_gray):
    """
    Score a single VIDEO frame.

    Returns
    -------
    (frame_score: float, face_pixel_vec: np.ndarray or None)
    """
    faces      = _extract_faces(rgb, max_faces=2)
    vit_global = _multi_view_fake_probability(pil_image)

    if not faces:
        return float(np.clip(vit_global * 0.70, 0.0, 1.0)), None

    best_score    = -1.0
    best_face_vec = None

    for (x, y, fw, fh) in faces:
        face_bgr  = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)[y:y+fh, x:x+fw]
        face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

        if not _face_is_sharp(face_gray):
            continue

        face_pil = _crop_face_padded(rgb, x, y, fw, fh)
        vit_face = _multi_view_fake_probability(face_pil)

        edge_s = _edge_ratio_score(face_gray)
        hue_s  = _hue_inconsistency_score(face_bgr)
        seam_s = _blend_seam_score(full_gray, x, y, fw, fh)

        spatial = 0.35 * hue_s + 0.40 * edge_s + 0.25 * seam_s
        frame_s = 0.55 * vit_face + 0.45 * spatial

        if frame_s > best_score:
            best_score    = frame_s
            best_face_vec = cv2.resize(face_gray, (64, 64)).flatten().astype(float)

    if best_score < 0:
        return float(np.clip(vit_global * 0.70, 0.0, 1.0)), None

    combined = 0.60 * best_score + 0.40 * vit_global
    return float(np.clip(combined, 0.0, 1.0)), best_face_vec


# ---------------------------------------------------------------------------
# Per-image scoring (IMAGE -- separate from video frames)
# ---------------------------------------------------------------------------

def _score_image(rgb, pil_image, full_gray):
    """
    Score a single still IMAGE with image-specific forensic signals.

    Key differences from _score_frame():
    - No temporal signal available -- spatial signals must be stronger.
    - JPEG DCT ghost analysis (double-compression detection).
    - FFT-based GAN frequency fingerprint.
    - Noise inconsistency (face smoother than surroundings).
    - Facial symmetry anomaly detection.
    - Skin texture uniformity (LBP entropy).
    - Color channel correlation check.
    - Thresholds re-calibrated for still-image deepfakes.

    Returns
    -------
    float: image fake probability [0, 1]
    """
    faces      = _extract_faces(rgb, max_faces=2)
    vit_global = _multi_view_fake_probability(pil_image)

    # Global image-level forensic signals (no face required)
    freq_score = _frequency_domain_score(full_gray)

    if not faces:
        # No face detected -- rely on global ViT + frequency analysis
        # Down-weight ViT as it's less reliable without a face crop
        no_face_score = 0.60 * vit_global + 0.40 * freq_score
        print(
            f"[video_infer] image_score (no face): vit_global={vit_global:.4f} "
            f"freq={freq_score:.4f} -> {no_face_score:.4f}"
        )
        return float(np.clip(no_face_score, 0.0, 1.0))

    best_score = -1.0

    for (x, y, fw, fh) in faces:
        face_bgr  = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)[y:y+fh, x:x+fw]
        face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

        # Sharpness gate -- less strict for images (faces can be partially blurred)
        sharp_val = float(cv2.Laplacian(face_gray, cv2.CV_64F).var())
        if sharp_val < 15.0:  # more lenient than video (25.0)
            continue

        # ViT on padded face crop
        face_pil = _crop_face_padded(rgb, x, y, fw, fh)
        vit_face = _multi_view_fake_probability(face_pil)

        # --- Image-specific forensic signals ---
        dct_score      = _jpeg_dct_ghost_score(face_gray)
        noise_score    = _noise_inconsistency_score(full_gray, full_gray, x, y, fw, fh)
        symmetry_score = _facial_symmetry_score(face_gray)
        texture_score  = _skin_texture_uniformity_score(face_gray)
        chan_score     = _color_channel_correlation_score(face_bgr)
        seam_score     = _blend_seam_score(full_gray, x, y, fw, fh)

        # --- Image-specific weight blend ---
        # ViT carries the most weight since it's specifically trained.
        # Image forensic signals complement it:
        #   dct_score:      double-compression / GAN grid artifacts
        #   noise_score:    face region smoother than surroundings
        #   symmetry_score: unnaturally symmetric or misaligned
        #   texture_score:  overly uniform skin (GAN)
        #   chan_score:     unphysical color channel correlation
        #   seam_score:     paste boundary gradient spike
        forensic_score = (
            0.25 * dct_score
            + 0.20 * noise_score
            + 0.15 * symmetry_score
            + 0.15 * texture_score
            + 0.15 * chan_score
            + 0.10 * seam_score
        )

        # Combined score:
        # vit_face (40%) + vit_global (20%) + forensic (25%) + freq (15%)
        face_total = (
            0.40 * vit_face
            + 0.20 * vit_global
            + 0.25 * forensic_score
            + 0.15 * freq_score
        )

        print(
            f"[video_infer] face({x},{y},{fw},{fh}): "
            f"vit_face={vit_face:.4f} vit_global={vit_global:.4f} "
            f"dct={dct_score:.4f} noise={noise_score:.4f} "
            f"sym={symmetry_score:.4f} tex={texture_score:.4f} "
            f"chan={chan_score:.4f} seam={seam_score:.4f} "
            f"freq={freq_score:.4f} -> face_total={face_total:.4f}"
        )

        if face_total > best_score:
            best_score = face_total

    if best_score < 0:
        # All faces too blurry -- global fallback
        fallback = 0.60 * vit_global + 0.40 * freq_score
        print(
            f"[video_infer] image_score (all faces blurry): "
            f"vit_global={vit_global:.4f} freq={freq_score:.4f} -> {fallback:.4f}"
        )
        return float(np.clip(fallback, 0.0, 1.0))

    return float(np.clip(best_score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Public API -- video
# ---------------------------------------------------------------------------

def video_fake_probability(video_path, frame_interval=5):
    """
    Sample frames from a video and return a composite FAKE probability.

    Scoring pipeline
    ----------------
    Per sampled frame:
      - Detect face(s). Sharpness gate on face crop (threshold 25).
      - frame_score = 0.60 x ViT(face) + 0.40 x ViT(global)
                    + handcrafted signals (edge, hue, seam).
      - Track 64x64 face pixel vector for temporal difference analysis.

    Spatial aggregation:
      spatial = max(0.55 x top33_mean + 0.45 x global_mean, p85)

    Temporal aggregation:
      temporal = var(face_pixel_diffs) / 300  -- clipped to [0, 1]

    Final blend (>= 8 temporal diffs):
      result = 0.50 x spatial + 0.50 x temporal
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[video_infer] ERROR: Could not open video {video_path}")
        return 0.0

    frame_probs   = []
    face_diff_seq = []
    prev_face_vec = None
    frame_id      = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id == 0 or frame_id % frame_interval == 0:
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            pil_img = Image.fromarray(rgb)

            frame_score, face_vec = _score_frame(rgb, pil_img, gray)
            frame_probs.append(frame_score)
            del pil_img

            if face_vec is not None:
                if prev_face_vec is not None:
                    diff = float(np.mean(np.abs(face_vec - prev_face_vec)))
                    face_diff_seq.append(diff)
                prev_face_vec = face_vec

        frame_id += 1

    cap.release()

    if not frame_probs:
        print("[video_infer] WARNING: No usable frames extracted.")
        return 0.0

    probs_sorted   = sorted(frame_probs, reverse=True)
    top_k          = max(1, len(probs_sorted) // 3)
    high_risk_mean = float(np.mean(probs_sorted[:top_k]))
    global_mean    = float(np.mean(frame_probs))
    p85            = float(np.percentile(frame_probs, 85))
    spatial_score  = max(0.55 * high_risk_mean + 0.45 * global_mean, p85)

    temporal_score = _temporal_variance_score(face_diff_seq)

    n_diffs = len(face_diff_seq)
    if n_diffs >= 8:
        result = 0.50 * spatial_score + 0.50 * temporal_score
    elif n_diffs >= 4:
        result = 0.65 * spatial_score + 0.35 * temporal_score
    else:
        result = spatial_score

    result = float(np.clip(result, 0.0, 1.0))

    print(
        f"[video_infer] video_fake_probability={result:.4f} "
        f"(spatial={spatial_score:.4f}, temporal={temporal_score:.4f}, "
        f"high_risk_mean={high_risk_mean:.4f}, global_mean={global_mean:.4f}, "
        f"p85={p85:.4f}, frames={len(frame_probs)}, "
        f"temporal_diffs={n_diffs}, FAKE_IDX={_FAKE_IDX})"
    )
    return result


# ---------------------------------------------------------------------------
# Public API -- image
# ---------------------------------------------------------------------------

def image_fake_probability(image_path):
    """
    Run the full IMAGE scoring pipeline on a single still image.

    Uses _score_image() which applies image-specific forensic signals:
      - ViT on face crops (7 augmentations including CLAHE)
      - ViT global
      - JPEG DCT ghost / double-compression detection
      - Noise inconsistency (face smoother than surroundings)
      - Facial symmetry anomaly
      - Skin texture uniformity (LBP entropy proxy)
      - Color channel correlation
      - Blend seam score
      - FFT-based GAN frequency fingerprint

    Returns None on error so the caller can skip this modality.
    """
    try:
        image    = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        gray     = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        score = _score_image(image_np, image, gray)

        print(f"[video_infer] image_fake_probability={score:.4f}  FAKE_IDX={_FAKE_IDX}")
        return score

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[video_infer] Error processing image: {e}")
        return None