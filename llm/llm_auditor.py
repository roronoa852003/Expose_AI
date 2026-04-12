import requests
import json

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "meta-llama-3.1-8b-instruct"


SYSTEM_PROMPT = """
You are a friendly AI helper for a deepfake detection system called EXPOSE AI.

You explain the system's decisions in a fun, simple, and lenient way so that a small kid can easily understand it.

Your rules:
- Use simple words like "pictures" (instead of visual tensor), "sounds" (instead of spectral tensor), and "hidden clues" (instead of pipeline metadata).
- Explain numbers like percentages in a very easy-to-understand way (e.g., "We found enough weird things in the picture (72%) to know something is up!").
- Do not use complex, technical, or scary words. Keep it light, fun, and easy!
- Do not introduce new evidence beyond what is given.
- Be very encouraging and kind.

Respond ONLY in JSON using this exact format:
{
  "consistency": "CONSISTENT or INCONSISTENT",
  "confidence_level": "HIGH or MEDIUM or LOW",
  "explanation": "A 2-4 sentence fun and simple explanation for a kid telling them why the video is real or fake based on the clues.",
  "warnings": ["optional list of simple system warnings"]
}
"""


def audit_decision(
    video_prob,
    audio_prob,
    meta_prob,
    final_label,
    detected_type,
    override_triggered
):
    video_str = f"{video_prob:.2f}" if video_prob is not None else "N/A (no visual track)"
    audio_str = f"{audio_prob:.2f}" if audio_prob is not None else "N/A (no audio track)"
    meta_str = f"{meta_prob:.2f}" if meta_prob is not None else "N/A (no metadata)"
    override_str = "YES — safeguard override triggered" if override_triggered else "NO — soft fusion applied"

    weights = {"video": 0.4, "audio": 0.4, "meta": 0.2}
    score = sum((prob * weights[k] for k, prob in zip(["video", "audio", "meta"], [video_prob, audio_prob, meta_prob]) if prob is not None))
    total = sum((weights[k] for k, prob in zip(["video", "audio", "meta"], [video_prob, audio_prob, meta_prob]) if prob is not None))
    final_score = (score / total) if total > 0 else 0.0

    user_prompt = f"""
Forensic detection summary for analyst review:

VISUAL TENSOR:           {video_str}  (threshold: 0.50 for hard override)
AUDIO / SPECTRAL TENSOR: {audio_str}  (threshold: 0.60 for hard override)
PIPELINE FORENSICS:      {meta_str}  (threshold: 0.85 for isolated override)

FUSION OUTPUT:
  Final score:     {final_score:.2f}
  Label:           {final_label}
  Subtype:         {detected_type}
  Override status: {override_str}

Generate a forensic intelligence audit narrative for the above.
"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2
    }

    try:
        response = requests.post(LM_STUDIO_URL, json=payload, timeout=10)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        # Strip markdown code fences if model wraps response
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content.strip())
    except Exception:
        # ── Fallback: structured simulation so the UI always renders ──
        vp_str = f"{round(video_prob * 100)}%" if video_prob is not None else "N/A"
        ap_str = f"{round(audio_prob * 100)}%" if audio_prob is not None else "N/A"
        mp_str = f"{round(meta_prob * 100)}%" if meta_prob is not None else "N/A"

        if final_label == "DEEPFAKE":
            drivers = []
            if video_prob is not None and video_prob >= 0.50:
                drivers.append(f"we saw enough mixed-up pixels in the picture ({vp_str})")
            if audio_prob is not None and audio_prob >= 0.60:
                drivers.append(f"the sounds and voices seemed a bit robotic ({ap_str})")
            if meta_prob is not None and meta_prob >= 0.85:
                drivers.append(f"we found hidden computer clues in how it was made ({mp_str})")
            driver_str = " and ".join(drivers) if drivers else "all our little clues added up together"
            explanation = (
                f"Our smart computer brain looked at everything and decided this media is a FAKE! "
                f"The biggest clue was that {driver_str}. "
                f"It's a {detected_type.replace('_', ' ')} fake! "
                f"We checked the pictures ({vp_str}), sounds ({ap_str}), and hidden clues ({mp_str}) to be super sure!"
            )
            consistency = "CONSISTENT"
        else:
            explanation = (
                f"Our smart computer brain looked at everything and decided this media is completely REAL! "
                f"The pictures looked perfectly normal ({vp_str}), the sounds were natural ({ap_str}), "
                f"and we didn't find any hidden computer tricks ({mp_str}). "
                f"Since all our checks came back clear, we are very happy to say this is real!"
            )
            consistency = "INCONSISTENT"  # no inconsistency between modalities flagging vs result

        return {
            "consistency": consistency,
            "confidence_level": "HIGH",
            "explanation": explanation,
            "warnings": [
                "SYSTEM OFFLINE: Local LM Studio on port 1234 unreachable.",
                "FALLBACK MATRIX ENGAGED: Simulated audit generated for demonstration."
            ]
        }