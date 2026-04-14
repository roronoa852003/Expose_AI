import argparse
import csv
import json
from pathlib import Path

from inference.video_infer import image_fake_probability
from utils.metadata_infer import metadata_fake_probability


def _parse_label(raw: str) -> int:
    text = str(raw).strip().lower()
    if text in {"1", "fake", "deepfake", "manipulated", "forged"}:
        return 1
    if text in {"0", "real", "authentic", "clean"}:
        return 0
    raise ValueError(f"Unsupported label '{raw}'. Use 0/1 or real/fake values.")


def _safe_div(num: float, den: float) -> float:
    return (num / den) if den else 0.0


def _metrics(rows, image_threshold: float, meta_threshold: float = 0.50):
    tp = fp = tn = fn = 0
    for row in rows:
        pred_fake = (row["image_prob"] >= image_threshold) or (
            row["image_prob"] >= 0.58 and row["meta_prob"] >= meta_threshold
        )
        actual_fake = row["label"] == 1
        if pred_fake and actual_fake:
            tp += 1
        elif pred_fake and not actual_fake:
            fp += 1
        elif not pred_fake and not actual_fake:
            tn += 1
        else:
            fn += 1

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)

    return {
        "image_threshold": image_threshold,
        "meta_threshold": meta_threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }


def _iter_thresholds(start: float, end: float, step: float):
    current = start
    while current <= end + 1e-9:
        yield round(current, 4)
        current += step


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate image-only deepfake performance and sweep thresholds."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="CSV with columns: path,label (label can be 0/1 or real/fake).",
    )
    parser.add_argument("--start-threshold", type=float, default=0.50)
    parser.add_argument("--end-threshold", type=float, default=0.85)
    parser.add_argument("--step", type=float, default=0.02)
    parser.add_argument(
        "--output-json",
        default="image_eval_results.json",
        help="Path to write threshold sweep + best config JSON.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            path = Path(row["path"]).expanduser()
            if not path.exists():
                print(f"[skip] row={idx} missing path: {path}")
                continue

            try:
                label = _parse_label(row["label"])
            except Exception as exc:
                print(f"[skip] row={idx} invalid label '{row.get('label')}': {exc}")
                continue

            image_prob = image_fake_probability(str(path))
            if image_prob is None:
                print(f"[skip] row={idx} image inference failed: {path}")
                continue
            meta_prob = metadata_fake_probability(str(path), media_type="image")

            rows.append(
                {
                    "path": str(path),
                    "label": label,
                    "image_prob": float(image_prob),
                    "meta_prob": float(meta_prob),
                }
            )

    if not rows:
        raise RuntimeError("No valid rows were evaluated. Check CSV and image paths.")

    evaluations = [
        _metrics(rows, image_threshold=th)
        for th in _iter_thresholds(args.start_threshold, args.end_threshold, args.step)
    ]
    best = max(evaluations, key=lambda m: (m["f1"], m["recall"], m["precision"]))

    payload = {
        "samples_evaluated": len(rows),
        "best": best,
        "evaluations": evaluations,
    }

    with Path(args.output_json).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Evaluated {len(rows)} images.")
    print(
        f"Best threshold={best['image_threshold']:.2f} "
        f"F1={best['f1']:.4f} Precision={best['precision']:.4f} Recall={best['recall']:.4f}"
    )
    print(f"Saved: {args.output_json}")


if __name__ == "__main__":
    main()
