import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from filter_kids import (
    IMAGE_EXTENSIONS,
    build_kid_embedding_matrices,
    build_kid_encodings,
    get_face_app,
    get_face_bbox,
    get_face_embedding,
    load_image,
    pick_best_kid_for_face,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune face-matching thresholds against labeled datasets.")
    parser.add_argument("--reference-dir", required=True, help="Reference kid folders.")
    parser.add_argument("--positive-dir", required=True, help="Folder with images that contain at least one target kid.")
    parser.add_argument("--negative-dir", required=True, help="Folder with images that contain none of the target kids.")
    parser.add_argument("--iterations", type=int, default=10, help="How many threshold sets to evaluate.")
    parser.add_argument("--report-path", required=True, help="Where to write tuning results JSON.")
    return parser.parse_args()


def list_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted([path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS])


def candidate_thresholds(limit: int) -> List[Dict[str, float]]:
    preset = [
        {"tolerance": 0.35, "minFacePx": 90, "minVotes": 2, "distanceMargin": 0.08},
        {"tolerance": 0.38, "minFacePx": 90, "minVotes": 2, "distanceMargin": 0.07},
        {"tolerance": 0.40, "minFacePx": 85, "minVotes": 2, "distanceMargin": 0.07},
        {"tolerance": 0.42, "minFacePx": 85, "minVotes": 2, "distanceMargin": 0.06},
        {"tolerance": 0.45, "minFacePx": 80, "minVotes": 2, "distanceMargin": 0.06},
        {"tolerance": 0.48, "minFacePx": 80, "minVotes": 2, "distanceMargin": 0.05},
        {"tolerance": 0.50, "minFacePx": 75, "minVotes": 1, "distanceMargin": 0.05},
        {"tolerance": 0.52, "minFacePx": 75, "minVotes": 1, "distanceMargin": 0.04},
        {"tolerance": 0.55, "minFacePx": 70, "minVotes": 1, "distanceMargin": 0.04},
        {"tolerance": 0.58, "minFacePx": 70, "minVotes": 1, "distanceMargin": 0.03},
    ]
    if limit <= len(preset):
        return preset[:limit]
    return preset


def extract_face_records(path: Path) -> List[Tuple[np.ndarray, int]]:
    app = get_face_app()
    image_data = load_image(path)
    if image_data is None:
        return []

    try:
        faces = app.get(image_data)
    except Exception:
        return []

    if not faces:
        return []

    records: List[Tuple[np.ndarray, int]] = []
    for face in faces:
        bbox = get_face_bbox(face)
        embedding = get_face_embedding(face)
        if bbox is None or embedding is None:
            continue
        left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
        width = max(0.0, float(right - left))
        height = max(0.0, float(bottom - top))
        records.append((embedding, int(min(width, height))))
    return records


def preload_face_records(images: List[Path]) -> Dict[str, List[Tuple[np.ndarray, int]]]:
    return {str(path): extract_face_records(path) for path in images}


def predicted_positive(
    records: List[Tuple[np.ndarray, int]],
    kid_matrices: Dict[str, np.ndarray],
    tolerance: float,
    min_face_px: int,
    min_votes: int,
    distance_margin: float,
) -> bool:
    for encoding, min_dim in records:
        if min_dim < min_face_px:
            continue
        winner = pick_best_kid_for_face(
            face_encoding=encoding,
            kid_matrices=kid_matrices,
            tolerance=tolerance,
            min_votes=min_votes,
            distance_margin=distance_margin,
        )
        if winner:
            return True
    return False


def evaluate(
    positives: List[Path],
    negatives: List[Path],
    positive_records: Dict[str, List[Tuple[np.ndarray, int]]],
    negative_records: Dict[str, List[Tuple[np.ndarray, int]]],
    kid_matrices: Dict[str, np.ndarray],
    tolerance: float,
    min_face_px: int,
    min_votes: int,
    distance_margin: float,
) -> Dict[str, float]:
    tp = tn = fp = fn = 0

    for image_path in positives:
        is_positive = predicted_positive(
            records=positive_records.get(str(image_path), []),
            kid_matrices=kid_matrices,
            tolerance=tolerance,
            min_face_px=min_face_px,
            min_votes=min_votes,
            distance_margin=distance_margin,
        )
        if is_positive:
            tp += 1
        else:
            fn += 1

    for image_path in negatives:
        is_positive = predicted_positive(
            records=negative_records.get(str(image_path), []),
            kid_matrices=kid_matrices,
            tolerance=tolerance,
            min_face_px=min_face_px,
            min_votes=min_votes,
            distance_margin=distance_margin,
        )
        if is_positive:
            fp += 1
        else:
            tn += 1

    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    total_errors = fp + fn
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": total,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
        "totalErrors": total_errors,
    }


def main() -> None:
    args = parse_args()
    reference_root = Path(args.reference_dir)
    positive_root = Path(args.positive_dir)
    negative_root = Path(args.negative_dir)
    report_path = Path(args.report_path)

    positives = list_images(positive_root)
    negatives = list_images(negative_root)
    known_kids = build_kid_encodings(reference_root)
    kid_matrices = build_kid_embedding_matrices(known_kids)
    runs = []

    if len(positives) == 0 or len(negatives) == 0:
        raise RuntimeError("Both positive-dir and negative-dir must contain images before tuning.")
    if len(known_kids) == 0:
        raise RuntimeError("No kid encodings found in reference-dir.")

    print("Precomputing face encodings for labeled datasets...")
    positive_records = preload_face_records(positives)
    negative_records = preload_face_records(negatives)

    candidates = candidate_thresholds(max(1, args.iterations))
    for index, candidate in enumerate(candidates, start=1):
        metrics = evaluate(
            positives=positives,
            negatives=negatives,
            positive_records=positive_records,
            negative_records=negative_records,
            kid_matrices=kid_matrices,
            tolerance=float(candidate["tolerance"]),
            min_face_px=int(candidate["minFacePx"]),
            min_votes=int(candidate["minVotes"]),
            distance_margin=float(candidate["distanceMargin"]),
        )
        row = {
            **candidate,
            **metrics,
        }
        runs.append(row)
        print(
            f"[{index}/{len(candidates)}] tol={candidate['tolerance']}, minFacePx={candidate['minFacePx']}, "
            f"minVotes={candidate['minVotes']}, margin={candidate['distanceMargin']} -> "
            f"errors={metrics['totalErrors']}, acc={metrics['accuracy']:.4f}, prec={metrics['precision']:.4f}, rec={metrics['recall']:.4f}"
        )

    ranked = sorted(
        runs,
        key=lambda row: (
            int(row["totalErrors"]),
            -float(row["accuracy"]),
            -float(row["f1"]),
            -float(row["recall"]),
        ),
    )
    best = ranked[0]
    output = {
        "summary": {
            "positiveCount": len(positives),
            "negativeCount": len(negatives),
            "runs": len(runs),
        },
        "best": best,
        "runs": runs,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)


if __name__ == "__main__":
    main()
