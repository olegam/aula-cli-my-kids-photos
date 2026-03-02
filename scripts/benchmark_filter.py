import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from filter_kids import (
    IMAGE_EXTENSIONS,
    build_kid_embedding_matrices,
    build_kid_encodings,
    get_face_provider,
    match_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark face filtering throughput and latency.")
    parser.add_argument("--reference-dir", required=True, help="Reference kid folders.")
    parser.add_argument("--images-dir", required=True, help="Images to benchmark.")
    parser.add_argument("--report-path", required=True, help="Where to write benchmark report.")
    parser.add_argument("--tolerance", type=float, default=0.45, help="Minimum face similarity.")
    parser.add_argument("--min-face-px", type=int, default=80, help="Ignore faces smaller than this size.")
    parser.add_argument("--min-votes", type=int, default=2, help="Required reference matches.")
    parser.add_argument("--distance-margin", type=float, default=0.03, help="Best-vs-runner-up gap requirement.")
    parser.add_argument("--limit", type=int, default=0, help="Optional image cap for faster benchmark cycles.")
    return parser.parse_args()


def list_images(root: Path, limit: int) -> List[Path]:
    if not root.exists():
        return []
    images = sorted([path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS])
    if limit > 0:
        return images[:limit]
    return images


def percentile(values_ms: List[float], pct: float) -> float:
    if not values_ms:
        return 0.0
    return float(np.percentile(np.asarray(values_ms, dtype=np.float64), pct))


def main() -> None:
    args = parse_args()
    reference_root = Path(args.reference_dir)
    images_root = Path(args.images_dir)
    report_path = Path(args.report_path)

    known_kids = build_kid_encodings(reference_root)
    if len(known_kids) == 0:
        raise RuntimeError("No kid encodings found in reference-dir.")
    kid_matrices = build_kid_embedding_matrices(known_kids)

    images = list_images(images_root, max(0, int(args.limit)))
    timings_ms: List[float] = []
    status_counts: Dict[str, int] = {}
    matched_images = 0
    processed_images = 0

    run_start = time.perf_counter()
    for image_path in images:
        started = time.perf_counter()
        matched_kids, status = match_image(
            path=image_path,
            kid_matrices=kid_matrices,
            tolerance=float(args.tolerance),
            min_face_px=int(args.min_face_px),
            min_votes=int(args.min_votes),
            distance_margin=float(args.distance_margin),
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        timings_ms.append(elapsed_ms)
        status_counts[status] = status_counts.get(status, 0) + 1
        processed_images += 1
        if len(matched_kids) > 0:
            matched_images += 1
    total_seconds = max(0.0, time.perf_counter() - run_start)

    report = {
        "totalImages": len(images),
        "processedImages": processed_images,
        "matchedImages": matched_images,
        "provider": get_face_provider(),
        "imagesPerSecond": (processed_images / total_seconds) if total_seconds > 0 else 0.0,
        "p50Ms": percentile(timings_ms, 50),
        "p95Ms": percentile(timings_ms, 95),
        "totalTimeSeconds": total_seconds,
        "statusCounts": status_counts,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)


if __name__ == "__main__":
    main()
