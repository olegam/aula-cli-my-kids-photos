import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import face_recognition


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter images and keep only photos containing known kids.")
    parser.add_argument("--manifest", required=True, help="Path to JSON file with downloaded image entries.")
    parser.add_argument("--reference-dir", required=True, help="Directory with one folder per kid containing reference images.")
    parser.add_argument("--output-dir", required=True, help="Directory where matched images are stored under kid folders.")
    parser.add_argument("--unmatched-dir", help="If set, move unmatched photos here for false-negative review.")
    parser.add_argument("--report-path", required=True, help="Where to write a JSON report.")
    parser.add_argument("--tolerance", type=float, default=0.48, help="Face match tolerance (smaller = stricter).")
    parser.add_argument("--min-face-px", type=int, default=70, help="Ignore detected faces smaller than this pixel size.")
    parser.add_argument("--min-votes", type=int, default=1, help="Require at least this many reference matches for a kid.")
    parser.add_argument("--distance-margin", type=float, default=0.02, help="Required gap between best and second-best kid distance.")
    parser.add_argument("--delete-unmatched", action="store_true", help="Delete non-matching images after analysis.")
    return parser.parse_args()


def list_reference_images(reference_root: Path) -> Dict[str, List[Path]]:
    kids: Dict[str, List[Path]] = {}
    if not reference_root.exists():
        return kids

    for child in sorted(reference_root.iterdir()):
        if not child.is_dir():
            continue
        images = [path for path in child.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
        if images:
            kids[child.name] = images
    return kids


def build_kid_encodings(reference_root: Path) -> Dict[str, List[List[float]]]:
    reference_images = list_reference_images(reference_root)
    known_encodings: Dict[str, List[List[float]]] = {}

    for kid_name, image_paths in reference_images.items():
        kid_vectors: List[List[float]] = []
        for image_path in image_paths:
            try:
                image_data = face_recognition.load_image_file(str(image_path))
                encodings = face_recognition.face_encodings(image_data)
                if encodings:
                    kid_vectors.append(encodings[0].tolist())
            except Exception:
                continue
        if kid_vectors:
            known_encodings[kid_name] = kid_vectors

    return known_encodings


def load_manifest(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def pick_best_kid_for_face(
    face_encoding: List[float],
    kid_encodings: Dict[str, List[List[float]]],
    tolerance: float,
    min_votes: int,
    distance_margin: float,
) -> str:
    best_by_kid: Dict[str, float] = {}
    votes_by_kid: Dict[str, int] = {}

    for kid_name, vectors in kid_encodings.items():
        if not vectors:
            continue
        distances = face_recognition.face_distance(vectors, face_encoding)
        if len(distances) == 0:
            continue
        best_distance = float(min(distances))
        votes = sum(1 for value in distances if float(value) <= tolerance)
        best_by_kid[kid_name] = best_distance
        votes_by_kid[kid_name] = votes

    if not best_by_kid:
        return ""

    ranked = sorted(best_by_kid.items(), key=lambda item: item[1])
    winner, winner_distance = ranked[0]
    runner_up_distance = ranked[1][1] if len(ranked) > 1 else 1.0
    winner_votes = votes_by_kid.get(winner, 0)

    if winner_distance > tolerance:
        return ""
    if winner_votes < min_votes:
        return ""
    if (runner_up_distance - winner_distance) < distance_margin:
        return ""
    return winner


def match_image(
    path: Path,
    kid_encodings: Dict[str, List[List[float]]],
    tolerance: float,
    min_face_px: int,
    min_votes: int,
    distance_margin: float,
) -> Tuple[List[str], str]:
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return [], "unsupported-format"
    if not path.exists():
        return [], "missing-file"

    try:
        image_data = face_recognition.load_image_file(str(path))
        face_locations = face_recognition.face_locations(image_data, model="hog")
    except Exception:
        return [], "load-failed"

    if not face_locations:
        return [], "no-face"

    eligible_locations = []
    for top, right, bottom, left in face_locations:
        width = max(0, right - left)
        height = max(0, bottom - top)
        if min(width, height) >= min_face_px:
            eligible_locations.append((top, right, bottom, left))

    if not eligible_locations:
        return [], "no-eligible-face"

    face_encodings = face_recognition.face_encodings(image_data, known_face_locations=eligible_locations)
    if not face_encodings:
        return [], "no-encoding"

    matched_kids = set()
    for face_encoding in face_encodings:
        winner = pick_best_kid_for_face(
            face_encoding=face_encoding,
            kid_encodings=kid_encodings,
            tolerance=tolerance,
            min_votes=min_votes,
            distance_margin=distance_margin,
        )
        if winner:
            matched_kids.add(winner)

    return sorted(matched_kids), "ok"


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    reference_root = Path(args.reference_dir)
    output_root = Path(args.output_dir)
    unmatched_root = Path(args.unmatched_dir) if args.unmatched_dir else None
    report_path = Path(args.report_path)
    output_root.mkdir(parents=True, exist_ok=True)
    if unmatched_root:
        unmatched_root.mkdir(parents=True, exist_ok=True)

    downloads = load_manifest(manifest_path)
    known_kids = build_kid_encodings(reference_root)

    report = {
        "total": len(downloads),
        "kept": 0,
        "removed": 0,
        "errors": 0,
        "keptByKid": {},
        "entries": [],
    }

    for item in downloads:
        output_path_raw = item.get("outputPath")
        if not isinstance(output_path_raw, str):
            report["errors"] += 1
            report["entries"].append({"status": "invalid-manifest-item", "item": item})
            continue

        image_path = Path(output_path_raw)
        matched_kids, status = match_image(
            path=image_path,
            kid_encodings=known_kids,
            tolerance=args.tolerance,
            min_face_px=args.min_face_px,
            min_votes=args.min_votes,
            distance_margin=args.distance_margin,
        )

        should_keep = len(matched_kids) > 0
        deleted = False
        stored_paths: List[str] = []

        if not should_keep:
            try:
                if image_path.exists():
                    if unmatched_root:
                        unmatched_path = unmatched_root / image_path.name
                        if unmatched_path.exists():
                            stem = unmatched_path.stem
                            suffix = unmatched_path.suffix
                            index = 2
                            while (unmatched_root / f"{stem}_{index}{suffix}").exists():
                                index += 1
                            unmatched_path = unmatched_root / f"{stem}_{index}{suffix}"
                        os.replace(image_path, unmatched_path)
                    elif args.delete_unmatched:
                        os.remove(image_path)
                        deleted = True
            except Exception:
                report["errors"] += 1

        if should_keep:
            if image_path.exists():
                primary_path = output_root / matched_kids[0] / image_path.name
                primary_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(image_path, primary_path)
                stored_paths.append(str(primary_path))
                for kid_name in matched_kids[1:]:
                    kid_path = output_root / kid_name / image_path.name
                    kid_path.parent.mkdir(parents=True, exist_ok=True)
                    if not kid_path.exists():
                        shutil.copy2(primary_path, kid_path)
                    stored_paths.append(str(kid_path))
            else:
                report["errors"] += 1

            report["kept"] += 1
            for kid_name in matched_kids:
                report["keptByKid"][kid_name] = report["keptByKid"].get(kid_name, 0) + 1
        else:
            report["removed"] += 1

        if status not in {"ok", "no-face"}:
            report["errors"] += 1

        report["entries"].append(
            {
                "path": stored_paths[0] if stored_paths else str(image_path),
                "storedPaths": stored_paths,
                "matchedKids": matched_kids,
                "status": status,
                "deleted": deleted,
            }
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)


if __name__ == "__main__":
    main()
