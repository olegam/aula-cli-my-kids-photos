import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import face_recognition


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter images and keep only photos containing known kids.")
    parser.add_argument("--manifest", required=True, help="Path to JSON file with downloaded image entries.")
    parser.add_argument("--reference-dir", required=True, help="Directory with one folder per kid containing reference images.")
    parser.add_argument("--report-path", required=True, help="Where to write a JSON report.")
    parser.add_argument("--tolerance", type=float, default=0.47, help="Face match tolerance (smaller = stricter).")
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


def match_image(path: Path, kid_encodings: Dict[str, List[List[float]]], tolerance: float) -> Tuple[List[str], str]:
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return [], "unsupported-format"
    if not path.exists():
        return [], "missing-file"

    try:
        image_data = face_recognition.load_image_file(str(path))
        face_encodings = face_recognition.face_encodings(image_data)
    except Exception:
        return [], "load-failed"

    if not face_encodings:
        return [], "no-face"

    matched_kids = set()
    for face_encoding in face_encodings:
        for kid_name, vectors in kid_encodings.items():
            matches = face_recognition.compare_faces(vectors, face_encoding, tolerance=tolerance)
            if any(matches):
                matched_kids.add(kid_name)

    return sorted(matched_kids), "ok"


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    reference_root = Path(args.reference_dir)
    report_path = Path(args.report_path)

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
        matched_kids, status = match_image(image_path, known_kids, args.tolerance)

        should_keep = len(matched_kids) > 0
        deleted = False

        if not should_keep and args.delete_unmatched:
            try:
                if image_path.exists():
                    os.remove(image_path)
                    deleted = True
            except Exception:
                report["errors"] += 1

        if should_keep:
            report["kept"] += 1
            for kid_name in matched_kids:
                report["keptByKid"][kid_name] = report["keptByKid"].get(kid_name, 0) + 1
        else:
            report["removed"] += 1

        if status not in {"ok", "no-face"}:
            report["errors"] += 1

        report["entries"].append(
            {
                "path": str(image_path),
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
