import argparse
import json
import os
import platform
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
_FACE_APP: Optional[FaceAnalysis] = None
_FACE_PROVIDER: str = "unknown"

DEFAULT_MODEL_NAME = "buffalo_l"
DEFAULT_DET_SIZE = (640, 640)
MODEL_NAME_ENV = "INSIGHTFACE_MODEL"
DET_SIZE_ENV = "INSIGHTFACE_DET_SIZE"
PROVIDERS_ENV = "INSIGHTFACE_PROVIDERS"


def parse_det_size(raw: str) -> Tuple[int, int]:
    value = raw.strip()
    if "," in value:
        left, right = value.split(",", maxsplit=1)
        width = max(64, int(left.strip()))
        height = max(64, int(right.strip()))
        return width, height
    if "x" in value.lower():
        left, right = value.lower().split("x", maxsplit=1)
        width = max(64, int(left.strip()))
        height = max(64, int(right.strip()))
        return width, height
    size = max(64, int(value))
    return size, size


def configured_model_name() -> str:
    value = os.getenv(MODEL_NAME_ENV, "").strip()
    if value:
        return value
    return DEFAULT_MODEL_NAME


def configured_det_size() -> Tuple[int, int]:
    raw = os.getenv(DET_SIZE_ENV, "").strip()
    if not raw:
        return DEFAULT_DET_SIZE
    try:
        return parse_det_size(raw)
    except Exception:
        return DEFAULT_DET_SIZE


def available_onnx_providers() -> List[str]:
    try:
        import onnxruntime as ort  # type: ignore

        providers = ort.get_available_providers()
        return [provider for provider in providers if isinstance(provider, str)]
    except Exception:
        return []


def requested_provider_order() -> List[str]:
    env_value = os.getenv(PROVIDERS_ENV, "").strip()
    if env_value:
        return [provider.strip() for provider in env_value.split(",") if provider.strip()]
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "darwin" and machine in {"arm64", "aarch64"}:
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return [
        "CUDAExecutionProvider",
        "ROCMExecutionProvider",
        "OpenVINOExecutionProvider",
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]


def provider_attempts() -> List[List[str]]:
    available = set(available_onnx_providers())
    preferred = requested_provider_order()
    attempts: List[List[str]] = []
    for provider in preferred:
        if available and provider not in available:
            continue
        if provider == "CPUExecutionProvider":
            attempts.append(["CPUExecutionProvider"])
        else:
            fallback = ["CPUExecutionProvider"]
            if available and "CPUExecutionProvider" not in available:
                fallback = []
            attempts.append([provider, *fallback])
    if not attempts:
        attempts.append(["CPUExecutionProvider"])
    return attempts


def active_provider(app: FaceAnalysis) -> str:
    try:
        detector = getattr(app, "det_model", None)
        session = getattr(detector, "session", None)
        providers = session.get_providers() if session is not None else []
        if isinstance(providers, list) and providers:
            return str(providers[0])
    except Exception:
        pass
    return "unknown"


def get_face_app() -> FaceAnalysis:
    global _FACE_APP, _FACE_PROVIDER
    if _FACE_APP is None:
        model_name = configured_model_name()
        det_size = configured_det_size()
        errors: List[str] = []
        for providers in provider_attempts():
            try:
                app = FaceAnalysis(name=model_name, providers=providers)
                app.prepare(ctx_id=0, det_size=det_size)
                _FACE_PROVIDER = active_provider(app)
                print(
                    f"Face runtime initialized: model={model_name}, det_size={det_size[0]}x{det_size[1]}, "
                    f"providers={providers}, active={_FACE_PROVIDER}"
                )
                _FACE_APP = app
                break
            except Exception as error:
                errors.append(f"{providers}: {error}")
        if _FACE_APP is None:
            raise RuntimeError("Could not initialize InsightFace runtime.\n" + "\n".join(errors))
    return _FACE_APP


def get_face_provider() -> str:
    _ = get_face_app()
    return _FACE_PROVIDER


def load_image(path: Path) -> Optional[np.ndarray]:
    try:
        image_bytes = np.fromfile(str(path), dtype=np.uint8)
    except Exception:
        return None
    if image_bytes.size == 0:
        return None
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return None
    return image


def normalize_embedding(vector: np.ndarray) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return None
    return vector / norm


def get_face_embedding(face: object) -> Optional[np.ndarray]:
    normed_embedding = getattr(face, "normed_embedding", None)
    if isinstance(normed_embedding, np.ndarray) and normed_embedding.size > 0:
        return normed_embedding.astype(np.float32)
    embedding = getattr(face, "embedding", None)
    if isinstance(embedding, np.ndarray) and embedding.size > 0:
        return normalize_embedding(embedding.astype(np.float32))
    return None


def get_face_bbox(face: object) -> Optional[np.ndarray]:
    bbox = getattr(face, "bbox", None)
    if isinstance(bbox, np.ndarray) and bbox.size >= 4:
        return bbox.astype(np.float32)
    return None


def face_area(face: object) -> float:
    bbox = get_face_bbox(face)
    if bbox is None:
        return -1.0
    width = max(0.0, float(bbox[2] - bbox[0]))
    height = max(0.0, float(bbox[3] - bbox[1]))
    return width * height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter images and keep only photos containing known kids.")
    parser.add_argument("--manifest", required=True, help="Path to JSON file with downloaded image entries.")
    parser.add_argument("--reference-dir", required=True, help="Directory with one folder per kid containing reference images.")
    parser.add_argument("--output-dir", required=True, help="Directory where matched images are stored under kid folders.")
    parser.add_argument("--unmatched-dir", help="If set, move unmatched photos here for false-negative review.")
    parser.add_argument("--report-path", required=True, help="Where to write a JSON report.")
    parser.add_argument("--tolerance", type=float, default=0.45, help="Minimum face similarity (higher = stricter).")
    parser.add_argument("--min-face-px", type=int, default=80, help="Ignore detected faces smaller than this pixel size.")
    parser.add_argument("--min-votes", type=int, default=2, help="Require at least this many reference matches for a kid.")
    parser.add_argument("--distance-margin", type=float, default=0.06, help="Required similarity gap between best and second-best kid.")
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
    app = get_face_app()
    reference_images = list_reference_images(reference_root)
    known_encodings: Dict[str, List[List[float]]] = {}

    for kid_name, image_paths in reference_images.items():
        kid_vectors: List[List[float]] = []
        for image_path in image_paths:
            try:
                image_data = load_image(image_path)
                if image_data is None:
                    continue
                faces = app.get(image_data)
                if not faces:
                    continue
                largest_face = max(faces, key=face_area)
                embedding = get_face_embedding(largest_face)
                if embedding is not None:
                    kid_vectors.append(embedding.tolist())
            except Exception:
                continue
        if kid_vectors:
            known_encodings[kid_name] = kid_vectors

    return known_encodings


def build_kid_embedding_matrices(kid_encodings: Dict[str, List[List[float]]]) -> Dict[str, np.ndarray]:
    matrices: Dict[str, np.ndarray] = {}
    for kid_name, vectors in kid_encodings.items():
        if not vectors:
            continue
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.size == 0:
            continue
        matrices[kid_name] = matrix
    return matrices


def load_manifest(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def pick_best_kid_for_face(
    face_encoding: np.ndarray,
    kid_matrices: Dict[str, np.ndarray],
    tolerance: float,
    min_votes: int,
    distance_margin: float,
) -> str:
    best_by_kid: Dict[str, float] = {}
    votes_by_kid: Dict[str, int] = {}

    for kid_name, candidate_vectors in kid_matrices.items():
        if candidate_vectors.size == 0:
            continue
        similarities = candidate_vectors @ face_encoding
        if similarities.size == 0:
            continue
        best_similarity = float(np.max(similarities))
        votes = int(np.sum(similarities >= tolerance))
        best_by_kid[kid_name] = best_similarity
        votes_by_kid[kid_name] = votes

    if not best_by_kid:
        return ""

    ranked = sorted(best_by_kid.items(), key=lambda item: item[1], reverse=True)
    winner, winner_similarity = ranked[0]
    runner_up_similarity = ranked[1][1] if len(ranked) > 1 else -1.0
    winner_votes = votes_by_kid.get(winner, 0)

    if winner_similarity < tolerance:
        return ""
    if winner_votes < min_votes:
        return ""
    if (winner_similarity - runner_up_similarity) < distance_margin:
        return ""
    return winner


def match_image(
    path: Path,
    kid_matrices: Dict[str, np.ndarray],
    tolerance: float,
    min_face_px: int,
    min_votes: int,
    distance_margin: float,
) -> Tuple[List[str], str]:
    app = get_face_app()
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return [], "unsupported-format"
    if not path.exists():
        return [], "missing-file"

    image_data = load_image(path)
    if image_data is None:
        return [], "load-failed"

    try:
        faces = app.get(image_data)
    except Exception:
        return [], "load-failed"

    if not faces:
        return [], "no-face"

    eligible_faces: List[object] = []
    for face in faces:
        bbox = get_face_bbox(face)
        if bbox is None:
            continue
        left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
        width = max(0.0, float(right - left))
        height = max(0.0, float(bottom - top))
        if min(width, height) >= min_face_px:
            eligible_faces.append(face)

    if not eligible_faces:
        return [], "no-eligible-face"

    face_encodings: List[np.ndarray] = []
    for face in eligible_faces:
        embedding = get_face_embedding(face)
        if embedding is not None:
            face_encodings.append(embedding)
    if not face_encodings:
        return [], "no-encoding"

    matched_kids = set()
    for face_encoding in face_encodings:
        winner = pick_best_kid_for_face(
            face_encoding=face_encoding,
            kid_matrices=kid_matrices,
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
    kid_matrices = build_kid_embedding_matrices(known_kids)

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
            kid_matrices=kid_matrices,
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
