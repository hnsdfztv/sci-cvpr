import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from .core import (
        compute_eme,
        compute_loe,
        is_image_file,
        load_rgb,
        resize_rgb,
        rgb_to_gray,
        try_compute_niqe,
    )
except ImportError:  # pragma: no cover - script execution fallback
    from core import (  # type: ignore
        compute_eme,
        compute_loe,
        is_image_file,
        load_rgb,
        resize_rgb,
        rgb_to_gray,
        try_compute_niqe,
    )


EVALUATION_VERSION = "2026-03-31"


def build_stem_map(folder: Path) -> Tuple[Dict[str, Path], Dict[str, List[Path]]]:
    mapping: Dict[str, Path] = {}
    duplicates: Dict[str, List[Path]] = {}
    for path in sorted(folder.rglob("*")):
        if not (path.is_file() and is_image_file(path)):
            continue

        stem = path.stem
        if stem not in mapping:
            mapping[stem] = path
            continue

        duplicates.setdefault(stem, [mapping[stem]]).append(path)
    return mapping, duplicates


def safe_mean(values: Sequence[float]) -> float:
    valid = [value for value in values if not math.isnan(value)]
    return mean(valid) if valid else float("nan")


def preview_strings(values: Iterable[str], limit: int) -> List[str]:
    items = sorted(values)
    return items[:limit]


def preview_duplicate_paths(duplicates: Dict[str, List[Path]], limit: int) -> Dict[str, List[str]]:
    preview: Dict[str, List[str]] = {}
    for stem in sorted(duplicates)[:limit]:
        preview[stem] = [str(path) for path in duplicates[stem]]
    return preview


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def relative_to(base: Path, target: Optional[Path]) -> Optional[str]:
    if target is None:
        return None
    try:
        return str(target.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(target)


def compute_image_metrics(gray: Any, eme_block: int) -> Tuple[float, float, Optional[str], Optional[str]]:
    niqe, backend, warning = try_compute_niqe(gray)
    eme = compute_eme(gray, block_size=eme_block)
    return niqe, eme, backend, warning


def evaluate_pairs(
    input_map: Dict[str, Path],
    enh_map: Dict[str, Path],
    stems: Sequence[str],
    eme_block: int,
    loe_max_side: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    input_niqe_vals: List[float] = []
    input_eme_vals: List[float] = []
    niqe_vals: List[float] = []
    eme_vals: List[float] = []
    loe_vals: List[float] = []
    delta_niqe_vals: List[float] = []
    delta_eme_vals: List[float] = []
    niqe_backends = set()
    niqe_warnings = set()

    for stem in stems:
        input_path = input_map[stem]
        enhanced_path = enh_map[stem]

        input_rgb = load_rgb(input_path)
        enhanced_rgb = load_rgb(enhanced_path)

        if input_rgb.shape[:2] != enhanced_rgb.shape[:2]:
            enhanced_rgb = resize_rgb(enhanced_rgb, input_rgb.shape[:2])

        input_gray = rgb_to_gray(input_rgb)
        enhanced_gray = rgb_to_gray(enhanced_rgb)

        input_niqe, input_eme, input_backend, input_warn = compute_image_metrics(input_gray, eme_block)
        niqe, eme, niqe_backend, niqe_warn = compute_image_metrics(enhanced_gray, eme_block)
        loe = compute_loe(input_gray, enhanced_gray, max_side=loe_max_side)

        for backend in (input_backend, niqe_backend):
            if backend:
                niqe_backends.add(backend)
        for warning in (input_warn, niqe_warn):
            if warning:
                niqe_warnings.add(warning)

        delta_niqe = niqe - input_niqe if not (math.isnan(niqe) or math.isnan(input_niqe)) else float("nan")
        delta_eme = eme - input_eme if not (math.isnan(eme) or math.isnan(input_eme)) else float("nan")

        input_niqe_vals.append(input_niqe)
        input_eme_vals.append(input_eme)
        niqe_vals.append(niqe)
        eme_vals.append(eme)
        loe_vals.append(loe)
        delta_niqe_vals.append(delta_niqe)
        delta_eme_vals.append(delta_eme)

        rows.append(
            {
                "stem": stem,
                "pairing_mode": "paired",
                "input": str(input_path),
                "enhanced": str(enhanced_path),
                "input_niqe": input_niqe,
                "niqe": niqe,
                "input_eme": input_eme,
                "eme": eme,
                "delta_niqe": delta_niqe,
                "delta_eme": delta_eme,
                "loe": loe,
                "niqe_backend": niqe_backend or input_backend,
            }
        )

    return rows, {
        "mode": "paired",
        "matched_images": len(rows),
        "mean_input_niqe": safe_mean(input_niqe_vals),
        "mean_niqe": safe_mean(niqe_vals),
        "mean_input_eme": safe_mean(input_eme_vals),
        "mean_eme": safe_mean(eme_vals),
        "mean_delta_niqe": safe_mean(delta_niqe_vals),
        "mean_delta_eme": safe_mean(delta_eme_vals),
        "mean_loe": safe_mean(loe_vals),
        "niqe_backend": sorted(niqe_backends),
        "niqe_warning": sorted(niqe_warnings) or None,
    }


def evaluate_unpaired(
    enh_map: Dict[str, Path],
    stems: Sequence[str],
    eme_block: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    niqe_vals: List[float] = []
    eme_vals: List[float] = []
    niqe_backends = set()
    niqe_warnings = set()

    for stem in stems:
        enhanced_path = enh_map[stem]
        enhanced_gray = rgb_to_gray(load_rgb(enhanced_path))
        niqe, eme, niqe_backend, niqe_warn = compute_image_metrics(enhanced_gray, eme_block)

        if niqe_backend:
            niqe_backends.add(niqe_backend)
        if niqe_warn:
            niqe_warnings.add(niqe_warn)

        niqe_vals.append(niqe)
        eme_vals.append(eme)
        rows.append(
            {
                "stem": stem,
                "pairing_mode": "unpaired_enhanced_only",
                "input": "",
                "enhanced": str(enhanced_path),
                "input_niqe": float("nan"),
                "niqe": niqe,
                "input_eme": float("nan"),
                "eme": eme,
                "delta_niqe": float("nan"),
                "delta_eme": float("nan"),
                "loe": float("nan"),
                "niqe_backend": niqe_backend,
            }
        )

    return rows, {
        "mode": "unpaired_enhanced_only",
        "matched_images": 0,
        "mean_input_niqe": float("nan"),
        "mean_niqe": safe_mean(niqe_vals),
        "mean_input_eme": float("nan"),
        "mean_eme": safe_mean(eme_vals),
        "mean_delta_niqe": float("nan"),
        "mean_delta_eme": float("nan"),
        "mean_loe": float("nan"),
        "niqe_backend": sorted(niqe_backends),
        "niqe_warning": sorted(niqe_warnings) or None,
    }


def write_csv(csv_path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "stem",
        "pairing_mode",
        "input",
        "enhanced",
        "input_niqe",
        "niqe",
        "input_eme",
        "eme",
        "delta_niqe",
        "delta_eme",
        "loe",
        "niqe_backend",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate low-light enhancement with NIQE/EME/LOE")
    parser.add_argument("--input_dir", type=str, required=True, help="original low-light images")
    parser.add_argument("--enhanced_dir", type=str, required=True, help="enhanced output images")
    parser.add_argument("--output_dir", type=str, default="./results/metrics", help="where to save csv/json")
    parser.add_argument("--tag", type=str, default="run", help="name prefix for output files")
    parser.add_argument("--max_images", type=int, default=0, help="0 means evaluate all")
    parser.add_argument("--eme_block", type=int, default=8, help="block size for EME")
    parser.add_argument("--loe_max_side", type=int, default=50, help="downsample max side for LOE speed")
    parser.add_argument(
        "--allow_unpaired_fallback",
        action="store_true",
        help="when no stems match, still compute NIQE/EME on enhanced images and leave LOE empty",
    )
    parser.add_argument(
        "--preview_limit",
        type=int,
        default=10,
        help="how many unmatched stems or duplicate groups to store in summary preview",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    enhanced_dir = Path(args.enhanced_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir does not exist: {input_dir}")
    if not enhanced_dir.exists():
        raise FileNotFoundError(f"enhanced_dir does not exist: {enhanced_dir}")

    input_map, input_duplicates = build_stem_map(input_dir)
    enh_map, enh_duplicates = build_stem_map(enhanced_dir)

    if input_duplicates:
        raise ValueError(
            f"Duplicate stems in input_dir {input_dir}: {preview_duplicate_paths(input_duplicates, args.preview_limit)}"
        )
    if enh_duplicates:
        raise ValueError(
            f"Duplicate stems in enhanced_dir {enhanced_dir}: {preview_duplicate_paths(enh_duplicates, args.preview_limit)}"
        )

    common = sorted(set(input_map) & set(enh_map))
    input_only = sorted(set(input_map) - set(enh_map))
    enhanced_only = sorted(set(enh_map) - set(input_map))

    if args.max_images > 0:
        common = common[: args.max_images]
        enhanced_stems = sorted(enh_map)[: args.max_images]
    else:
        enhanced_stems = sorted(enh_map)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = output_dir / f"{args.tag}_{timestamp}_metrics.csv"
    json_path = output_dir / f"{args.tag}_{timestamp}_summary.json"
    project_root = output_dir.resolve().parents[1]

    if common:
        rows, aggregates = evaluate_pairs(
            input_map=input_map,
            enh_map=enh_map,
            stems=common,
            eme_block=args.eme_block,
            loe_max_side=args.loe_max_side,
        )
        note = "Paired evaluation completed. NIQE/LOE lower is usually better; EME higher is usually better."
    elif args.allow_unpaired_fallback and enh_map:
        rows, aggregates = evaluate_unpaired(
            enh_map=enh_map,
            stems=enhanced_stems,
            eme_block=args.eme_block,
        )
        note = (
            "No paired stems were found, so the script computed NIQE/EME on enhanced images only. "
            "LOE is unavailable without original inputs."
        )
    else:
        rows = []
        aggregates = {
            "mode": "no_pairs",
            "matched_images": 0,
            "mean_input_niqe": float("nan"),
            "mean_niqe": float("nan"),
            "mean_input_eme": float("nan"),
            "mean_eme": float("nan"),
            "mean_delta_niqe": float("nan"),
            "mean_delta_eme": float("nan"),
            "mean_loe": float("nan"),
            "niqe_backend": [],
            "niqe_warning": None,
        }
        note = (
            "No matched image pairs by stem name were found. "
            "Check stems in input/enhanced folders or rerun with --allow_unpaired_fallback."
        )

    if rows:
        write_csv(csv_path, rows)
        csv_value = relative_to(project_root, csv_path)
    else:
        csv_value = None

    summary = {
        "evaluation_version": EVALUATION_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "tag": args.tag,
        "input_dir": relative_to(project_root, input_dir),
        "enhanced_dir": relative_to(project_root, enhanced_dir),
        "mode": aggregates["mode"],
        "num_images": len(rows),
        "matched_images": aggregates["matched_images"],
        "input_only_count": len(input_only),
        "enhanced_only_count": len(enhanced_only),
        "mean_input_niqe": aggregates["mean_input_niqe"],
        "mean_niqe": aggregates["mean_niqe"],
        "mean_input_eme": aggregates["mean_input_eme"],
        "mean_eme": aggregates["mean_eme"],
        "mean_delta_niqe": aggregates["mean_delta_niqe"],
        "mean_delta_eme": aggregates["mean_delta_eme"],
        "mean_loe": aggregates["mean_loe"],
        "note": note,
        "niqe_backend": aggregates["niqe_backend"] or None,
        "niqe_warning": aggregates["niqe_warning"],
        "sample_input_only_stems": preview_strings(input_only, args.preview_limit),
        "sample_enhanced_only_stems": preview_strings(enhanced_only, args.preview_limit),
        "input_duplicate_preview": preview_duplicate_paths(input_duplicates, args.preview_limit),
        "enhanced_duplicate_preview": preview_duplicate_paths(enh_duplicates, args.preview_limit),
        "csv": csv_value,
    }

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(summary), handle, ensure_ascii=False, indent=2)

    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
