import argparse
import csv
import json
import math
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt


SUMMARY_RE = re.compile(r"^(?P<tag>.+)_(?P<ts>\d{8}-\d{6})_summary\.json$")


@dataclass
class RunRecord:
    summary_path: Path
    csv_path: Optional[Path]
    tag: str
    canonical_tag: str
    timestamp: str
    data: Dict[str, Any]
    status: str
    reason: str
    priority: int


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return float(value)
    return None


def relative_to(base: Path, target: Optional[Path]) -> Optional[str]:
    if target is None:
        return None
    try:
        return os.path.relpath(target.resolve(), start=base.resolve())
    except Exception:
        return str(target)


def canonicalize_tag(tag: str) -> str:
    normalized = tag.replace("additonal_", "additional_")
    if normalized.endswith("_niqe_ok"):
        normalized = normalized[:-8]
    return normalized


def classify(summary_data: Dict[str, Any], tag: str) -> Tuple[str, str, int]:
    num_images = int(summary_data.get("num_images", 0) or 0)
    mean_niqe = safe_float(summary_data.get("mean_niqe"))
    mode = summary_data.get("mode") or ""
    niqe_warning = summary_data.get("niqe_warning")

    if mode == "unpaired_enhanced_only":
        return "partial_unpaired", "enhanced-only metrics without LOE", 2
    if num_images == 0:
        return "invalid_no_pairs", "no matched stems", 0
    if mean_niqe is None:
        return "legacy_niqe_failed", "NIQE unavailable or failed", 1
    if summary_data.get("evaluation_version"):
        return "trusted_valid", "current evaluation version", 5
    if tag.endswith("_niqe_ok"):
        return "trusted_valid", "post-fix NIQE run", 4
    if niqe_warning:
        return "legacy_warning", "legacy run with warnings", 2
    return "valid_legacy", "legacy run with usable metrics", 3


def discover_runs(metrics_dir: Path) -> List[RunRecord]:
    runs: List[RunRecord] = []
    project_root = metrics_dir.parents[1]

    for summary_path in sorted(metrics_dir.rglob("*_summary.json")):
        if any(part in {"archive", "plots", "reports", "__pycache__"} for part in summary_path.parts):
            continue
        match = SUMMARY_RE.match(summary_path.name)
        if not match:
            continue

        data = read_json(summary_path)
        csv_field = data.get("csv")
        csv_path = None
        if csv_field:
            candidate = project_root / csv_field
            if candidate.exists():
                csv_path = candidate
            else:
                alt = summary_path.with_name(summary_path.name.replace("_summary.json", "_metrics.csv"))
                if alt.exists():
                    csv_path = alt

        tag = match.group("tag")
        timestamp = match.group("ts")
        status, reason, priority = classify(data, tag)
        runs.append(
            RunRecord(
                summary_path=summary_path,
                csv_path=csv_path,
                tag=tag,
                canonical_tag=canonicalize_tag(tag),
                timestamp=timestamp,
                data=data,
                status=status,
                reason=reason,
                priority=priority,
            )
        )
    return runs


def choose_current(runs: Iterable[RunRecord]) -> Dict[str, RunRecord]:
    grouped: Dict[str, List[RunRecord]] = defaultdict(list)
    for run in runs:
        grouped[run.canonical_tag].append(run)

    selected: Dict[str, RunRecord] = {}
    for canonical_tag, items in grouped.items():
        selected[canonical_tag] = sorted(
            items,
            key=lambda run: (run.priority, run.timestamp, run.summary_path.name),
            reverse=True,
        )[0]
    return selected


def move_run(record: RunRecord, destination_dir: Path, project_root: Path) -> RunRecord:
    destination_dir.mkdir(parents=True, exist_ok=True)
    new_csv_path = None
    if record.csv_path and record.csv_path.exists():
        new_csv_path = destination_dir / record.csv_path.name
        if record.csv_path.resolve() != new_csv_path.resolve():
            shutil.move(str(record.csv_path), str(new_csv_path))

    new_summary_path = destination_dir / record.summary_path.name
    if record.summary_path.resolve() != new_summary_path.resolve():
        shutil.move(str(record.summary_path), str(new_summary_path))

    data = read_json(new_summary_path)
    data["csv"] = relative_to(project_root, new_csv_path)
    with new_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)

    return RunRecord(
        summary_path=new_summary_path,
        csv_path=new_csv_path,
        tag=record.tag,
        canonical_tag=record.canonical_tag,
        timestamp=record.timestamp,
        data=data,
        status=record.status,
        reason=record.reason,
        priority=record.priority,
    )


def organize_runs(metrics_dir: Path, runs: List[RunRecord], selected: Dict[str, RunRecord]) -> None:
    archive_dirs = {
        "legacy_niqe_failed": metrics_dir / "archive" / "legacy_niqe_failed",
        "legacy_warning": metrics_dir / "archive" / "legacy_warning",
        "invalid_no_pairs": metrics_dir / "archive" / "invalid_no_pairs",
        "partial_unpaired": metrics_dir / "archive" / "partial_unpaired",
        "valid_legacy": metrics_dir / "archive" / "superseded_valid",
        "trusted_valid": metrics_dir / "archive" / "superseded_valid",
    }
    current_dir = metrics_dir / "current"
    project_root = metrics_dir.parents[1]
    current_summary_paths = {run.summary_path.resolve() for run in selected.values()}

    for run in runs:
        if run.summary_path.resolve() in current_summary_paths:
            move_run(run, current_dir, project_root)
            continue
        destination = archive_dirs.get(run.status)
        if destination:
            move_run(run, destination, project_root)


def write_csv_table(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_manifest_rows(runs: Iterable[RunRecord], selected: Dict[str, RunRecord]) -> List[Dict[str, Any]]:
    current_paths = {run.summary_path.resolve() for run in selected.values()}
    rows: List[Dict[str, Any]] = []
    for run in sorted(runs, key=lambda item: (item.canonical_tag, item.timestamp, item.summary_path.name)):
        data = run.data
        rows.append(
            {
                "canonical_tag": run.canonical_tag,
                "tag": run.tag,
                "timestamp": run.timestamp,
                "status": run.status,
                "reason": run.reason,
                "is_current": run.summary_path.resolve() in current_paths,
                "mode": data.get("mode"),
                "num_images": data.get("num_images"),
                "mean_niqe": data.get("mean_niqe"),
                "mean_eme": data.get("mean_eme"),
                "mean_loe": data.get("mean_loe"),
                "summary": str(run.summary_path),
                "csv": str(run.csv_path) if run.csv_path else None,
            }
        )
    return rows


def current_rows(selected: Dict[str, RunRecord]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for canonical_tag in sorted(selected):
        run = selected[canonical_tag]
        data = run.data
        rows.append(
            {
                "canonical_tag": canonical_tag,
                "tag": run.tag,
                "timestamp": run.timestamp,
                "mode": data.get("mode"),
                "num_images": data.get("num_images"),
                "mean_input_niqe": data.get("mean_input_niqe"),
                "mean_niqe": data.get("mean_niqe"),
                "mean_delta_niqe": data.get("mean_delta_niqe"),
                "mean_input_eme": data.get("mean_input_eme"),
                "mean_eme": data.get("mean_eme"),
                "mean_delta_eme": data.get("mean_delta_eme"),
                "mean_loe": data.get("mean_loe"),
                "summary": str(run.summary_path),
                "csv": str(run.csv_path) if run.csv_path else None,
            }
        )
    return rows


def plot_metric_bars(selected: Dict[str, RunRecord], output_path: Path) -> None:
    tags = [tag for tag, run in sorted(selected.items()) if safe_float(run.data.get("mean_niqe")) is not None]
    if not tags:
        return

    niqe = [safe_float(selected[tag].data.get("mean_niqe")) or 0.0 for tag in tags]
    eme = [safe_float(selected[tag].data.get("mean_eme")) or 0.0 for tag in tags]
    loe = [safe_float(selected[tag].data.get("mean_loe")) or 0.0 for tag in tags]

    x = range(len(tags))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([idx - width for idx in x], niqe, width=width, label="NIQE", color="#4063D8")
    ax.bar(list(x), eme, width=width, label="EME", color="#389826")
    ax.bar([idx + width for idx in x], loe, width=width, label="LOE", color="#CB3C33")
    ax.set_xticks(list(x))
    ax.set_xticklabels(tags, rotation=20, ha="right")
    ax.set_title("Current Metric Summary")
    ax.set_ylabel("Metric Value")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_delta_bars(selected: Dict[str, RunRecord], output_path: Path) -> None:
    rows = []
    for tag, run in sorted(selected.items()):
        delta_niqe = safe_float(run.data.get("mean_delta_niqe"))
        delta_eme = safe_float(run.data.get("mean_delta_eme"))
        if delta_niqe is None and delta_eme is None:
            continue
        rows.append((tag, delta_niqe or 0.0, delta_eme or 0.0))
    if not rows:
        return

    tags = [row[0] for row in rows]
    delta_niqe = [row[1] for row in rows]
    delta_eme = [row[2] for row in rows]
    x = range(len(tags))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([idx - width / 2 for idx in x], delta_niqe, width=width, label="Delta NIQE", color="#9558B2")
    ax.bar([idx + width / 2 for idx in x], delta_eme, width=width, label="Delta EME", color="#FF7F0E")
    ax.axhline(0.0, color="#333333", linewidth=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(tags, rotation=20, ha="right")
    ax.set_title("Change Versus Input")
    ax.set_ylabel("Delta")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def load_benchmarks(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None or not path.exists():
        return []
    data = read_json(path)
    return data.get("benchmarks", [])


def render_report(
    path: Path,
    selected: Dict[str, RunRecord],
    runs: List[RunRecord],
    benchmark_items: List[Dict[str, Any]],
    plots_dir: Path,
) -> None:
    current = current_rows(selected)
    status_counts: Dict[str, int] = defaultdict(int)
    for run in runs:
        status_counts[run.status] += 1

    lines = [
        "# CVPR Metrics Audit Report",
        "",
        "## Key Findings",
        "",
    ]

    for row in current:
        if row["mode"] == "paired":
            lines.append(
                f"- `{row['canonical_tag']}`: NIQE delta {row['mean_delta_niqe']:.4f}, "
                f"EME delta {row['mean_delta_eme']:.4f}, LOE {row['mean_loe']:.4f}."
            )
        else:
            lines.append(
                f"- `{row['canonical_tag']}`: only unpaired NIQE/EME are available because the enhanced filenames do not match the current input set."
            )

    lines.extend(
        [
            "",
        "## Current Canonical Runs",
        "",
        "| Tag | Mode | Images | Mean NIQE | Mean EME | Mean LOE | Delta NIQE | Delta EME |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in current:
        lines.append(
            "| {canonical_tag} | {mode} | {num_images} | {mean_niqe} | {mean_eme} | {mean_loe} | {mean_delta_niqe} | {mean_delta_eme} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Inventory",
            "",
            "| Status | Count |",
            "| --- | ---: |",
        ]
    )
    for status, count in sorted(status_counts.items()):
        lines.append(f"| {status} | {count} |")

    lines.extend(
        [
            "",
            "## Charts",
            "",
            f"![Current metrics]({relative_to(path.parent, plots_dir / 'current_metrics.png')})",
            "",
            f"![Delta metrics]({relative_to(path.parent, plots_dir / 'delta_metrics.png')})",
            "",
        ]
    )

    if benchmark_items:
        lines.extend(
            [
                "## Literature Reference Ranges",
                "",
                "| Model | Dataset | Metric | Value | Source | Note |",
                "| --- | --- | --- | ---: | --- | --- |",
            ]
        )
        for item in benchmark_items:
            lines.append(
                f"| {item['model']} | {item['dataset']} | {item['metric']} | {item['value']} | [{item['source_label']}]({item['source_url']}) | {item.get('note', '')} |"
            )
        lines.extend(
            [
                "",
                "These literature values come from different datasets and papers, so they are for rough orientation only and should not be treated as a strict apples-to-apples leaderboard.",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser("Organize and visualize evaluation outputs")
    parser.add_argument("--metrics_dir", type=str, default="./results/metrics", help="root metrics directory")
    parser.add_argument("--docs_dir", type=str, default="../../docs/evaluation", help="documentation output directory")
    parser.add_argument("--benchmark_json", type=str, default="./evaluation/literature_benchmarks.json")
    parser.add_argument("--archive_legacy", action="store_true", help="move runs into current/archive folders")
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir).resolve()
    docs_dir = Path(args.docs_dir).resolve()
    benchmark_path = Path(args.benchmark_json).resolve()

    runs = discover_runs(metrics_dir)
    selected = choose_current(runs)

    if args.archive_legacy:
        organize_runs(metrics_dir, runs, selected)
        runs = discover_runs(metrics_dir)
        selected = choose_current(runs)

    reports_dir = metrics_dir / "reports"
    plots_dir = metrics_dir / "plots"
    manifests_dir = metrics_dir / "manifests"

    write_csv_table(manifests_dir / "metrics_manifest.csv", make_manifest_rows(runs, selected))
    write_csv_table(reports_dir / "current_metrics.csv", current_rows(selected))
    plot_metric_bars(selected, plots_dir / "current_metrics.png")
    plot_delta_bars(selected, plots_dir / "delta_metrics.png")

    manifest_json = {
        "current": current_rows(selected),
        "all_runs": make_manifest_rows(runs, selected),
    }
    manifests_dir.mkdir(parents=True, exist_ok=True)
    with (manifests_dir / "metrics_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest_json, handle, ensure_ascii=False, indent=2)

    render_report(
        path=docs_dir / "CVPR_metrics_audit_report.md",
        selected=selected,
        runs=runs,
        benchmark_items=load_benchmarks(benchmark_path),
        plots_dir=plots_dir,
    )


if __name__ == "__main__":
    main()
