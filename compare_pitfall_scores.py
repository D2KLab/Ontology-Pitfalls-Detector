from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

PITFALL_TAXONOMY: Tuple[Tuple[str, str], ...] = (
    ("P1.1", "Parent disjoint with children"),
    ("P1.2", "Entity as subclass of both parent and grandparent"),
    ("P1.3", "Logical inconsistencies"),
    ("P2.1", "Not connected hierarchies"),
    ("P2.2", "Single subclass parent"),
    ("P2.3", "Superfluous disjointness"),
    ("P2.4", "Single subproperty parent"),
    ("P2.5", "Range/Domain expansion"),
    ("P2.6", "Possible hierarchy among properties"),
    ("P3.1", "Properties replicating standard RDF ones"),
    ("P3.2", "Range in property title"),
    ("P3.3", "Domain in property title"),
    ("P4.1", "Overly generic classes"),
    ("P4.2", "Synonyms in superclasses"),
    ("P4.3", "Conflicting hierarchy"),
    ("P4.4", "Subclasses with same semantics as superclasses"),
    ("P4.5", "Synonyms in properties"),
    ("P4.6", "Inverse properties not declared"),
    ("P4.7", "DataProperties that can become ObjectProperties"),
)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _extract_pitfall_metrics(pitfall_id: str, pitfall_result: object) -> Tuple[float, float]:
    """Extract count and ratio from pitfall result. Returns (count, ratio)."""
    if not isinstance(pitfall_result, dict):
        return 0.0, 0.0

    primary_count = pitfall_result.get("count")
    if _is_number(primary_count):
        ratio = pitfall_result.get("ratio")
        ratio_value = float(ratio) if _is_number(ratio) else 0.0
        return float(primary_count), ratio_value

    # P2.5 exposes two count buckets without a top-level "count" field.
    if pitfall_id == "P2.5":
        domain_count = pitfall_result.get("multi_domain_count")
        range_count = pitfall_result.get("multi_range_count")
        domain_ratio = pitfall_result.get("multi_domain_ratio")
        range_ratio = pitfall_result.get("multi_range_ratio")
        total_count = 0.0
        total_ratio = 0.0
        if _is_number(domain_count):
            total_count += float(domain_count)
        if _is_number(range_count):
            total_count += float(range_count)
        if _is_number(domain_ratio):
            total_ratio = float(domain_ratio)  # Use domain ratio as representative
        elif _is_number(range_ratio):
            total_ratio = float(range_ratio)
        return total_count, total_ratio

    total_count = pitfall_result.get("total_count")
    if _is_number(total_count):
        ratio = pitfall_result.get("ratio")
        ratio_value = float(ratio) if _is_number(ratio) else 0.0
        return float(total_count), ratio_value

    returned_count = pitfall_result.get("returned_count")
    if _is_number(returned_count):
        ratio = pitfall_result.get("ratio")
        ratio_value = float(ratio) if _is_number(ratio) else 0.0
        return float(returned_count), ratio_value

    fallback_counts = [
        float(value)
        for key, value in pitfall_result.items()
        if key.endswith("_count") and key != "checked_count" and _is_number(value)
    ]
    if fallback_counts:
        total_return = float(sum(fallback_counts))
        ratio = pitfall_result.get("ratio")
        ratio_value = float(ratio) if _is_number(ratio) else 0.0
        return total_return, ratio_value

    return 0.0, 0.0


def _harmonic_mean(values: Iterable[float]) -> float:
    positives = [value for value in values if value > 0]
    if not positives:
        return 0.0
    return len(positives) / sum(1.0 / value for value in positives)


def _harmonic_mean_of_ratios(values: Iterable[float]) -> float:
    """Harmonic mean of ratios (handled differently since ratios are 0-1)."""
    positives = [value for value in values if 0 < value < 1]
    if not positives:
        return 0.0
    return len(positives) / sum(1.0 / value for value in positives)


def _format_count(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _write_markdown_table(
    output_path: Path,
    per_file_scores: Dict[str, Dict[str, Tuple[float, float]]],
    filenames: List[str],
    column_names: List[str],
) -> None:
    """Write comparison data to a markdown table file."""
    lines: List[str] = []
    
    # Build header
    header = ["pitfall_id", "pitfall_title"]
    for col_name in column_names:
        header.append(f"{col_name}_count")
        header.append(f"{col_name}_ratio")
    lines.append("| " + " | ".join(header) + " |")
    
    # Separator row
    separator = [":-" if i == 0 else "-:" for i in range(len(header))]
    lines.append("| " + " | ".join(separator) + " |")
    
    # Data rows
    for pitfall_id, pitfall_title in PITFALL_TAXONOMY:
        row = [pitfall_id, pitfall_title]
        for filename in filenames:
            count, ratio = per_file_scores[filename][pitfall_id]
            row.append(_format_count(count))
            row.append(f"{ratio:.2f}")
        lines.append("| " + " | ".join(row) + " |")
    
    # Harmonic mean row
    harmonic_row = ["HARMONIC_MEAN", "Across pitfalls"]
    for filename in filenames:
        count_values = [per_file_scores[filename][pitfall_id][0] for pitfall_id, _ in PITFALL_TAXONOMY]
        ratio_values = [per_file_scores[filename][pitfall_id][1] for pitfall_id, _ in PITFALL_TAXONOMY]
        harmonic_row.append(f"{_harmonic_mean(count_values):.2f}")
        harmonic_row.append(f"{_harmonic_mean_of_ratios(ratio_values):.2f}")
    lines.append("| " + " | ".join(harmonic_row) + " |")
    
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _collect_scores(json_path: Path) -> Dict[str, Tuple[float, float]]:
    """Collect count and ratio for each pitfall. Returns {pitfall_id: (count, ratio)}."""
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    results = payload.get("results")
    if not isinstance(results, dict):
        raise ValueError("Missing or invalid 'results' object")

    scores: Dict[str, Tuple[float, float]] = {}
    for pitfall_id, _ in PITFALL_TAXONOMY:
        scores[pitfall_id] = _extract_pitfall_metrics(pitfall_id, results.get(pitfall_id))
    return scores


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare numeric pitfall count scores across all JSON files in a folder "
            "and export as CSV and markdown table with all pitfalls plus harmonic mean."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="output",
        help="Folder containing JSON outputs to compare (default: output).",
    )
    parser.add_argument(
        "--output",
        default="output/pitfall_score_comparison.csv",
        help="Output CSV path (default: output/pitfall_score_comparison.csv).",
    )
    parser.add_argument(
        "--output-markdown",
        default="output/pitfall_score_comparison.md",
        help="Output markdown table path (default: output/pitfall_score_comparison.md).",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input folder not found: {input_dir}", file=sys.stderr)
        return 1

    json_files = sorted(path for path in input_dir.glob("*.json") if path.is_file())
    if not json_files:
        print(f"No JSON files found in {input_dir}", file=sys.stderr)
        return 1

    per_file_scores: Dict[str, Dict[str, Tuple[float, float]]] = {}

    for json_file in json_files:
        try:
            per_file_scores[json_file.name] = _collect_scores(json_file)
        except Exception as exc:
            print(f"Skipping {json_file.name}: {exc}", file=sys.stderr)

    if not per_file_scores:
        print("No valid JSON files to compare.", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    filenames = sorted(per_file_scores.keys())
    column_names = [Path(filename).stem for filename in filenames]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        
        # Build header: pitfall_id, pitfall_title, then for each file: count, ratio
        header = ["pitfall_id", "pitfall_title"]
        for col_name in column_names:
            header.append(f"{col_name}_count")
            header.append(f"{col_name}_ratio")
        writer.writerow(header)

        for pitfall_id, pitfall_title in PITFALL_TAXONOMY:
            row = [pitfall_id, pitfall_title]
            for filename in filenames:
                count, ratio = per_file_scores[filename][pitfall_id]
                row.append(_format_count(count))
                row.append(f"{ratio:.2f}")
            writer.writerow(row)

        # Harmonic mean row for counts and ratios
        harmonic_row = ["HARMONIC_MEAN", "Across pitfalls"]
        for filename in filenames:
            count_values = [per_file_scores[filename][pitfall_id][0] for pitfall_id, _ in PITFALL_TAXONOMY]
            ratio_values = [per_file_scores[filename][pitfall_id][1] for pitfall_id, _ in PITFALL_TAXONOMY]
            harmonic_row.append(f"{_harmonic_mean(count_values):.2f}")
            harmonic_row.append(f"{_harmonic_mean_of_ratios(ratio_values):.2f}")
        writer.writerow(harmonic_row)

    print(f"Comparison CSV written to {output_path}")
    
    # Write markdown table
    markdown_path = Path(args.output_markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown_table(markdown_path, per_file_scores, filenames, column_names)
    print(f"Comparison markdown written to {markdown_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
