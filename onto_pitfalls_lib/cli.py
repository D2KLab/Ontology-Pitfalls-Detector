from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .runner import OntologyPatternToolkit
from .utils import parse_pattern_selection

DEFAULT_ONTOLOGY_PATH = 'data/ontology_AIQL_V1_2.ttl'


def _format_pitfall_taxonomy() -> str:
    taxonomy = OntologyPatternToolkit.pitfall_taxonomy()
    lines: List[str] = []
    current_category: Optional[str] = None

    for entry in taxonomy:
        category = entry["category"]
        if category != current_category:
            if lines:
                lines.append("")
            lines.append(f"{category}:")
            current_category = category

        lines.append(
            f"  {entry['pitfall_id']} ({entry['legacy_pattern']}): {entry['title']}"
        )

    mapped_patterns = {entry["legacy_pattern"] for entry in taxonomy}
    unmapped_patterns = [
        pattern_id
        for pattern_id in OntologyPatternToolkit.available_patterns()
        if pattern_id not in mapped_patterns
    ]

    if unmapped_patterns:
        lines.append("")
        lines.append("Unmapped legacy patterns:")
        for pattern_id in unmapped_patterns:
            lines.append(f"  {pattern_id}")

    return "\n".join(lines)


def _reorganize_results(
    selected_patterns: Sequence[str],
    results: Dict[str, Dict[str, Any]],
) -> Tuple[List[str], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Dict[str, Any]]]]:
    taxonomy = OntologyPatternToolkit.pitfall_taxonomy()
    mapped_patterns = {entry["legacy_pattern"] for entry in taxonomy}
    selected_set = set(selected_patterns)

    selected_pitfalls: List[str] = []
    pitfall_results: Dict[str, Dict[str, Any]] = {}
    grouped_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for entry in taxonomy:
        legacy_pattern = entry["legacy_pattern"]
        if legacy_pattern not in selected_set:
            continue

        category = entry["category"]
        pitfall_id = entry["pitfall_id"]
        payload_entry = {
            "legacy_pattern": legacy_pattern,
            "title": entry["title"],
            "result": results[legacy_pattern],
        }

        selected_pitfalls.append(pitfall_id)
        pitfall_results[pitfall_id] = payload_entry

        if category not in grouped_results:
            grouped_results[category] = {}
        grouped_results[category][pitfall_id] = payload_entry

    unmapped_selected = [
        pattern_id for pattern_id in selected_patterns if pattern_id not in mapped_patterns
    ]
    if unmapped_selected:
        category = "Unmapped Legacy Issues"
        grouped_results[category] = {}
        for pattern_id in unmapped_selected:
            payload_entry = {
                "legacy_pattern": pattern_id,
                "title": "Legacy pattern without taxonomy mapping",
                "result": results[pattern_id],
            }
            selected_pitfalls.append(pattern_id)
            pitfall_results[pattern_id] = payload_entry
            grouped_results[category][pattern_id] = payload_entry

    return selected_pitfalls, pitfall_results, grouped_results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ontology toolkit pitfall checks (legacy P1-P19 and mapped P1.1-P4.7).",
    )
    parser.add_argument(
        "--ontology",
        help="Path to ontology .ttl file. If omitted, uses the default Engagen ontology when available.",
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["all"],
        help="Pitfalls to run, e.g. P1.1 P2.3 or legacy P4 P6; use all to run every legacy pattern.",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model used by semantic similarity checks.",
    )
    parser.add_argument(
        "--output",
        help="Optional output JSON file path.",
        default='output/pitfall_results.json',
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print compact JSON instead of pretty output.",
    )
    parser.add_argument(
        "--list-patterns",
        action="store_true",
        help="Print the pitfall taxonomy mapping and exit.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_patterns:
        print(_format_pitfall_taxonomy())
        return 0

    ontology_path = Path(args.ontology) if args.ontology else Path(DEFAULT_ONTOLOGY_PATH)

    if not ontology_path.exists():
        parser.error(f"Ontology file not found: {ontology_path}")

    toolkit = OntologyPatternToolkit(str(ontology_path), model_name=args.model)

    try:
        selected_patterns = parse_pattern_selection(
            args.patterns,
            toolkit.available_patterns(),
            normalizer=OntologyPatternToolkit.resolve_pattern_id,
        )
    except ValueError as exc:
        parser.error(str(exc))

    results = {pattern_id: toolkit.run_pattern(pattern_id) for pattern_id in selected_patterns}
    selected_pitfalls, pitfall_results, grouped_results = _reorganize_results(
        selected_patterns,
        results,
    )

    payload = {
        "metadata": toolkit.metadata(),
        "selected_patterns": selected_patterns,
        "selected_pitfalls": selected_pitfalls,
        "results": results,
        "pitfall_results": pitfall_results,
        "grouped_results": grouped_results,
    }

    text = json.dumps(payload, indent=None if args.compact else 2, ensure_ascii=True)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(f"{text}\n", encoding="utf-8")
    else:
        print(text)

    return 0
