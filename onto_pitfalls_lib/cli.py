from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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
            f"  {entry['pitfall_id']}: {entry['title']}"
        )

    return "\n".join(lines)


def _group_results_by_category(
    selected_pitfalls: Sequence[str],
    results: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    taxonomy_by_id = {
        entry["pitfall_id"]: entry for entry in OntologyPatternToolkit.pitfall_taxonomy()
    }
    grouped_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for pitfall_id in selected_pitfalls:
        entry = taxonomy_by_id.get(pitfall_id)
        if entry is None:
            continue

        category = entry["category"]
        payload_entry = {
            "title": entry["title"],
            "result": results[pitfall_id],
        }

        if category not in grouped_results:
            grouped_results[category] = {}
        grouped_results[category][pitfall_id] = payload_entry

    return grouped_results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ontology toolkit pitfall checks using taxonomy IDs (P1.1-P4.7).",
    )
    parser.add_argument(
        "--ontology",
        help="Path to ontology .ttl file. If omitted, uses the default Engagen ontology when available.",
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["all"],
        help="Pitfalls to run, e.g. P1.1 P2.3 P4.6; use all to run every pitfall.",
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
        selected_pitfalls = parse_pattern_selection(
            args.patterns,
            toolkit.available_patterns(),
            normalizer=OntologyPatternToolkit.normalize_pitfall_id,
        )
    except ValueError as exc:
        parser.error(str(exc))

    results = {pitfall_id: toolkit.run_pattern(pitfall_id) for pitfall_id in selected_pitfalls}
    grouped_results = _group_results_by_category(
        selected_pitfalls,
        results,
    )

    payload = {
        "metadata": toolkit.metadata(),
        "selected_pitfalls": selected_pitfalls,
        "results": results,
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
