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


def _pick_value(item: Dict[str, Any], keys: Sequence[str], fallback: str = "Unknown") -> str:
    for key in keys:
        value = item.get(key)
        if value is None:
            continue

        text = str(value).strip()
        if text:
            return text

    return fallback


def _format_score(item: Dict[str, Any], key: str) -> str:
    value = item.get(key)
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"
    return "n/a"


def _summarize_list(values: Any, limit: int = 4) -> str:
    if not isinstance(values, list):
        return "none"

    labels: List[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            labels.append(text)

    if not labels:
        return "none"
    if len(labels) <= limit:
        return ", ".join(labels)

    hidden = len(labels) - limit
    return f"{', '.join(labels[:limit])}, +{hidden} more"


def _describe_item(pitfall_id: str, item: Any, bucket: str = "items") -> str:
    if not isinstance(item, dict):
        raw = str(item).strip()
        if not raw:
            return "Potential pitfall detected."
        if pitfall_id == "P1.3":
            return f"{raw} is logically inconsistent."
        return f"Potential pitfall detected: {raw}."

    if pitfall_id == "P1.1":
        child = _pick_value(item, ("child_label", "child_uri"))
        parent = _pick_value(item, ("parent_label", "parent_uri"))
        return f"{child} is child of {parent} even though they are disjoint."

    if pitfall_id == "P1.2":
        child = _pick_value(item, ("child_label", "child_uri"))
        parent = _pick_value(item, ("parent_label", "parent_uri"))
        grandparent = _pick_value(item, ("grandparent_label", "grandparent_uri"))
        return f"{child} is subclass of both {parent} and {grandparent}."

    if pitfall_id == "P2.1":
        class_label = _pick_value(item, ("class_label", "class_uri"))
        missing_parent = _pick_value(item, ("missing_parent_label", "missing_parent_uri"))
        return f"{class_label} is not child of {missing_parent}."

    if pitfall_id == "P2.2":
        class_label = _pick_value(item, ("class_label", "class_uri"))
        return f"{class_label} has only one direct subclass."

    if pitfall_id == "P2.3":
        class_1 = _pick_value(item, ("class_1_label", "class_1_uri"))
        class_2 = _pick_value(item, ("class_2_label", "class_2_uri"))
        score = _format_score(item, "label_similarity")
        return f"{class_1} and {class_2} are disjoint but semantically similar (similarity {score})."

    if pitfall_id == "P2.4":
        prop = _pick_value(item, ("property_label", "property_uri"))
        return f"{prop} has only one direct subproperty."

    if pitfall_id == "P2.5" and bucket == "multi_domain_items":
        prop = _pick_value(item, ("property_label", "property_uri"))
        domains = _summarize_list(item.get("domains"))
        supers = _summarize_list(item.get("common_superclasses"))
        return f"{prop} has multiple domains ({domains}) with shared superclasses ({supers})."

    if pitfall_id == "P2.5" and bucket == "multi_range_items":
        prop = _pick_value(item, ("property_label", "property_uri"))
        ranges = _summarize_list(item.get("ranges"))
        supers = _summarize_list(item.get("common_superclasses"))
        return f"{prop} has multiple ranges ({ranges}) with shared superclasses ({supers})."

    if pitfall_id == "P2.6":
        p1 = _pick_value(item, ("p1_label", "p1_uri"))
        p2 = _pick_value(item, ("p2_label", "p2_uri"))
        score = _format_score(item, "combined_similarity")

        if bucket == "already_siblings":
            parents = _summarize_list(item.get("common_parent_labels"))
            return (
                f"{p1} and {p2} are similar properties and already share parent(s): "
                f"{parents} (similarity {score})."
            )

        domain_matched = "yes" if bool(item.get("domain_matched")) else "no"
        return (
            f"{p1} and {p2} are similar properties and may need a shared parent "
            f"(similarity {score}, domain/range match: {domain_matched})."
        )

    if pitfall_id == "P3.1":
        prop = _pick_value(item, ("property_label", "property_uri"))
        matches = _summarize_list(item.get("matched_standard_props"), limit=5)
        if item.get("exact_match"):
            return f"{prop} exactly matches standard property name(s): {matches}."
        return f"{prop} resembles standard property name(s): {matches}."

    if pitfall_id == "P3.2":
        prop = _pick_value(item, ("property_label", "property_uri"))
        range_label = _pick_value(item, ("range_label", "range_uri"))
        return f"{prop} repeats its range {range_label} in the property name."

    if pitfall_id == "P3.3":
        prop = _pick_value(item, ("property_label", "property_uri"))
        domain_label = _pick_value(item, ("domain_label", "domain_uri"))
        return f"{prop} repeats its domain {domain_label} in the property name."

    if pitfall_id == "P4.1":
        class_label = _pick_value(item, ("class_label", "class_uri"))
        distance = _pick_value(item, ("distance",), fallback="unknown")
        return f"{class_label} appears overly generic (distance {distance} from top-level concepts)."

    if pitfall_id == "P4.2":
        class_1 = _pick_value(item, ("class_1_label", "class_1_uri"))
        class_2 = _pick_value(item, ("class_2_label", "class_2_uri"))
        score = _format_score(item, "combined_similarity")
        return f"{class_1} and {class_2} may be synonyms (combined similarity {score})."

    if pitfall_id == "P4.3":
        child = _pick_value(item, ("child_label", "child_uri"))
        parent = _pick_value(item, ("parent_label", "parent_uri"))
        reason = _pick_value(item, ("reason",), fallback="semantic contrast")
        return f"{child} may conflict with parent {parent} ({reason})."

    if pitfall_id == "P4.4":
        child = _pick_value(item, ("child_label", "child_uri"))
        parent = _pick_value(item, ("parent_label", "parent_uri"))
        score = _format_score(item, "semantic_similarity")
        return f"{child} is semantically very close to parent {parent} (similarity {score})."

    if pitfall_id == "P4.5":
        p1 = _pick_value(item, ("p1_label", "p1_uri"))
        p2 = _pick_value(item, ("p2_label", "p2_uri"))
        score = _format_score(item, "combined_similarity")
        return f"{p1} and {p2} may be synonymous properties (combined similarity {score})."

    if pitfall_id == "P4.6":
        p1 = _pick_value(item, ("p1_label", "p1_uri"))
        p2 = _pick_value(item, ("p2_label", "p2_uri"))
        score = _format_score(item, "combined_similarity")
        if item.get("domain_swapped"):
            relation = "with swapped domain/range"
        else:
            relation = "with partially mirrored domain/range"
        return (
            f"{p1} and {p2} may be undeclared inverse properties {relation} "
            f"(similarity {score})."
        )

    if pitfall_id == "P4.7":
        short = _pick_value(item, ("short_label", "short_uri"))
        long = _pick_value(item, ("long_label", "long_uri"))
        return f"{short} overlaps with {long} and may hide an object-property relation."

    focus = _pick_value(
        item,
        ("class_label", "property_label", "child_label", "parent_label", "p1_label", "short_label"),
        fallback="the item",
    )
    return f"Potential pitfall detected for {focus}."


def _add_human_descriptions(results: Dict[str, Dict[str, Any]]) -> None:
    list_keys = ("items", "already_siblings", "multi_domain_items", "multi_range_items")

    for pitfall_id, result in results.items():
        if not isinstance(result, dict):
            continue

        for list_key in list_keys:
            entries = result.get(list_key)
            if not isinstance(entries, list):
                continue

            extra_descriptions: List[str] = []
            for entry in entries:
                description = _describe_item(pitfall_id, entry, bucket=list_key)
                if isinstance(entry, dict):
                    entry["description"] = description
                else:
                    extra_descriptions.append(description)

            if extra_descriptions:
                desc_key = "descriptions" if list_key == "items" else f"{list_key}_descriptions"
                result[desc_key] = extra_descriptions


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
    _add_human_descriptions(results)

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
