# Ontology Pitfalls Detector

This project contains ontology quality checks (`P1` to `P19`) extracted from the original notebook-style script and refactored into a reusable Python library.

The checks are now also exposed through a grouped pitfall taxonomy (`P1.1` to `P4.7`) for reporting.

The core package is `onto_pitfalls_lib`, and supports:

- Running one specific pitfall (`P1.1`, `P2.3`, ...) or legacy pattern (`P4`, `P6`, ...)
- Running multiple pitfalls/patterns in one execution
- Running all legacy patterns (`all`)
- JSON output to stdout or to file, including grouped pitfall results

## Project Structure

Key files for the refactored workflow:

- `onto_pitfalls_lib/runner.py`: `OntologyPatternToolkit` class with `run_p1()` ... `run_p19()`
- `onto_pitfalls_lib/utils.py`: shared utility functions used across pattern sections
- `onto_pitfalls_lib/cli.py`: command line interface
- `onto_pitfalls_lib/__main__.py`: enables `python -m onto_pitfalls_lib`
- `onto_pitfalls.py`: backward-compatible shim that delegates to the new CLI
- `requirements.txt`: Python dependencies
- `requirement.txt`: alias file that points to `requirements.txt`

## Installation

Use your preferred Python environment, then install dependencies:

```bash
pip install -r requirement.txt
```

Equivalent:

```bash
pip install -r requirements.txt
```

## CLI Usage

### List Available Patterns

```bash
python -m onto_pitfalls_lib --list-patterns
```

### Run Specific Patterns

Mapped pitfall identifiers:

```bash
python -m onto_pitfalls_lib --patterns P1.1 P2.3 P4.6
```

Space-separated:

```bash
python -m onto_pitfalls_lib --patterns P1 P4 P12
```

Comma-separated:

```bash
python -m onto_pitfalls_lib --patterns P1,P4,P12
```

Dotted format is accepted too:

```bash
python -m onto_pitfalls_lib --patterns P1. P4. P12.
```

### Run All Patterns

```bash
python -m onto_pitfalls_lib --patterns all
```

`all` runs all legacy patterns (`P1` ... `P19`).

### Select Ontology File

If `--ontology` is omitted, the CLI tries this default path:

`data/Engagen - Ontology Toolkit files/onto_engagen_V2.ttl`

To use another ontology:

```bash
python -m onto_pitfalls_lib \
  --ontology "data/AIQL - Ontology Toolkit files/ontology_AIQL_V1_2.ttl" \
  --patterns P4 P9
```

### Save Output to File

```bash
python -m onto_pitfalls_lib --patterns P4 P5 --output outputs/p4_p5.json
```

Use compact JSON (single line):

```bash
python -m onto_pitfalls_lib --patterns P4 --compact
```

### Backward Compatibility

The old entrypoint still works:

```bash
python onto_pitfalls.py --patterns P4
```

## Python API Usage

```python
from onto_pitfalls_lib import OntologyPatternToolkit

toolkit = OntologyPatternToolkit(
    "data/Engagen - Ontology Toolkit files/onto_engagen_V2.ttl"
)

# Single pattern
p4_result = toolkit.run_pattern("P4")

# Multiple patterns
subset_results = toolkit.run_patterns(["P1", "P4", "P12"])

# All patterns
all_results = toolkit.run_all()
```

## Output Format

CLI responses are JSON with this structure:

```json
{
  "metadata": {
    "ontology_path": "...",
    "classes": 0,
    "object_properties": 0,
    "datatype_properties": 0
  },
  "selected_patterns": ["P4", "P5"],
  "selected_pitfalls": ["P1.1", "P2.2"],
  "results": {
    "P4": {"count": 0, "items": []},
    "P5": {"count": 0, "items": []}
  },
  "pitfall_results": {
    "P1.1": {
      "legacy_pattern": "P4",
      "title": "Parent disjoint with children",
      "result": {"count": 0, "items": []}
    },
    "P2.2": {
      "legacy_pattern": "P5",
      "title": "Single subclass parent",
      "result": {"count": 0, "items": []}
    }
  },
  "grouped_results": {
    "Logical Issues": {
      "P1.1": {
        "legacy_pattern": "P4",
        "title": "Parent disjoint with children",
        "result": {"count": 0, "items": []}
      }
    },
    "Structural Issues": {
      "P2.2": {
        "legacy_pattern": "P5",
        "title": "Single subclass parent",
        "result": {"count": 0, "items": []}
      }
    }
  }
}
```

## Pitfall Taxonomy

### 1. Logical Issues

- `P1.1` <- `P4`: Parent disjoint with children
- `P1.2` <- `P9`: Entity as subclass of both parent and grandparent
- `P1.3` <- `P19`: Logical inconsistencies

### 2. Structural Issues

- `P2.1` <- `P2`: Not connected hierarchies
- `P2.2` <- `P5`: Single subclass parent
- `P2.3` <- `P6`: Superfluous disjointness
- `P2.4` <- `P13`: Single subproperty parent
- `P2.5` <- `P14`: Range/Domain expansion
- `P2.6` <- `P12`: Possible hierarchy among properties

### 3. Redundancy / Naming Issues

- `P3.1` <- `P15`: Properties replicating standard RDF ones
- `P3.2` <- `P17`: Range in property title
- `P3.3` <- `P18`: Domain in property title

### 4. Semantic Issues

- `P4.1` <- `P1`: Overly generic classes
- `P4.2` <- `P3`: Synonyms in superclasses
- `P4.3` <- `P7`: Conflicting hierarchy
- `P4.4` <- `P8`: Subclasses with same semantics as superclasses
- `P4.5` <- `P10`: Synonyms in properties
- `P4.6` <- `P11`: Inverse properties not declared
- `P4.7` <- `P16`: DataProperties that can become ObjectProperties

## Notes

- Some patterns rely on `SentenceTransformer` models and may download model weights on first run.
- NLTK resources are fetched on demand when needed by a pattern.
- `P19` uses `owlready2` reasoner integration and can take longer than purely syntactic checks.
