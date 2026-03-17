# Ontology Pitfalls Detector

This project contains ontology quality checks organized as a grouped pitfall taxonomy (`P1.1` to `P4.7`), extracted from the original notebook-style script and refactored into a reusable Python library.

The core package is `onto_pitfalls_lib`, and supports:

- Running one specific pitfall (`P1.1`, `P2.3`, ...)
- Running multiple pitfalls in one execution
- Running all pitfalls (`all`)
- JSON output to stdout or to file, including grouped pitfall results

## Project Structure

Key files for the refactored workflow:

- `onto_pitfalls_lib/runner.py`: `OntologyPatternToolkit` class with taxonomy-aligned methods (`run_p1_1()` ... `run_p4_7()`)
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

Space-separated:

```bash
python -m onto_pitfalls_lib --patterns P1.1 P2.3 P4.6
```

Comma-separated:

```bash
python -m onto_pitfalls_lib --patterns P1.1,P2.3,P4.6
```

Dotted format is accepted too:

```bash
python -m onto_pitfalls_lib --patterns P1.1. P2.3. P4.6.
```

### Run All Patterns

```bash
python -m onto_pitfalls_lib --patterns all
```

`all` runs all taxonomy pitfalls (`P1.1` ... `P4.7`).

### Select Ontology File

If `--ontology` is omitted, the CLI tries a default path.

To use another ontology:

```bash
python -m onto_pitfalls_lib \
  --ontology "data/my_onto.ttl" \
  --patterns P1.1 P1.2
```

### Save Output to File

```bash
python -m onto_pitfalls_lib --patterns P1.1 P2.2 --output outputs/p1_1_p2_2.json
```

Use compact JSON (single line):

```bash
python -m onto_pitfalls_lib --patterns P1.1 --compact
```

## Compare Multiple JSON Outputs

You can compare numeric pitfall counts across all JSON files in `output/` and export a CSV containing all taxonomy pitfalls (`P1.1` to `P4.7`) plus a harmonic mean row:

```bash
python compare_pitfall_scores.py
```

Custom paths:

```bash
python compare_pitfall_scores.py \
  --input-dir output \
  --output output/pitfall_score_comparison.csv
```

### Backward Compatibility

The old entrypoint still works:

```bash
python onto_pitfalls.py --patterns P1.1
```

## Python API Usage

```python
from onto_pitfalls_lib import OntologyPatternToolkit

toolkit = OntologyPatternToolkit(
    "data/my_onto.ttl"
)

# Single pitfall
p11_result = toolkit.run_pattern("P1.1")

# Multiple pitfalls
subset_results = toolkit.run_patterns(["P1.1", "P2.3", "P4.6"])

# All pitfalls
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
  "selected_pitfalls": ["P1.1", "P2.2"],
  "results": {
    "P1.1": {
      "count": 0,
      "items": []
    },
    "P2.2": {
      "count": 0,
      "items": []
    }
  },
  "grouped_results": {
    "Logical Issues": {
      "P1.1": {
        "title": "Parent disjoint with children",
        "result": {"count": 0, "items": []}
      }
    },
    "Structural Issues": {
      "P2.2": {
        "title": "Single subclass parent",
        "result": {"count": 0, "items": []}
      }
    }
  }
}
```

## Pitfall Taxonomy

### 1. Logical Issues

- `P1.1`: Parent disjoint with children
- `P1.2`: Entity as subclass of both parent and grandparent
- `P1.3`: Logical inconsistencies

### 2. Structural Issues

- `P2.1`: Not connected hierarchies
- `P2.2`: Single subclass parent
- `P2.3`: Superfluous disjointness
- `P2.4`: Single subproperty parent
- `P2.5`: Range/Domain expansion
- `P2.6`: Possible hierarchy among properties

### 3. Redundancy / Naming Issues

- `P3.1`: Properties replicating standard RDF ones
- `P3.2`: Range in property title
- `P3.3`: Domain in property title

### 4. Semantic Issues

- `P4.1`: Overly generic classes
- `P4.2`: Synonyms in superclasses
- `P4.3`: Conflicting hierarchy
- `P4.4`: Subclasses with same semantics as superclasses
- `P4.5`: Synonyms in properties
- `P4.6`: Inverse properties not declared
- `P4.7`: DataProperties that can become ObjectProperties

## Notes

- Some patterns rely on `SentenceTransformer` models and may download model weights on first run.
- NLTK resources are fetched on demand when needed by a pattern.
- `P19` uses `owlready2` reasoner integration and can take longer than purely syntactic checks.

## Tested ontologies

The library has been tested on 2 ontologies generated by [Ontology Toolkit](https://www.lettria.com/blogpost/say-goodbye-to-manual-ontology-building-introducing-the-ontology-toolkit), on two corpora available in the `documents corpora` folder:

- CSRD generated on a set of 4 PDF documents:
  - the [annual report](https://www.bouygues.com/wp-content/uploads/2025/03/ri_bouygues_2025_web.pdf) from 2025 of Bouygues
  - the [annual report](https://www.bnains.org/archives/communiques/LVMH/20250325_Document_d_enregistrement_universel_2024_LVMH.pdf) from 2024 of LVMH,
  - the [annual report](https://www.nestle.com/sites/default/files/2025-02/annual-review-2024-en.pdf) from 2024 of Nestle,
  - the [non financial report](https://www.nestle.com/sites/default/files/2025-02/non-financial-statement-2024.pdf) from 2024 of Nestle.
- RED generated on 30 short documents from BBC.
