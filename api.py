from __future__ import annotations

import os
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field

from onto_pitfalls_lib import OntologyPatternToolkit
from onto_pitfalls_lib.cli import _add_human_descriptions, _group_results_by_category
from onto_pitfalls_lib.utils import parse_pattern_selection


app = FastAPI(
	title="Ontology Pitfalls Detector API",
	description="HTTP API for ontology pitfall analysis (P1.1-P4.7)",
	version="1.0.0",
)


class AnalyzeRequest(BaseModel):
	ontology: str = Field(..., description="Path to ontology .ttl file")
	patterns: list[str] = Field(
		default_factory=lambda: ["all"],
		description="Pitfalls to run, e.g. ['P1.1', 'P2.3'] or ['all']",
	)
	model: str = Field(
		default="all-MiniLM-L6-v2",
		description="SentenceTransformer model for semantic checks",
	)
	include_grouped_results: bool = Field(
		default=True,
		description="Whether to include grouped_results in the response",
	)
	include_descriptions: bool = Field(
		default=True,
		description="Whether to attach human-readable descriptions to each detection",
	)


def _resolve_patterns(raw_patterns: list[str]) -> list[str]:
	return parse_pattern_selection(
		raw_patterns,
		OntologyPatternToolkit.available_patterns(),
		normalizer=OntologyPatternToolkit.normalize_pitfall_id,
	)


@app.get("/health")
def health() -> dict[str, str]:
	return {"status": "ok"}


@app.get("/")
def root() -> dict[str, Any]:
	return {
		"name": "Ontology Pitfalls Detector API",
		"version": "1.0.0",
		"docs": "/docs",
		"health": "/health",
		"taxonomy": "/taxonomy",
		"analyze": "/analyze",
	}



@app.get("/taxonomy")
def taxonomy() -> dict[str, Any]:
	return {
		"pitfalls": OntologyPatternToolkit.pitfall_taxonomy(),
		"available_patterns": OntologyPatternToolkit.available_patterns(),
	}


@app.post(
	"/analyze",
	summary="Analyze ontology pitfalls",
	description="Run one or more pitfall checks (P1.1-P4.7) against a .ttl ontology file.",
)
def analyze(
	payload: AnalyzeRequest = Body(
		...,
		openapi_examples={
			"quick_start": {
				"summary": "Quick start",
				"description": "Run three pitfalls on a sample ontology.",
				"value": {
					"ontology": "data/red-otkv3.ttl",
					"patterns": ["P1.1", "P2.3", "P4.6"],
					"model": "all-MiniLM-L6-v2",
					"include_grouped_results": True,
					"include_descriptions": True,
				},
			},
			"all_pitfalls": {
				"summary": "Run all pitfalls",
				"description": "Execute the full taxonomy using the default model.",
				"value": {
					"ontology": "data/red-otkv3.ttl",
					"patterns": ["all"],
				},
			},
		},
	)
) -> dict[str, Any]:
	try:
		toolkit = OntologyPatternToolkit(payload.ontology, model_name=payload.model)
		selected_pitfalls = _resolve_patterns(payload.patterns)
		results = toolkit.run_patterns(selected_pitfalls)
		if payload.include_descriptions:
			_add_human_descriptions(results)
	except FileNotFoundError as exc:
		raise HTTPException(status_code=404, detail=str(exc)) from exc
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc
	except Exception as exc:  # pragma: no cover
		raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

	response: dict[str, Any] = {
		"metadata": toolkit.metadata(),
		"selected_pitfalls": selected_pitfalls,
		"results": results,
	}

	if payload.include_grouped_results:
		response["grouped_results"] = _group_results_by_category(selected_pitfalls, results)

	return response


if __name__ == "__main__":
	import uvicorn

	host = os.getenv("API_HOST", "0.0.0.0")
	port = int(os.getenv("API_PORT", "8000"))
	reload_enabled = os.getenv("API_RELOAD", "false").strip().lower() in {"1", "true", "yes"}

	uvicorn.run("api:app", host=host, port=port, reload=reload_enabled)

