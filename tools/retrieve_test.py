#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
from typing import Any, Sequence

from utils.client import (
	SONG_COLLECTION_NAME,
	CHUNK_COLLECTION_NAME,
	init_openai_client,
	init_qdrant_client_and_collections,
)
from utils.query import (
	EMBEDDING_DIM,
	query
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Run a one-off query against the Qdrant instance to verify the "
			"helper in `data.query` works as expected."
		)
	)
	parser.add_argument(
		"--query_text",
		help="Free-form text for semantic search; requires OpenAI embeddings.",
	)
	parser.add_argument(
		"--top_k",
		type=int,
		default=5,
		help="Number of results to retrieve.",
	)
	parser.add_argument(
		"--query_chunks",
		action="store_true",
		help="Query chunk-level collection.",
	)
	parser.add_argument(
		"--name",
		type=str,
		help="Song name filter (substring match).",
	)
	parser.add_argument(
		"--producers_any",
		nargs="+",
		help="Producer names (OR).",
	)
	parser.add_argument(
		"--producers_all",
		nargs="+",
		help="Producer names (AND).",
	)
	parser.add_argument(
		"--producers_must",
		nargs="+",
		help="Producer names (MUST).",
	)
	parser.add_argument(
		"--vsingers_any",
		nargs="+",
		help="Vsinger names (OR).",
	)
	parser.add_argument(
		"--vsingers_all",
		nargs="+",
		help="Vsinger names (AND).",
	)
	parser.add_argument(
		"--vsingers_must",
		nargs="+",
		help="Vsinger names (MUST).",
	)
	parser.add_argument(
		"--tags_any",
		nargs="+",
		help="Tag names (OR).",
	)
	parser.add_argument(
		"--tags_all",
		nargs="+",
		help="Tag names (AND).",
	)
	parser.add_argument(
		"--year_min",
		type=int,
		help="Minimum release year (inclusive).",
	)
	parser.add_argument(
		"--year_max",
		type=int,
		help="Maximum release year (inclusive).",
	)
	parser.add_argument(
		"--month_min",
		type=int,
		help="Minimum release month (1-12).",
	)
	parser.add_argument(
		"--month_max",
		type=int,
		help="Maximum release month (1-12).",
	)
	parser.add_argument(
		"--culture",
		help="Primary culture code (ja, cn, en, ...).",
	)
	parser.add_argument(
		"--rating_min",
		type=float,
		help="Minimum rating score.",
	)
	parser.add_argument(
		"--rating_max",
		type=float,
		help="Maximum rating score.",
	)
	parser.add_argument(
		"--favorite_min",
		type=int,
		help="Minimum favorite count.",
	)
	parser.add_argument(
		"--favorite_max",
		type=int,
		help="Maximum favorite count.",
	)
	parser.add_argument(
		"--length_min",
		type=int,
		help="Minimum length in seconds.",
	)
	parser.add_argument(
		"--length_max",
		type=int,
		help="Maximum length in seconds.",
	)
	parser.add_argument(
		"--verbose",
		action="store_true",
		help="Print full payloads instead of concise summaries.",
	)
	parser.add_argument(
		"--log_level",
		default="INFO",
		choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
		help="Logging verbosity.",
	)

	return parser.parse_args()


def _none_if_empty(value: Sequence[str] | None):
	return value if value else None


def _format_point(point: Any, verbose: bool = False) -> str:
	payload = getattr(point, "payload", {}) or {}
	summary = {
		"id": getattr(point, "id", None),
		"name": payload.get("name")
		or payload.get("songName")
		or payload.get("defaultName"),
		"vsingers": payload.get("vsingerNames") or payload.get("producerNames"),
		"ratingScore": payload.get("ratingScore"),
		"favoriteCount": payload.get("favoriteCount")
		or payload.get("favoritedTimes"),
		"lengthSeconds": payload.get("length") or payload.get("lengthSeconds"),
		"year": payload.get("year"),
		"tags": payload.get("tagNames"),
	}
	lines = [json.dumps(summary, ensure_ascii=False, indent=2)]
	if verbose:
		lines.append(json.dumps(payload, ensure_ascii=False, indent=2))
	return "\n".join(lines)


def main() -> None:
	args = parse_args()
	logging.basicConfig(
		level=getattr(logging, args.log_level),
		format="[%(levelname)s] %(message)s",
	)

	if args.query_text:
		logging.warning(
			"--pure-payload is set; query_text will only influence payload filters."
		)

	qdrant_client = init_qdrant_client_and_collections(
		embedding_dim=EMBEDDING_DIM,
		chunk_collection_name=CHUNK_COLLECTION_NAME,
		song_collection_name=SONG_COLLECTION_NAME,
	)

	openai_client = None
	if args.query_text:
		openai_client = init_openai_client()

	if args.query_chunks:
		collection = CHUNK_COLLECTION_NAME
	else:
		collection = SONG_COLLECTION_NAME

	results = query(
		qdrant_client=qdrant_client,
		openai_client=openai_client,
		top_k=args.top_k,
		query_text=args.query_text,
		name = args.name,
		producers_any=_none_if_empty(args.producers_any),
		producers_all=_none_if_empty(args.producers_all),
		producers_must=_none_if_empty(args.producers_must),
		vsingers_any=_none_if_empty(args.vsingers_any),
		vsingers_all=_none_if_empty(args.vsingers_all),
		vsingers_must=_none_if_empty(args.vsingers_must),
		tags_any=_none_if_empty(args.tags_any),
		tags_all=_none_if_empty(args.tags_all),
		year_min=args.year_min,
		year_max=args.year_max,
		month_min=args.month_min,
		month_max=args.month_max,
		culture=args.culture,
		rating_min=args.rating_min,
		rating_max=args.rating_max,
		favorite_min=args.favorite_min,
		favorite_max=args.favorite_max,
		length_min=args.length_min,
		length_max=args.length_max,
		collection=collection,
	)

	if not results:
		logging.info("No results returned.")
		return

	logging.info("Received %d results:", len(results))
	for idx, point in enumerate(results, start=1):
		print(f"\nResult #{idx}")
		print(_format_point(point, verbose=args.verbose))


if __name__ == "__main__":
	main()
