"""Retriever agent for Qdrant song/chunk search."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import QdrantClient

from core.context import ConversationContext
from core.task import AgentArtifact, AgentResult, Task
from utils import client as client_utils
from utils import query as query_utils
from .base import BaseAgent


FILTER_STRING_FIELDS = {"name", "culture"}
FILTER_SEQUENCE_FIELDS = {
	"producers_any",
	"producers_all",
	"producers_must",
	"vsingers_any",
	"vsingers_all",
	"vsingers_must",
	"tags_any",
	"tags_all",
}
FILTER_INT_FIELDS = {
	"producers_min",
	"producers_max",
	"vsingers_min",
	"vsingers_max",
	"year_min",
	"year_max",
	"month_min",
	"month_max",
	"favorite_min",
	"favorite_max",
	"length_min",
	"length_max",
}
FILTER_FLOAT_FIELDS = {"rating_min", "rating_max"}
ALLOWED_FILTER_FIELDS = (
	FILTER_STRING_FIELDS
	| FILTER_SEQUENCE_FIELDS
	| FILTER_INT_FIELDS
	| FILTER_FLOAT_FIELDS
)

FILTER_INFERENCE_PROMPT = """
You convert VOCALOID search requests into two parts:
1) `filters` – structured Qdrant filter JSON.
2) `semantic_query` – the remaining natural-language intent with all filter hints removed.

Allowed filter keys (omit any you cannot infer):
- name (string)
- producers_any | producers_all | producers_must (string lists)
- producers_min | producers_max (ints, count bounds)
- vsingers_any | vsingers_all | vsingers_must (string lists)
- vsingers_min | vsingers_max (ints)
- tags_any | tags_all (string lists)
- year_min | year_max (ints)
- month_min | month_max (ints, 1-12)
- culture (string)
- rating_min | rating_max (floats 0-5)
- favorite_min | favorite_max (ints)
- length_min | length_max (ints, seconds)

Rules:
- `semantic_query` must REMOVE any explicit years, months, producers, vsingers, tags, rating or other filters that you already put inside `filters`. It should retain only the musical/lyrical idea (mood, theme, story beats, instrumentation, etc.).
- Use official VocaDB names for vsingers/producers. Convert aliases, e.g.
  * "初音未来" / "Hatsune Miku" -> "初音ミク"
  * "Miku" -> "初音ミク"
  * "匹诺曹P" / "PinocchioP" -> "ピノキオピー"
  * "镜音リン" / "Rin Kagamine" -> "鏡音リン"
- Keep numbers as numbers, not strings.
- If no filter is implied, set `filters` to an empty object `{}`.
- Prefer concise tag terms (e.g., "happy", "ballad").
- Output JSON ONLY with this schema:
{
	"semantic_query": "text...",
	"filters": { ... }
}

Example:
{
	"semantic_query": "bittersweet ballads about longing",
	"filters": {
		"vsingers_any": ["初音ミク"],
		"tags_any": ["sad", "ballad"],
		"year_min": 2015
	}
}
"""


class RetrieverAgent(BaseAgent):
	def __init__(
		self,
		qdrant_client: QdrantClient,
		default_collection: str = client_utils.SONG_COLLECTION_NAME,
		**kwargs: Any,
	) -> None:
		super().__init__(name="retriever", description="Qdrant retriever", **kwargs)
		self.qdrant = qdrant_client
		self.default_collection = default_collection

	def run(
		self,
		task: Task,
		context: ConversationContext,
		query_text: Optional[str] = None,
		top_k: int = 8,
		pure_payload: Optional[bool] = None,
		level: Optional[str] = None,
		**kwargs: Any,
	) -> AgentResult:
		del task, kwargs
		collection_name = self._resolve_collection(level or context.get_attachment("retrieval_level"))
		semantic_query_raw = query_text or context.get_attachment("semantic_query")
		payload_only = pure_payload if pure_payload is not None else bool(context.get_attachment("pure_payload"))
		if not semantic_query_raw:
			return AgentResult(content="", error="Retriever requires query_text input")

		inferred_filters, cleaned_query = self._infer_filters_via_llm(semantic_query_raw, context)
		semantic_query = cleaned_query or semantic_query_raw

		try:
			results = query_utils.query(
				qdrant_client=self.qdrant,
				openai_client=None if payload_only else self.openai,
				top_k=top_k,
				query_text=semantic_query,
				pure_payload=payload_only,
				collection=collection_name,
				**inferred_filters,
			)
		except Exception as exc:
			return AgentResult(content="", error=f"Retriever error: {exc}")

		normalized = self._normalize_points(results)
		summary = self._summarize(normalized)
		
		return AgentResult(
			content=summary,
			citations=[{"collection": collection_name, "count": len(normalized)}],
			artifacts=[AgentArtifact(kind="documents", payload=normalized)],
		)

	# ------------------------------------------------------------------
	def _infer_filters_via_llm(
		self,
		query_text: Optional[str],
		context: ConversationContext,
	) -> Tuple[Dict[str, Any], Optional[str]]:
		if not query_text or not self.openai:
			return {}, None
		recent = self._format_messages(context.iter_messages(limit=4))
		messages = [
			{"role": "system", "content": FILTER_INFERENCE_PROMPT},
			{
				"role": "user",
				"content": (
					f"Query text: {query_text}\n"
					f"Recent context (optional):\n{recent or '[none]'}"
				),
			},
		]
		try:
			raw = self._chat(messages, temperature=0.1, max_tokens=350)
		except Exception:
			return {}, None
		parsed = self._safe_json_loads(raw)
		if not isinstance(parsed, dict):
			return {}, None
		filters = self._sanitize_filters(parsed.get("filters"))
		semantic_query = parsed.get("semantic_query")
		if isinstance(semantic_query, str):
			semantic_query = semantic_query.strip() or None
		else:
			semantic_query = None
		return filters, semantic_query

	# ------------------------------------------------------------------
	def _resolve_collection(self, level: Optional[str]) -> str:
		if level == "chunk":
			return client_utils.CHUNK_COLLECTION_NAME
		if level == "song":
			return client_utils.SONG_COLLECTION_NAME
		return self.default_collection

	def _normalize_points(self, points: List[Any]) -> List[Dict[str, Any]]:
		normalized: List[Dict[str, Any]] = []
		for point in points or []:
			normalized.append(
				{
					"id": getattr(point, "id", None),
					"score": getattr(point, "score", None),
					"payload": getattr(point, "payload", {}),
				}
			)
		return normalized

	def _summarize(self, docs: List[Dict[str, Any]]) -> str:
		if not docs:
			return "No matching songs or lyrics were found."
		lines = []
		for doc in docs[:3]:
			payload = doc.get("payload") or {}
			name = payload.get("defaultName") or payload.get("name") or "Unknown title"
			producers = ", ".join(payload.get("producerNames", [])[:2])
			vsingers = ", ".join(payload.get("vsingerNames", [])[:2])
			lines.append(f"• {name} | Prod: {producers or '-'} | Vocal: {vsingers or '-'}")
		return "Top matches:\n" + "\n".join(lines)

	def _sanitize_filters(self, data: Any) -> Dict[str, Any]:
		if not isinstance(data, dict):
			return {}
		sanitized: Dict[str, Any] = {}
		for key, value in data.items():
			if key not in ALLOWED_FILTER_FIELDS:
				continue
			coerced = self._coerce_filter_value(key, value)
			if coerced is not None:
				sanitized[key] = coerced
		return sanitized

	def _coerce_filter_value(self, key: str, value: Any) -> Any:
		if value is None:
			return None
		if isinstance(value, str):
			value = value.strip()
			if not value:
				return None
		if key in FILTER_STRING_FIELDS:
			return str(value)
		if key in FILTER_SEQUENCE_FIELDS:
			return self._ensure_list_of_strings(value)
		if key in FILTER_INT_FIELDS:
			try:
				return int(value)
			except (TypeError, ValueError):
				return None
		if key in FILTER_FLOAT_FIELDS:
			try:
				return float(value)
			except (TypeError, ValueError):
				return None
		return None

	def _ensure_list_of_strings(self, value: Any) -> Optional[List[str]]:
		items: List[str] = []
		if isinstance(value, str):
			items = [value.strip()] if value.strip() else []
		elif isinstance(value, list):
			for entry in value:
				text = str(entry).strip()
				if text:
					items.append(text)
		else:
			return None
		return items or None

