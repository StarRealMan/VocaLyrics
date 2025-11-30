"""Analyst agent for lyrical/style analysis tasks."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.context import ConversationContext
from core.task import AgentArtifact, AgentResult, Task
from .base import BaseAgent


ANALYST_PROMPT = """
You are a VOCALOID music analyst.

Inputs arrive as JSON with:
{
	"focus": "analysis target",
	"references": [ {"payload": {...}}, ... ] // optional list of song payloads
}

Always read `focus` first, then use `references` to cite concrete evidence. If the
list is empty, rely on the latest conversation context but still analyze the topic.

Output format (plain text, no JSON):
Summary:
- one or two sentences describing the overall style.

Style Markers:
- bullet list calling out imagery, vocal traits, instrumentation, tempo, etc.

Actionable Ideas:
- bullet list translating the analysis into concrete guidance for writers/composers.
"""


class AnalystAgent(BaseAgent):
	def __init__(self, **kwargs: Any) -> None:
		super().__init__(name="analyst", description="Lyrics and style analyst", **kwargs)

	def run(
		self,
		task: Task,
		context: ConversationContext,
		focus: Optional[str] = None,
		references: Optional[List[Dict[str, Any]]] = None,
		**_: Any,
	) -> AgentResult:
		reference_text = self._format_references(references)
		user_prompt = focus or self._recent_user_prompt(context) or task.description

		messages = [
			{"role": "system", "content": ANALYST_PROMPT},
			{
				"role": "user",
				"content": (
					f"Focus: {user_prompt}\n\nReferences:\n{reference_text or 'N/A'}"
				),
			},
		]
		analysis = self._chat(messages, temperature=0.45, max_tokens=700)

		artifacts = []
		if references:
			artifacts.append(AgentArtifact(kind="references", payload=references))

		return AgentResult(content=analysis, artifacts=artifacts)

	def _format_references(self, refs: Optional[List[Dict[str, Any]]]) -> str:
		if not refs:
			return ""
		flat_items: List[Dict[str, Any]] = []
		for ref in refs:
			if isinstance(ref, dict) and "payload" in ref:
				payload = ref["payload"]
				if isinstance(payload, list):
					flat_items.extend(payload)
				elif isinstance(payload, dict):
					flat_items.append(payload)
			elif isinstance(ref, list):
				flat_items.extend([entry for entry in ref if isinstance(entry, dict)])
			elif isinstance(ref, dict):
				flat_items.append(ref)
		if not flat_items:
			return ""
		lines = []
		for item in flat_items[:5]:
			name = item.get("defaultName") or item.get("title") or "Unknown"
			tags = item.get("tagNames") or item.get("tags") or []
			if isinstance(tags, list):
				tag_text = ", ".join(tags[:5]) if tags else "N/A"
			else:
				tag_text = str(tags)
			lines.append(f"- {name} | Tags: {tag_text}")
		return "\n".join(lines)

