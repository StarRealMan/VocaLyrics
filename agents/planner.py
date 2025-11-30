"""Planner agent that decomposes user intent into actionable steps."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from core.context import ConversationContext
from core.task import AgentArtifact, AgentResult, Task
from .base import BaseAgent


PLANNER_INSTRUCTIONS = """
You are the lead planner for a VOCALOID research and lyric-writing assistant.
Produce 1-4 steps. Every step must follow this JSON schema (return a JSON array only):
[
	{
		"id": "step-1",
		"agent": "retriever" | "analyst" | "parser" | "composer" | "writer" | "general",
		"goal": "plain-language objective",
		"inputs": { /* structured parameters for that agent */ }
	}
]

Agent input contracts (fill every field, even if value is null/empty):
- retriever: {
	"query_text": str,
	"level": "song" | "chunk",
	"top_k": int,
	"pure_payload": bool | null
}
- analyst: {
	"focus": str,
	"references": [],
	"references_from": {"step": "step-1", "artifact": "documents"} // to pull docs
}
- parser: {
	"midi_path": str // ensure attachment or explicit path exists
}
- composer: {
	"brief": str,
	"style": str | null,
	"references": [],
	"references_from": {"step": "step-x"},
	"seed_lyrics": str | null,
	"seed_lyrics_from": {"step": "step-y", "field": "content"},
	"midi_summary": {},
	"midi_summary_from": {"step": "step-z", "artifact": "midi", "mode": "last"}
}
- writer: {
	"brief": str,
	"format_hint": str | null,
	"brief_from": {"step": "step-a"}
}
- general: {
	"brief": str,
	"prompt": str | null
}

Reference passing:
- Use the suffix `_from` to copy outputs from earlier steps. Example:
	"references_from": {"step": "step-1", "artifact": "documents"}
	"brief_from": {"step": "step-2", "field": "content"}
- When `artifact` is omitted, text fields default to the prior step's response content. When `artifact` is omitted for `references` it defaults to `documents`; for `midi_summary` it defaults to the latest `midi` artifact.
- Only point to steps that already exist above the current step in the list.

Output rules:
- Return JSON only, no markdown or extra prose.
- Keep numbers/booleans as primitives, never strings.
- Ensure every step has `inputs` defined even when values are empty objects.
"""


class PlannerAgent(BaseAgent):
	def __init__(self, **kwargs: Any) -> None:
		super().__init__(name="planner", description="Task planner", **kwargs)

	def run(
		self,
		task: Task,
		context: ConversationContext,
		system_prompt: Optional[str] = None,
		**_: Any,
	) -> AgentResult:
		recent_history = self._format_messages(context.iter_messages(limit=6))
		user_goal = self._recent_user_prompt(context) or task.description

		messages = [
			{"role": "system", "content": system_prompt or PLANNER_INSTRUCTIONS},
			{
				"role": "user",
				"content": (
					"User goal:\n" + user_goal + "\n\nRecent context:\n" + recent_history
				),
			},
		]
		raw_plan = self._chat(messages, temperature=0.2, max_tokens=500)

		plan_data = self._safe_json_loads(raw_plan)
		steps: List[Dict[str, Any]] = []
		if isinstance(plan_data, list):
			for idx, step in enumerate(plan_data, 1):
				if not isinstance(step, dict):
					continue
				agent_name = step.get("agent", "general")
				step_inputs = step.get("inputs")
				inputs = step_inputs if isinstance(step_inputs, dict) else {}
				if agent_name == "general" and "prompt" not in inputs:
					inputs["prompt"] = user_goal
				steps.append(
					{
						"id": step.get("id") or f"step-{idx}",
						"agent": agent_name,
						"goal": step.get("goal") or step.get("description") or user_goal,
						"inputs": inputs,
					}
				)

		summary_lines = [f"{s['id']}: ({s['agent']}) {s['goal']}" for s in steps]
		summary = "\n".join(summary_lines) if summary_lines else raw_plan

		return AgentResult(
			content=summary,
			artifacts=[AgentArtifact(kind="plan", payload=steps or plan_data or raw_plan)],
		)

