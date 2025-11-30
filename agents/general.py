"""General-purpose fallback agent."""

from __future__ import annotations

from typing import Any, Optional

from core.context import ConversationContext
from core.task import AgentResult, Task
from .base import BaseAgent


GENERAL_PROMPT = """
You are a friendly, knowledgeable assistant who can discuss VOCALOID culture,
music trends, and creative tips.

Inputs arrive as JSON with:
{
	"brief": "primary question or concern (required)",
	"prompt": "verbatim user wording" | null
}

Output format:
- Answer in 2-4 short paragraphs.
- Use inline bullet lists when sharing resources or tips.
- Cite concrete song/producer examples when available.
"""


class GeneralAgent(BaseAgent):
	def __init__(self, **kwargs: Any) -> None:
		super().__init__(name="general", description="General VOCALOID expert", **kwargs)

	def run(
		self,
		task: Task,
		context: ConversationContext,
		brief: Optional[str] = None,
		**_: Any,
	) -> AgentResult:
		user_prompt = brief or self._recent_user_prompt(context) or task.description
		messages = [
			{"role": "system", "content": GENERAL_PROMPT},
			{"role": "user", "content": user_prompt},
		]
		reply = self._chat(messages, temperature=0.4, max_tokens=600)
		return AgentResult(content=reply)

