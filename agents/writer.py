"""Writer agent for non-lyric creative writing tasks."""

from __future__ import annotations

from typing import Any, Optional

from core.context import ConversationContext
from core.task import AgentResult, Task
from .base import BaseAgent


WRITER_PROMPT = """
You are a creative writer who expands VOCALOID-inspired ideas into prose,
worldbuilding notes, summaries, and scripts.

Input JSON schema:
{
	"brief": "topic or assignment (required)",
	"format_hint": "requested format such as outline, diary, script" | null
}

Output format (plain text):
- Begin with a one-sentence logline.
- Follow the requested format when provided; otherwise produce ~2 paragraphs of vivid prose.
- End with a short "Next steps" bullet list suggesting how to continue the narrative.
"""


class WriterAgent(BaseAgent):
	def __init__(self, **kwargs: Any) -> None:
		super().__init__(name="writer", description="Creative writer", **kwargs)

	def run(
		self,
		task: Task,
		context: ConversationContext,
		brief: Optional[str] = None,
		format_hint: Optional[str] = None,
		**_: Any,
	) -> AgentResult:
		user_prompt = brief or self._recent_user_prompt(context) or task.description
		messages = [
			{"role": "system", "content": WRITER_PROMPT},
			{
				"role": "user",
				"content": f"Format: {format_hint or 'freeform'}\nBrief: {user_prompt}",
			},
		]
		prose = self._chat(messages, temperature=0.55, max_tokens=700)
		return AgentResult(content=prose)

