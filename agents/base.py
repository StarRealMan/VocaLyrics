"""Base classes and helpers for all VocaLyrics agents."""

from __future__ import annotations

import abc
import json
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI

from core.context import ConversationContext, Message
from core.task import AgentResult, Task


DEFAULT_CHAT_MODEL = "gpt-4o-mini"


class BaseAgent(abc.ABC):
	"""Abstract agent with shared OpenAI chat utilities."""

	def __init__(
		self,
		name: str,
		description: str,
		openai_client: Optional[OpenAI] = None,
		default_model: str = DEFAULT_CHAT_MODEL,
		temperature: float = 0.35,
		max_output_tokens: int = 800,
	) -> None:
		self.name = name
		self.description = description
		self.openai = openai_client
		self.default_model = default_model
		self.temperature = temperature
		self.max_output_tokens = max_output_tokens

	# ------------------------------------------------------------------
	@abc.abstractmethod
	def run(self, task: Task, context: ConversationContext, **kwargs: Any) -> AgentResult:
		"""Execute the agent logic for the provided task."""

	# ------------------------------------------------------------------
	# Helper utilities shared by subclasses
	# ------------------------------------------------------------------
	def _chat(
		self,
		messages: List[Dict[str, str]],
		model: Optional[str] = None,
		temperature: Optional[float] = None,
		max_tokens: Optional[int] = None,
	) -> str:
		if not self.openai:
			raise RuntimeError(f"Agent {self.name} requires an OpenAI client")
		response = self.openai.chat.completions.create(
			model=model or self.default_model,
			messages=messages,
			temperature=temperature if temperature is not None else self.temperature,
			max_tokens=max_tokens or self.max_output_tokens,
		)
		return response.choices[0].message.content.strip()

	def _recent_user_prompt(self, context: ConversationContext) -> str:
		for message in reversed(context.messages):
			if message.role == "user":
				return message.content
		return ""

	def _format_messages(self, messages: Iterable[Message]) -> str:
		"""Turn a few recent messages into a lightweight transcript string."""
		parts: List[str] = []
		for message in messages:
			speaker = message.name or message.role
			parts.append(f"[{speaker}] {message.content}")
		return "\n".join(parts)

	def _safe_json_loads(self, blob: str) -> Any:
		try:
			return json.loads(blob)
		except Exception:
			return None

