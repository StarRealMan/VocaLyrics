"""Conversation context utilities shared across agents."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class Message:
	"""Single entry in the conversation history."""

	role: str  # "user", "assistant", "system", "tool", etc.
	content: str
	name: Optional[str] = None
	metadata: Dict[str, Any] = field(default_factory=dict)
	timestamp: float = field(default_factory=time.time)


class ConversationContext:
	"""Stateful store that keeps recent conversation and scratch data."""

	def __init__(self, max_messages: int = 64) -> None:
		self.max_messages = max_messages
		self.messages: List[Message] = []
		self.scratchpad: Dict[str, Any] = {}
		self.attachments: Dict[str, Any] = {}

	# ------------------------------------------------------------------
	# Message helpers
	# ------------------------------------------------------------------
	def add_message(
		self,
		role: str,
		content: str,
		name: Optional[str] = None,
		metadata: Optional[Dict[str, Any]] = None,
	) -> None:
		self.messages.append(
			Message(role=role, content=content, name=name, metadata=metadata or {})
		)
		if len(self.messages) > self.max_messages:
			overflow = len(self.messages) - self.max_messages
			self.messages = self.messages[overflow:]

	def add_user_message(self, content: str, **metadata: Any) -> None:
		self.add_message("user", content, metadata=metadata)

	def add_agent_message(self, agent: str, content: str, **metadata: Any) -> None:
		self.add_message("assistant", content, name=agent, metadata=metadata)

	def add_tool_message(self, tool_name: str, content: str, **metadata: Any) -> None:
		self.add_message("tool", content, name=tool_name, metadata=metadata)

	def iter_messages(self, limit: Optional[int] = None) -> Iterable[Message]:
		if limit is None or limit >= len(self.messages):
			return iter(self.messages)
		return iter(self.messages[-limit:])

	def to_openai_messages(self, system_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
		payload: List[Dict[str, Any]] = []
		if system_prompt:
			payload.append({"role": "system", "content": system_prompt})
		for msg in self.messages:
			payload.append(
				{
					"role": msg.role,
					"content": msg.content,
					**({"name": msg.name} if msg.name else {}),
				}
			)
		return payload

	# ------------------------------------------------------------------
	# Scratchpad helpers
	# ------------------------------------------------------------------
	def set_scratch(self, key: str, value: Any) -> None:
		self.scratchpad[key] = value

	def get_scratch(self, key: str, default: Any = None) -> Any:
		return self.scratchpad.get(key, default)

	def clear_scratch(self) -> None:
		self.scratchpad.clear()

	# ------------------------------------------------------------------
	# Attachments (e.g., uploaded MIDI files, filters)
	# ------------------------------------------------------------------
	def set_attachment(self, key: str, value: Any) -> None:
		self.attachments[key] = value

	def get_attachment(self, key: str, default: Any = None) -> Any:
		return self.attachments.get(key, default)

