"""Composer agent responsible for lyric writing and transformations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.context import ConversationContext
from core.task import AgentArtifact, AgentResult, Task
from .base import BaseAgent


COMPOSER_PROMPT = """
You are a professional VOCALOID lyricist.

Inputs arrive as JSON with the following fields:
{
	"brief": "overall assignment (required)",
	"style": "adjectives or genre tags (optional)",
	"references": [ {"payload": {...}} ],
	"seed_lyrics": "existing lines to remix" | null,
	"midi_summary": {"meta": {...}, "notes": [...] } | null
}

Always acknowledge any references or seed lyrics you receive. If `midi_summary`
exists, align syllable density with BPM/time signature.

Output format (plain text):
- Organize lyrics into labeled sections such as Verse, Pre, Chorus, Bridge.
- Keep lines short (6-11 syllables) and mix romaji/kanji when it improves flow.
- Close with a short "Notes:" section summarizing how the brief/style were satisfied.
"""


class ComposerAgent(BaseAgent):
	def __init__(self, **kwargs: Any) -> None:
		super().__init__(name="composer", description="Lyric composer", **kwargs)

	def run(
		self,
		task: Task,
		context: ConversationContext,
		brief: Optional[str] = None,
		style: Optional[str] = None,
		references: Optional[List[Dict[str, Any]]] = None,
		seed_lyrics: Optional[str] = None,
		midi_summary: Optional[Dict[str, Any]] = None,
		**_: Any,
	) -> AgentResult:
		user_prompt = brief or self._recent_user_prompt(context) or task.description
		reference_text = self._format_references(references)
		midi_text = self._summarize_midi(midi_summary)

		messages = [
			{"role": "system", "content": COMPOSER_PROMPT},
			{
				"role": "user",
				"content": (
					f"Brief: {user_prompt}\n"
					f"Style cues: {style or 'inherit user tone'}\n"
					f"Seed lyrics (optional):\n{seed_lyrics or 'N/A'}\n\n"
					f"References:\n{reference_text or 'N/A'}\n\n"
					f"MIDI summary:\n{midi_text or 'N/A'}"
				),
			},
		]
		lyrics = self._chat(messages, temperature=0.6, max_tokens=900)

		artifacts = []
		if references:
			artifacts.append(AgentArtifact(kind="references", payload=references))
		if midi_summary:
			artifacts.append(AgentArtifact(kind="midi", payload=midi_summary))

		return AgentResult(content=lyrics, artifacts=artifacts)

	def _format_references(self, refs: Optional[List[Dict[str, Any]]]) -> str:
		if not refs:
			return ""
		lines = []
		for ref in refs[:3]:
			payload = ref.get("payload", {})
			lines.append(
				f"- {payload.get('defaultName', ref.get('title', 'Unknown'))}"
				f" | Mood: {', '.join(payload.get('tagNames', [])[:4])}"
			)
		return "\n".join(lines)

	def _summarize_midi(self, midi_summary: Optional[Dict[str, Any]]) -> str:
		if not midi_summary:
			return ""
		meta = midi_summary.get("meta", {})
		bpm = meta.get("bpm", "?")
		ts = meta.get("time_signature", "?")
		notes = midi_summary.get("notes", [])
		return f"BPM {bpm}, Time Signature {ts}, {len(notes)} notes"

