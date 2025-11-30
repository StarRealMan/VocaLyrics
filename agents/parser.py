"""Parser agent that converts MIDI files into structured JSON."""

from __future__ import annotations

from typing import Any, Optional

from core.context import ConversationContext
from core.task import AgentArtifact, AgentResult, Task
from utils import midi as midi_utils
from .base import BaseAgent


class ParserAgent(BaseAgent):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="parser", description="MIDI parser", **kwargs)

    def run(
        self,
        task: Task,
        context: ConversationContext,
        midi_path: Optional[str] = None,
        **_: Any,
    ) -> AgentResult:
        midi_path = midi_path or context.get_attachment("midi_path")
        if not midi_path:
            return AgentResult(content="", error="Parser requires midi_path input")

        try:
            midi_dict = midi_utils.parse_midi(midi_path)
        except FileNotFoundError:
            return AgentResult(content="", error=f"MIDI file not found: {midi_path}")
        except Exception as exc:
            return AgentResult(content="", error=f"Failed to parse MIDI: {exc}")

        note_count = len(midi_dict.get("notes", []))
        summary = (
            f"Parsed MIDI '{midi_path}' with {note_count} notes. "
            f"Time signature {midi_dict['meta'].get('time_signature')} and BPM {midi_dict['meta'].get('bpm')}"
        )

        return AgentResult(
            content=summary,
            artifacts=[AgentArtifact(kind="midi", payload=midi_dict)],
        )
