import os
from typing import Any
from dotenv import load_dotenv

from core.context import Context
from core.task import Task
from agents.base import Agent


class Lyricist(Agent):
    """Lyricist Agent

    负责根据风格、主题、MIDI 结构等信息生成或续写歌词。
    """

    def __init__(self, openai_client):
        super().__init__(name="Lyricist", description="Composes or rewrites lyrics based on style, theme, and MIDI structure.")
        self.openai_client = openai_client
        load_dotenv()
        self.model = os.getenv("OPENAI_API_MODEL", "gpt-5.1")

    def run(self, context: Context, task: Task) -> Any:
        """执行作词任务"""
        params = task.input_params

        style = params.style
        theme = params.theme
        midi_key = params.midi_key
        source_key = params.source_key
        source = params.source

        if midi_key:
            midi_data = context.get_memory(midi_key)
            if midi_data:
                midi_structure = midi_data.get("structure", {})
            else:
                raise ValueError(f"MIDI key '{midi_key}' not found in shared memory.")
        
        source_content = ""
        if source_key:
            data = context.get_memory(source_key)
            if data:
                source_content += f"Source from ({source_key}):\n{str(data)}"
            else:
                raise ValueError(f"Source key '{source_key}' not found in shared memory.")
        if source:
            source_content += f"\nSource provided by planner:\n{source}"
        
        if not style and not theme and not source_content and not midi_key:
            raise ValueError("Lyricist requires at least one of 'style', 'theme', 'source_key', 'source', or 'midi_key' parameter.")

        system_prompt = self._build_system_prompt()

        user_prompt = self._build_user_prompt(style, theme, midi_structure, source_content)
        self.logger.debug(f"Composing lyrics with style='{style}', theme='{theme}'...")

        response = self.openai_client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        lyrics = response.output_text

        # 保存结果
        self._save_to_memory(context, task, lyrics)
        return lyrics

    def _build_user_prompt(
        self,
        style: str,
        theme: str,
        midi_structure: dict,
        source_content: str,
    ) -> str:
        parts = []
        if style:
            parts.append(f"Style: {style}")
        if theme:
            parts.append(f"Theme: {theme}")
        if midi_structure:
            parts.append(f"MIDI structure (for reference, optional):\n{midi_structure}")
        if source_content:
            parts.append(f"Existing draft lyrics (to refine or continue):\n{source_content}")
        else:
            parts.append("No existing lyrics. Please write from scratch.")

        parts.append("Please output the complete lyrics.")
        return "\n\n".join(parts)

    def _build_system_prompt(self) -> str:
        return """
You are a professional Vocaloid lyricist.
You write singable, structured lyrics that can fit a typical J-pop style song structure.

Guidelines:
- Respect the given style and theme.
- If MIDI structure is provided, roughly align verse/chorus length with the structure hints.
- If base_lyrics is provided, treat it as draft: keep its core imagery/theme but improve flow and structure.
- Prefer Japanese lyrics if style suggests it, otherwise follow the language implied by the theme/base_lyrics.
- Output only the final lyrics text, no explanations.
"""
