import os
from typing import Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel

from core.context import Context
from core.task import Task, ComposerInput
from agents.base import Agent
from utils.client import init_openai_client


class Composer(Agent):
    """Composer Agent

    负责根据风格、主题、MIDI 结构等信息生成或续写歌词。
    """

    def __init__(self):
        super().__init__(name="Composer", description="Composes or rewrites lyrics based on style, theme, and MIDI structure.")
        load_dotenv()
        self.openai_client = init_openai_client()
        self.model = os.getenv("OPENAI_API_MODEL", "gpt-5.1")

    def run(self, context: Context, task: Task) -> Any:
        """执行作词任务"""
        params = ComposerInput(**task.input_params)

        style = params.style or "Vocaloid J-pop"
        theme = params.theme or ""
        midi_structure = params.midi_structure or {}
        base_lyrics = params.base_lyrics or ""

        system_prompt = self._build_system_prompt()

        user_prompt = self._build_user_prompt(style, theme, midi_structure, base_lyrics)

        self.logger.debug(f"Composing lyrics with style='{style}', theme='{theme}'...")

        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        lyrics = response.choices[0].message.content

        # 保存结果
        self._save_to_memory(context, task, lyrics)
        return lyrics

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

    def _build_user_prompt(
        self,
        style: str,
        theme: str,
        midi_structure: dict,
        base_lyrics: str,
    ) -> str:
        parts = [
            f"Style: {style}",
        ]
        if theme:
            parts.append(f"Theme: {theme}")
        if midi_structure:
            parts.append(f"MIDI structure (for reference, optional):\n{midi_structure}")
        if base_lyrics:
            parts.append(f"Existing draft lyrics (to refine or continue):\n{base_lyrics}")
        else:
            parts.append("No existing lyrics. Please write from scratch.")

        parts.append("Please output the complete lyrics.")
        return "\n\n".join(parts)
