import os
from typing import Any
from dotenv import load_dotenv

from core.context import Context
from core.task import Task
from agents.base import Agent


class GeneralAgent(Agent):
    """
    通用 Agent
    
    负责处理与 Vocaloid、歌词或音乐主题无关的通用查询。
    它会温和地回答用户，并尝试将话题引导回 Vocaloid 相关主题。
    """
    
    def __init__(self, openai_client):
        super().__init__(name="General", description="Handles general queries unrelated to Vocaloid or specific tools.")
        load_dotenv()
        self.openai_client = openai_client
        self.model = os.getenv("OPENAI_API_MODEL", "gpt-5.1")

    def run(self, context: Context, task: Task) -> Any:
        """
        执行通用对话任务
        """
        params = task.input_params
        query = params.query
        
        system_prompt = self._build_system_prompt()

        self.logger.debug(f"Handling general query: {query}")
        
        response = self.openai_client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        )

        result = response.output_text
        
        # 保存结果
        self._save_to_memory(context, task, result)
        
        return result

    def _build_system_prompt(self) -> str:
        return """
You are the General Agent for the VocaLyrics System.
Your role is to handle user queries that are NOT directly related to Vocaloid, lyrics, or music analysis.

GUIDELINES:
1. Answer the user's question gently and briefly.
2. Try to creatively associate the user's topic with Vocaloid, music, lyrics, or creativity.
3. Suggest 1-2 follow-up questions the user might want to ask related to the system's core capabilities (Vocaloid/Lyrics).
4. Maintain a helpful and polite tone.

Example:
User: "What is the weather like?"
Response: "I'm not sure about the real-time weather, but rainy days often remind me of the song 'Ame to Petra'. Speaking of which, would you like to analyze the lyrics of a rain-themed Vocaloid song?"
"""
