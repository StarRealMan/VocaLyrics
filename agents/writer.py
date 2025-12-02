import os
from typing import Any, Optional
from dotenv import load_dotenv

from core.context import Context
from core.task import Task
from agents.base import Agent
from utils.client import init_openai_client


from pydantic import BaseModel


class WriterInput(BaseModel):
    """Writer 所需的输入参数模型。"""

    topic: str
    source_material_key: Optional[str] = None
    source_material: Optional[str] = None

class Writer(Agent):
    """
    作家 Agent
    
    负责根据上下文信息生成最终的自然语言回复，或者进行创意写作。
    通常作为任务链的最后一步，将结构化数据转化为用户友好的文本。
    """
    
    def __init__(self):
        super().__init__(name="Writer", description="Generates natural language responses, summaries, or creative content based on data.")
        load_dotenv()
        self.client = init_openai_client()
        self.model = os.getenv("OPENAI_API_MODEL", "gpt-5.1")

    def run(self, context: Context, task: Task) -> Any:
        """
        执行写作任务
        """
        params = WriterInput(**task.input_params)
        topic = params.topic
        source_material_key = params.source_material_key
        
        source_content = ""
        if source_material_key:
            data = context.get_memory(source_material_key)
            if data:
                source_content = f"Source Material ({source_material_key}):\n{str(data)}"
            else:
                source_content = f"Source Material ({source_material_key}) was empty or not found."
        
        # 如果没有指定 source_material_key，尝试从 input_params 直接获取 source_material
        if not source_content and params.source_material:
            source_content = f"Source Material:\n{params.source_material}"

        system_prompt = """
You are a creative and helpful Writer Agent for a Vocaloid Lyrics Analysis System.
Your goal is to write a response for the user based on the provided source material and topic.

GUIDELINES:
- Tone: Helpful, knowledgeable, and engaging.
- If summarizing search results: List the songs clearly (Title, Producer) and explain why they match the user's request if possible.
- If writing a story or world setting: Be creative and use the lyrics/themes provided.
- If the source material is empty or indicates no results, politely inform the user.
- Language: Use the same language as the user's request (mostly Chinese based on context).
"""

        user_prompt = f"Topic/Instruction: {topic}\n\n{source_content}"

        self.logger.debug(f"Generating content for topic: {topic}...")
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        result = response.choices[0].message.content
        
        # 保存结果
        self._save_to_memory(context, task, result)
        
        return result
