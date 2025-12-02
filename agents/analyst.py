import os
from typing import Any, List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from core.context import Context
from core.task import Task
from agents.base import Agent


class AnalysisResult(BaseModel):
    summary: str = Field(..., description="A concise summary of the analysis.")
    themes: List[str] = Field(..., description="Key themes identified in the lyrics.")
    emotions: List[str] = Field(..., description="Emotions conveyed by the song(s).")
    imagery: List[str] = Field(..., description="Specific imagery or keywords found in the lyrics.")
    style_description: str = Field(..., description="A description of the lyrical style (e.g., abstract, narrative, dark).")
    search_query_suggestion: Optional[str] = Field(None, description="A suggested search query string for finding similar songs based on this analysis.")

class Analyst(Agent):
    """
    分析师 Agent
    
    负责分析歌词、风格、情感等。
    输入可以是直接的文本，也可以是 Context 中存储的检索结果（歌曲列表）。
    """
    
    def __init__(self, openai_client):
        super().__init__(name="Analyst", description="Analyzes lyrics, style, emotions, and imagery.")
        self.openai_client = openai_client
        load_dotenv()
        self.model = os.getenv("OPENAI_API_MODEL", "gpt-5.1")

    def run(self, context: Context, task: Task) -> Any:
        """
        执行分析任务
        """
        params = task.input_params
        retrieved_keys = params.get("retrieved_keys")
        source_key = params.get("source_key")
        source = params.get("source")

        source_content = ""
        
        # 1. 获取待分析内容
        if source_key:
            data = context.get_memory(source_key)
            if data:
                if retrieved_keys:
                    source_content += "Source from retriever"
                    for key in retrieved_keys:
                        item = data["payload"].get(key)
                        if item:
                            source_content += f"\n{key.upper()}:\n{str(item)}"
                else:
                    source_content += f"Source from ({source_key}):\n{str(data)}"
            else:
                raise ValueError(f"Source key '{source_key}' not found in shared memory.")
        
        if source:
            source_content += f"\nSource provided by planner:\n{source}"
        
        if not source_content:
            raise ValueError(f"Analyst requires either 'retrieved_key', 'source_key' or 'source' parameter.")

        if not source_content.strip():
             return "No content to analyze."

        # 2. 调用 LLM 进行分析
        self.logger.debug(f"Analyzing content (length: {len(source_content)})...")
        analysis_result = self._perform_analysis(source_content)
        
        # 3. 保存结果
        # 将 Pydantic 对象转为 dict 保存，方便序列化和后续 Agent 读取
        result_dict = analysis_result.model_dump()
        self._save_to_memory(context, task, result_dict)
        
        return result_dict

    def _perform_analysis(self, content: str) -> AnalysisResult:
        system_prompt = self._build_system_prompt()

        response = self.openai_client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Content to analyze:\n{content}"}
            ],
            text_format=AnalysisResult
        )
        parsed: AnalysisResult = response.output_parsed

        return parsed

    def _build_system_prompt(self) -> str:
        return """
You are an expert Vocaloid Lyrics Analyst.
Your task is to analyze the provided song lyrics or metadata and extract key stylistic features.

You must output a JSON object matching the following structure:
{
    "summary": "Brief summary of the content",
    "themes": ["theme1", "theme2"],
    "emotions": ["emotion1", "emotion2"],
    "imagery": ["image1", "image2"],
    "style_description": "Description of the writing style",
    "search_query_suggestion": "A string of keywords derived from the analysis that can be used to search for SIMILAR songs in a vector database."
}

For 'search_query_suggestion', focus on concrete imagery and emotional keywords found in the lyrics, rather than abstract genre names. 
Example: Instead of "sad rock song", use "tears, rain, falling, dark room, screaming".
"""