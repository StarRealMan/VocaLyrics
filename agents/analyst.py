import os
import json
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from core.context import Context
from core.task import Task
from agents.base import Agent
from utils.client import init_openai_client


class AnalystInput(BaseModel):
    """Analyst 所需的输入参数模型。"""

    target_text: Optional[str] = None
    data_key: Optional[str] = None

# 定义 Analyst 的输出结构
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
    
    def __init__(self):
        super().__init__(name="Analyst", description="Analyzes lyrics, style, emotions, and imagery.")
        load_dotenv()
        self.client = init_openai_client()
        self.model = os.getenv("OPENAI_API_MODEL", "gpt-5.1")

    def run(self, context: Context, task: Task) -> Any:
        """
        执行分析任务
        """
        params = AnalystInput(**task.input_params)
        target_text = params.target_text
        data_key = params.data_key
        
        content_to_analyze = ""
        
        # 1. 获取待分析内容
        if target_text:
            content_to_analyze = target_text
        elif data_key:
            data = context.get_memory(data_key)
            if not data:
                raise ValueError(f"Data key '{data_key}' not found in shared memory.")
            
            # 处理从 Retriever 返回的数据结构
            if isinstance(data, list):
                # 假设是歌曲列表，提取歌词或摘要
                # 这里简化处理，将所有 payload 转为字符串
                # 实际场景中可能需要更精细的提取，比如只提取 'lyrics' 字段
                formatted_data = []
                for item in data:
                    payload = item.get("payload", {})
                    # 尝试提取歌词，如果没有则使用 metadata
                    lyrics = payload.get("lyrics", "")
                    title = payload.get("defaultName", "Unknown Song")
                    producer = payload.get("producerNames", [])
                    formatted_data.append(f"Title: {title}\nProducers: {producer}\nLyrics: {lyrics[:500]}...") # 截断以防过长
                content_to_analyze = "\n\n".join(formatted_data)
            else:
                content_to_analyze = str(data)
        else:
            raise ValueError("Analyst requires either 'target_text' or 'data_key' parameter.")

        if not content_to_analyze.strip():
             return "No content to analyze."

        # 2. 调用 LLM 进行分析
        self.logger.info(f"Analyzing content (length: {len(content_to_analyze)})...")
        analysis_result = self._perform_analysis(content_to_analyze)
        
        # 3. 保存结果
        # 将 Pydantic 对象转为 dict 保存，方便序列化和后续 Agent 读取
        result_dict = analysis_result.model_dump()
        self._save_to_memory(context, task, result_dict)
        
        return result_dict

    def _perform_analysis(self, content: str) -> AnalysisResult:
        system_prompt = """
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
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Content to analyze:\n{content}"}
                ],
                response_format=AnalysisResult
            )
            return response.choices[0].message.parsed
        except Exception as e:
            # Fallback if structured output fails (though parse() usually handles it)
            self.logger.error(f"Error during analysis: {e}")
            # Return a dummy result to prevent crash
            return AnalysisResult(
                summary="Analysis failed.",
                themes=[],
                emotions=[],
                imagery=[],
                style_description="Error",
                search_query_suggestion=None
            )
