import os
import json
from typing import List, Any, Dict, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from core.context import Context
from core.task import Task
from agents.base import Agent
from utils.client import SONG_COLLECTION_NAME
from utils.query import query

class RetrieverFilter(BaseModel):
    """用于 payload 过滤的条件字典结构。"""

    name: Optional[str] = Field(None, description="Exact match for song name.")
    producers_any: Optional[List[str]] = Field(None, description="Match ANY of these producers.")
    producers_all: Optional[List[str]] = Field(None, description="Match ALL of these producers.")
    producers_must: Optional[List[str]] = Field(None, description="Must be exactly these producers.")
    producers_min: Optional[int] = Field(None, description="Minimum number of producers.")
    producers_max: Optional[int] = Field(None, description="Maximum number of producers.")
    vsingers_any: Optional[List[str]] = Field(None, description="Match ANY of these vocalists.")
    vsingers_all: Optional[List[str]] = Field(None, description="Match ALL of these vocalists.")
    vsingers_must: Optional[List[str]] = Field(None, description="Must be exactly these vocalists.")
    vsingers_min: Optional[int] = Field(None, description="Minimum number of vocalists.")
    vsingers_max: Optional[int] = Field(None, description="Maximum number of vocalists.")
    tagNames: Optional[List[str]] = Field(None, description="Match ANY of these tags (e.g., 'rock', 'sad', 'summer').")
    rating_min: Optional[float] = Field(None, description="Minimum rating score.")
    rating_max: Optional[float] = Field(None, description="Maximum rating score")
    favorite_min: Optional[int] = Field(None, description="Minimum number of favorites.")
    favorite_max: Optional[int] = Field(None, description="Maximum number of favorites.")
    length_min: Optional[int] = Field(None, description="Minimum song length in seconds.")
    length_max: Optional[int] = Field(None, description="Maximum song length in seconds.")
    culture: Optional[str] = Field(None, description="Primary culture code (e.g., 'ja', 'en', 'zh').")
    year_min: Optional[int] = Field(None, description="Minimum year.")
    year_max: Optional[int] = Field(None, description="Maximum year.")
    month_min: Optional[int] = Field(None, description="Minimum month.")
    month_max: Optional[int] = Field(None, description="Maximum month.")

class RetrieverAnalyseResult(BaseModel):
    """LLM 对检索请求解析后的结构化结果。"""

    collection: Literal = Field(default=SONG_COLLECTION_NAME, description="Collection level to query ('vocadb_songs' or 'vocadb_chunks').")
    query_text: Optional[str] = Field(None, description="The semantic search query string, None if using payload filter only")
    top_k: int = Field(default=10, description="Number of results to return.")
    filters: RetrieverFilter = Field(default_factory=RetrieverFilter, description="Payload filter conditions.")

class Retriever(Agent):
    """
    检索者 Agent
    
    负责从 Qdrant 数据库中检索歌曲或歌词片段。
    采用两阶段工作流：
    1. 使用 LLM 将自然语言请求转换为结构化的查询参数 (Query Parsing)。
    2. 执行数据库查询。
    """
    
    def __init__(self, openai_client, qdrant_client):
        super().__init__(name="Retriever", description="Retrieves songs and lyrics from the database based on natural language requests.")
        self.openai_client = openai_client
        self.qdrant_client = qdrant_client
        load_dotenv()
        # 初始化 Qdrant Client，这里假设 collection 已经存在，所以不需要 create_payload_indexes

        self.model = os.getenv("OPENAI_API_MODEL", "gpt-5.1")

    def run(self, context: Context, task: Task) -> Any:
        """
        执行检索任务
        """
        params = task.input_params
        request = params.request

        # 阶段 1: Query Parsing (LLM)
        self.logger.debug(f"Analyzing request: {request}")
        query_params = self._analyze_request(request)
        self.logger.debug(f"Generated query params: {json.dumps(query_params, ensure_ascii=False)}")

        # 阶段 2: Execution
        results = self._execute_search(query_params)
        
        # 结果处理：将 PointStruct 对象转换为字典，以便序列化
        serialized_results = []
        for point in results:
            if query_params.query_text:
                serialized_results.append({
                    "id": point.id,
                    "score": point.score,
                    "payload": point.payload
                })
            else:
                serialized_results.append({
                    "id": point.id,
                    "payload": point.payload
                })
            
        # 如果指定了 output_key，保存结果
        self._save_to_memory(context, task, serialized_results)
        
        return f"Retrieved {len(serialized_results)} items."

    def _analyze_request(self, request: str) -> Dict[str, Any]:
        """
        使用 LLM 分析自然语言请求，生成 utils.query.query 所需的参数。
        """
        system_prompt = self._build_system_prompt()

        response = self.openai_client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request},
            ],
            text_format=RetrieverAnalyseResult,
        )

        parsed: RetrieverAnalyseResult = response.output_parsed

        return parsed

    def _execute_search(self, params: Dict[str, Any]) -> List[Any]:
        """
        调用 utils.query.query 执行实际查询
        """
        collection = params.collection
        query_text = params.query_text

        self.logger.debug(f"Query text: {query_text}")

        top_k = params.top_k
        filters = params.filters
        
        # 将 filters 字典展开作为参数传递给 query 函数
        # 注意：utils.query.query 的参数名与 filters 中的 key 需要对应
        return query(
            qdrant_client=self.qdrant_client,
            openai_client=self.openai_client,
            top_k=top_k,
            query_text=query_text,
            collection=collection,
            **filters
        )
    
    def _build_system_prompt(self) -> str:
        return """
You are an expert Query Parser for a Vocaloid Song Database.
Your goal is to convert a natural language search request into a structured JSON object containing parameters for a database query function.

The database has two collections:
1. "vocadb_songs": Contains full song metadata and lyrics. The vector embeddings are generated from LYRICS.
2. "vocadb_chunks": Contains lyrics segments. Use this for specific lyrics search or detailed lyrical analysis.

Available Filter Fields (for payload filtering):
- name (str): Exact match for song name.
- producers_any (list[str]): Match ANY of these producers.
- producers_all (list[str]): Match ALL of these producers.
- vsingers_any (list[str]): Match ANY of these vocalists.
- vsingers_all (list[str]): Match ALL of these vocalists.
- tagNames (list[str]): Match ANY of these tags (e.g., "rock", "sad", "summer").
- year_min / year_max (int): Publication year range.
- month_min / month_max (int): Publication month range.
- rating_min / rating_max (float): Rating score range.
- favorite_min / favorite_max (int): Number of favorites range.
- length_min / length_max (int): Song length in seconds.
- culture (str): Primary culture code (e.g., "ja", "en", "zh").

IMPORTANT RULES FOR NAMES:
- You MUST convert common English or Chinese names to their OFFICIAL Japanese/Original names found in VocaDB.
- Examples:
  - "Hatsune Miku" / "初音未来" -> "初音ミク"
  - "PinocchioP" / "匹诺曹P" -> "ピノキオピー"
  - "DECO*27" -> "DECO*27" (Keep as is)
  - "Giga" -> "Giga"
  - "Mitchie M" -> "Mitchie M"
  - "Kagamine Rin" / "镜音铃" -> "鏡音リン"
  - "Kagamine Len" / "镜音连" -> "鏡音レン"
  - "Luo Tianyi" / "洛天依" -> "洛天依" (Chinese vocaloids usually keep Chinese names)

Output JSON Schema:
{
  "collection": "vocadb_songs" | "vocadb_chunks",
  "query_text": "string" | null, (The semantic search query. This matches against LYRICS. Do not use abstract queries like "similar to X". Use specific imagery, themes, or words.),
  "top_k": int, (Default 10),
  "filters": {
    "producers_any": [],
    "vsingers_any": [],
    "year_min": int,
    ... (include only used filters)
  }
}

Example 1: "Find happy songs by PinocchioP"
{
  "collection": "vocadb_songs",
  "query_text": "happy cheerful positive lyrics",
  "top_k": 5,
  "filters": {
    "producers_any": ["ピノキオピー"]
  }
}

Example 2: "Songs by Miku published in 2023"
{
  "collection": "vocadb_songs",
  "query_text": null,
  "top_k": 10,
  "filters": {
    "vsingers_any": ["初音ミク"],
    "year_min": 2023,
    "year_max": 2023
  }
}
"""