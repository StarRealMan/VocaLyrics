import os
import json
from typing import List, Any, Dict, Optional, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from core.context import Context
from core.task import Task
from agents.base import Agent
from utils.query import query

class RetrieverFilter(BaseModel):
    """用于 payload 过滤的条件字典结构。"""

    name: Optional[str] = Field(None, description="Exact match for song name.")
    producers_any: Optional[List[str]] = Field(None, description="Match ANY of these producers.")
    producers_all: Optional[List[str]] = Field(None, description="Match ALL of these producers.")
    producers_must: Optional[List[str]] = Field(None, description="EXACT match of these producers.")
    producers_min: Optional[int] = Field(None, description="Minimum number of producers.")
    producers_max: Optional[int] = Field(None, description="Maximum number of producers.")
    vsingers_any: Optional[List[str]] = Field(None, description="Match ANY of these vocalists.")
    vsingers_all: Optional[List[str]] = Field(None, description="Match ALL of these vocalists.")
    vsingers_must: Optional[List[str]] = Field(None, description="EXACT match of these vocalists.")
    vsingers_min: Optional[int] = Field(None, description="Minimum number of vocalists.")
    vsingers_max: Optional[int] = Field(None, description="Maximum number of vocalists.")
    tags_any: Optional[List[str]] = Field(None, description="Match ANY of these tags (e.g., 'rock', 'sad', 'summer')."),
    tags_all: Optional[List[str]] = Field(None, description="Match ALL of these tags (e.g., 'rock', 'sad', 'summer')."),
    rating_min: Optional[float] = Field(None, description="Minimum rating score.")
    rating_max: Optional[float] = Field(None, description="Maximum rating score.")
    favorite_min: Optional[int] = Field(None, description="Minimum number of favorites.")
    favorite_max: Optional[int] = Field(None, description="Maximum number of favorites.")
    length_min: Optional[int] = Field(None, description="Minimum song length in seconds.")
    length_max: Optional[int] = Field(None, description="Maximum song length in seconds.")
    culture: Optional[str] = Field(None, description="ISO language code (e.g., 'ja', 'en', 'zh').")
    year_min: Optional[int] = Field(None, description="Minimum year.")
    year_max: Optional[int] = Field(None, description="Maximum year.")
    month_min: Optional[int] = Field(None, description="Minimum month.")
    month_max: Optional[int] = Field(None, description="Maximum month.")

class RetrieverAnalyseResult(BaseModel):
    """LLM 对检索请求解析后的结构化结果。"""

    collection: Literal[
        "vocadb_songs",
        "vocadb_chunks"
    ] = Field(..., description="Collection level to query (song level or chunk (section) level).")
    top_k: int = Field(..., description="Number of final results desired to return.")
    use_rerank: bool = Field(..., description="Whether to use reranking.")
    query_text: Optional[str] = Field(None, description="The semantic search query string, None if using payload filter only.")
    filters: Optional[RetrieverFilter] = Field(None, description="Payload filter conditions.")
    prefilt_key: Optional[Literal[
        "year",
        "month",
        "rating",
        "favorite",
        "length"
    ]] = Field(None, description="Prefilt retrieved results using this key. When using prefilting, relax corresponding filters.")

key_reference_table = {
    "year": "year",
    "month": "month",
    "rating": "ratingScore",
    "favorite": "favoritedTimes",
    "length": "lengthSeconds"
}

class Retriever(Agent):
    """
    检索者 Agent (集成 Cohere Rerank)
    
    负责从 Qdrant 数据库中检索歌曲或歌词片段。
    采用三阶段工作流：
    1. Query Parsing: 使用 LLM 将自然语言请求转换为结构化的查询参数。
    2. Recall: 执行数据库向量查询，召回大量候选 (Top 100+)。
    3. Rerank: 使用 Cohere Rerank 模型对候选进行精排，返回 Top K。
    """
    
    def __init__(self, openai_client, cohere_client, qdrant_client):
        super().__init__(name="Retriever", description="Retrieves songs and lyrics from the database based on natural language requests.")
        load_dotenv()
        self.openai_client = openai_client
        self.qdrant_client = qdrant_client
        self.cohere_client = cohere_client
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
        
        # --- 策略：大召回 -> 精排 ---
        target_top_k = query_params.top_k
        
        # 如果启用 Rerank，召回数量设为 100 (或者 target 的 5-10 倍)
        if query_params.use_rerank:
            rerank_top_k = max(200, target_top_k * 10) 
            self.logger.info(f"Recall strategy: Fetch {rerank_top_k} items -> Rerank -> Return Top {target_top_k}")
            if query_params.query_text:
                filting_top_k = max(100, target_top_k * 5) 
            else:
                filting_top_k = target_top_k
        else:
            rerank_top_k = target_top_k

        # 阶段 2: Vector Search (Recall)
        query_params.top_k = rerank_top_k

        ranking_keys = []
        
        # 注意：这里需要适配 _execute_search 的参数格式
        # _execute_search 接受的是 RetrieverAnalyseResult 对象，我们需要临时修改它的 top_k
        # 但由于 Pydantic 模型是不可变的(默认情况下不是，但为了安全起见)，我们最好传入 dict 给 _execute_search
        # 不过之前的 _execute_search 实现是接收 dict 的，所以我们直接传 dict
        results = self._execute_search(query_params)

        # 阶段 3: Reranking (Cohere)
        if query_params.use_rerank and len(results) > target_top_k:

            if query_params.prefilt_key and query_params.filters and len(results) > filting_top_k:

                filters = query_params.filters.model_dump(exclude_none=True)
                for key in filters.keys():
                    if key.split("_")[0] == query_params.prefilt_key:
                        
                        reverse = False if key.endswith("max") else True
                        self.logger.info(f"Filtering {filting_top_k} items based on field number before reranking.")
                        results = sorted(results, key=lambda x: x.payload.get(query_params.prefilt_key, 0), reverse=reverse)[:filting_top_k]
            
            if query_params.query_text:
                try:
                    self.logger.info(f"Reranking {len(results)} items with Cohere...")
                    
                    # 准备文档
                    docs = []
                    valid_indices = []
                    
                    for i, point in enumerate(results):
                        content = point.payload.get("lyrics") or point.payload.get("name", "")
                        # 截断以防万一，虽然 Cohere 支持较长文本
                        if content:
                            docs.append(str(content)[:2000]) 
                            valid_indices.append(i)
                    
                    rerank_response = self.cohere_client.rerank(
                        model="rerank-multilingual-v3.0",
                        query=query_params.query_text,
                        documents=docs,
                        top_n=target_top_k, # 只取最终需要的 Top K
                    )
                    
                    reranked_points = []
                    for r in rerank_response.results:
                        original_idx = valid_indices[r.index]
                        point = results[original_idx]
                        point.score = r.relevance_score
                        reranked_points.append(point)
                    
                    final_results = reranked_points
                    self.logger.info("Reranking completed.")

                except Exception as e:
                    self.logger.error(f"Reranking failed: {e}. Falling back to vector scores.")
                    final_results = results[:target_top_k]
        else:
            final_results = results

        # 结果处理：序列化
        serialized_results = []
        for point in final_results:
            serialized_results.append({
                "id": point.id,
                "score": point.score,
                "payload": point.payload
            })
            
        self._save_to_memory(context, task, serialized_results)
        
        return f"Retrieved {len(serialized_results)} items (Reranked)."

    def _analyze_request(self, request: str) -> RetrieverAnalyseResult:
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
        filters = params.filters.model_dump() if params.filters else {}
        
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