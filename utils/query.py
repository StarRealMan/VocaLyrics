#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
通用 Qdrant 查询工具：

功能：
- 支持 chunk-level / song-level 查询
- 支持可选 embedding（query_text 为 None 时只按 payload 条件筛选）
- 支持各种 payload 条件
"""

from typing import List, Optional, Sequence, Any, Dict

from openai import OpenAI

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchAny,
    MatchValue,
    Range,
)

# Embedding 配置
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

# ---------- Embedding 工具 ----------

def embed_text(client: OpenAI, text: str) -> List[float]:
    """对单条文本做 embedding。"""
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text],
        dimensions=EMBEDDING_DIM,
    )
    return resp.data[0].embedding


# ---------- Filter 构建（核心） ----------

def _add_match_any(
    must_list: List[FieldCondition],
    key: str,
    values: Optional[Sequence[Any]],
) -> None:
    """辅助函数：给 must_list 加一个 MatchAny 条件（OR）。"""
    if not values:
        return
    must_list.append(
        FieldCondition(
            key=key,
            match=MatchAny(any=list(values)),
        )
    )


def _add_match_all(
    must_list: List[FieldCondition],
    key: str,
    values: Optional[Sequence[Any]],
) -> None:
    """
    辅助函数：给 must_list 加“全部匹配”的条件（AND）。
    思路：对同一个 key 加多个条件，每个都是 MatchAny([单个值])，
    Qdrant 会要求这些条件都满足，相当于 “包含所有这些值”。
    """
    if not values:
        return
    for v in values:
        must_list.append(
            FieldCondition(
                key=key,
                match=MatchValue(value=v),
            )
        )


def _add_match_value(
    must_list: List[FieldCondition],
    key: str,
    value: Optional[Any],
) -> None:
    """辅助函数：给 must_list 加一个 MatchValue 条件（等于某个值）。"""
    if value is None:
        return
    must_list.append(
        FieldCondition(
            key=key,
            match=MatchValue(value=value),
        )
    )

def _add_range(
    must_list: List[FieldCondition],
    key: str,
    min_value: Optional[Any],
    max_value: Optional[Any],
) -> None:
    """辅助函数：给 must_list 加一个 Range 条件（范围）。"""
    if min_value is None and max_value is None:
        return
    r: Dict[str, Any] = {}
    if min_value is not None:
        r["gte"] = min_value
    if max_value is not None:
        r["lte"] = max_value
    must_list.append(
        FieldCondition(
            key=key,
            range=Range(**r),
        )
    )

def _add_must(
    must_list: List[FieldCondition],
    key: str,
    count_key: int,
    values: Optional[Sequence[Any]],
) -> None:
    """辅助函数：给 must_list 加一个“list必须全等”的条件。"""
    if not values or count_key is None:
        return
    for v in values:
        must_list.append(
            FieldCondition(
                key=key,
                match=MatchValue(value=v),
            )
        )
    must_list.append(
        FieldCondition(
            key=count_key,
            match=MatchValue(value=len(values)),
        )
    )

def build_payload_filter(
    name : Optional[str] = None,
    # producer 相关
    producers_any: Optional[Sequence[str]] = None,
    producers_all: Optional[Sequence[str]] = None,
    producers_must: Optional[Sequence[str]] = None,
    producers_min: Optional[int] = None,
    producers_max: Optional[int] = None,
    # vsinger 相关
    vsingers_any: Optional[Sequence[str]] = None,
    vsingers_all: Optional[Sequence[str]] = None,
    vsingers_must: Optional[Sequence[str]] = None,
    vsingers_min: Optional[int] = None,
    vsingers_max: Optional[int] = None,
    # tag 相关
    tags_any: Optional[Sequence[str]] = None,
    tags_all: Optional[Sequence[str]] = None,
    # 年份 / 月份
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    month_min: Optional[int] = None,
    month_max: Optional[int] = None,
    # culture 相关
    culture: Optional[str] = None,
    # rating 相关
    rating_min: Optional[float] = None,
    rating_max: Optional[float] = None,
    # favorite 相关
    favorite_min: Optional[int] = None,
    favorite_max: Optional[int] = None,
    # length 相关
    length_min: Optional[int] = None,
    length_max: Optional[int] = None,

) -> Optional[Filter]:
    """
    根据各种条件构造 Qdrant 的 Filter。
    所有条件都是 AND 关系（must），其中某些内部是 OR（如 *any）。
    """
    must: List[FieldCondition] = []

    # name: string
    _add_match_value(must, "defaultName", name)

    # producerNames: list[string]
    _add_match_any(must, "producerNames", producers_any)
    _add_match_all(must, "producerNames", producers_all)
    _add_range(must, "producerNum", producers_min, producers_max)
    _add_must(must, "producerNames", "producerNum", producers_must)

    # vsinger: list[string]
    _add_match_any(must, "vsingerNames", vsingers_any)
    _add_match_all(must, "vsingerNames", vsingers_all)
    _add_range(must, "vsingerNum", vsingers_min, vsingers_max)
    _add_must(must, "vsingerNames", "vsingerNum", vsingers_must)

    # tagNames: list[string]
    _add_match_any(must, "tagNames", tags_any)
    _add_match_all(must, "tagNames", tags_all)

    # cultureCode: string (ja, cn, en, etc.)
    _add_match_value(must, "primaryCultureCode", culture)

    # year range
    _add_range(must, "year", year_min, year_max)

    # month range
    _add_range(must, "month", month_min, month_max)

    # ratingScore range
    _add_range(must, "ratingScore", rating_min, rating_max)

    # favoritedTimes range
    _add_range(must, "favoritedTimes", favorite_min, favorite_max)

    # lengthSeconds range
    _add_range(must, "lengthSeconds", length_min, length_max)

    if not must:
        return None

    return Filter(must=must)


# ---------- 查询函数 ----------

def query(
    qdrant_client: QdrantClient,
    openai_client: Optional[OpenAI] = None,
    top_k: int = 10,
    query_text: Optional[str] = None,
    name: Optional[str] = None,
    producers_any: Optional[Sequence[str]] = None,
    producers_all: Optional[Sequence[str]] = None,
    producers_must: Optional[Sequence[str]] = None,
    producers_min: Optional[int] = None,
    producers_max: Optional[int] = None,
    vsingers_any: Optional[Sequence[str]] = None,
    vsingers_all: Optional[Sequence[str]] = None,
    vsingers_must: Optional[Sequence[str]] = None,
    vsingers_min: Optional[int] = None,
    vsingers_max: Optional[int] = None,
    tags_any: Optional[Sequence[str]] = None,
    tags_all: Optional[Sequence[str]] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    month_min: Optional[int] = None,
    month_max: Optional[int] = None,
    culture: Optional[str] = None,
    rating_min: Optional[float] = None,
    rating_max: Optional[float] = None,
    favorite_min: Optional[int] = None,
    favorite_max: Optional[int] = None,
    length_min: Optional[int] = None,
    length_max: Optional[int] = None,
    collection: Optional[str] = None,
):
    """
    查询 chunk-level（vocadb_chunks）：

    - 如果 query_text 不为空：
        → 先做 embedding，再用 query_points(query + filter) 做向量检索。
    - 否则：
        → 只用 payload Filter + scroll 做硬条件筛选。
    """

    # 构建 Filter
    qfilter = build_payload_filter(
        name = name,
        producers_any=producers_any,
        producers_all=producers_all,
        producers_must=producers_must,
        producers_min=producers_min,
        producers_max=producers_max,
        vsingers_any=vsingers_any,
        vsingers_all=vsingers_all,
        vsingers_must=vsingers_must,
        vsingers_min=vsingers_min,
        vsingers_max=vsingers_max,
        tags_any=tags_any,
        tags_all=tags_all,
        culture=culture,
        year_min=year_min,
        year_max=year_max,
        month_min=month_min,
        month_max=month_max,
        rating_min=rating_min,
        rating_max=rating_max,
        favorite_min=favorite_min,
        favorite_max=favorite_max,
        length_min=length_min,
        length_max=length_max,
    )

    # 情况 1：有文本 & 要做向量检索
    if query_text:
        if openai_client is None:
            raise ValueError("query_text 不为空时，需要提供 openai_client。")
        vec = embed_text(openai_client, query_text)
        resp = qdrant_client.query_points(
            collection_name=collection,
            query=vec,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            query_filter=qfilter,
        )
        return resp.points or []

    # 情况 2：纯 payload 检索（不算 embedding）
    else:
        points, _ = qdrant_client.scroll(
            collection_name=collection,
            scroll_filter=qfilter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        return points
