#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 VocaDB 歌曲 JSON 构建双向量 Qdrant 数据库（使用 OpenAI embeddings）：

1. song-level collection: vocadb_songs
   - 每首歌 1 条向量
   - payload.document: 标题 + 空行 + 全部歌词

2. chunk-level collection: vocadb_chunks
   - 每首歌若干条向量
   - 使用 “自然段 + 长度控制” 的 chunk 策略
   - payload.document: 单个 chunk 文本

payload / metadata 包含：
   song_id, defaultName, year, primaryCultureCode,
   ratingScore, favoritedTimes, lengthSeconds,
   producerNames, tagNames, (chunk-level 还有 chunk_index, chunk_id)

使用模型：
   text-embedding-3-small （1536 维，多语言，便宜）

用法示例：
   python build_database.py \
      --json_dir ./vocadb_songs \
      --qdrant_dir ./qdrant_vdb \
      --batch_size 64 \
      --max_songs 0 \
      --song_level --chunk_level \
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from openai import OpenAI

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from data.client import (
    init_openai_client,
    init_qdrant_client_and_collections,
)

# ------------ 配置区 ------------
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

SONG_COLLECTION_NAME = "vocadb_songs"
CHUNK_COLLECTION_NAME = "vocadb_chunks"
# -------------------------------


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build dual Qdrant vector DB (song-level + chunk-level) from VocaDB JSON."
    )
    parser.add_argument(
        "--json_dir",
        type=str,
        required=True,
        help="VocaDB 歌曲 JSON 文件目录（song_*.json）",
    )
    parser.add_argument(
        "--qdrant_dir",
        type=str,
        required=True,
        help="Qdrant 持久化目录",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="一次发送给 OpenAI 的文本数量（默认 64）",
    )
    parser.add_argument(
        "--max_songs",
        type=int,
        default=0,
        help="最多处理多少首歌（0 表示全部）",
    )
    parser.add_argument(
        "--min_lines",
        type=int,
        default=2,
        help="chunk-level 最小行数（默认 2 行）",
    )
    parser.add_argument(
        "--on_disk",
        action="store_true",
        help="将向量存储在磁盘上（默认不存储）",
    )
    parser.add_argument(
        "--song_level",
        action="store_true",
        help="构建 song-level",
    )
    parser.add_argument(
        "--chunk_level",
        action="store_true",
        help="构建 chunk-level",
    )
    return parser.parse_args()


def chunk_lyrics(
    lyrics: str,
    min_lines: int = 1
) -> List[str]:
    """
    以空行分段，并保证每段至少 min_lines 行，不够则合并下一段。
    """
    raw_sections = [sec.strip() for sec in lyrics.split("\n\n") if sec.strip()]
    sections = [sec.split("\n") for sec in raw_sections]

    chunks = []
    i = 0
    n = len(sections)

    while i < n:
        lines = sections[i]
        while len(lines) < min_lines and i + 1 < n:
            i += 1
            lines += [""] + sections[i]

        chunks.append("\n".join(lines))
        i += 1

    return chunks


def build_song_document(song: Dict[str, Any]) -> str:
    """整首歌的 document：标题 + 空行 + 全部歌词。"""
    title = str(song.get("defaultName") or "")
    lyrics = str(song.get("originalLyrics") or "")

    if lyrics.strip():
        return f"{title}\n\n{lyrics}"
    else:
        return title


def build_common_metadata(song: Dict[str, Any]) -> Dict[str, Any]:
    """song-level 和 chunk-level 共用的 metadata 基础部分。"""
    tags = song.get("tags") or []
    tag_names = [
        t.get("tagName") for t in tags
        if isinstance(t, dict) and t.get("tagName")
    ]

    artists = song.get("artists") or []
    vsinger_names = [
        a.get("name") for a in artists
        if isinstance(a, dict) and a.get("name") and a.get("role") and a.get("role") == "Vocalist"
    ]

    meta = {
        "song_id": song.get("id"),
        "defaultName": song.get("defaultName"),
        "year": song.get("year"),
        "month": song.get("month"),
        "primaryCultureCode": song.get("primaryCultureCode"),
        "ratingScore": song.get("ratingScore"),
        "favoritedTimes": song.get("favoritedTimes"),
        "lengthSeconds": song.get("lengthSeconds"),
        "mainPicture": song.get("mainPictureUrlOriginal"),
        "producerNames": song.get("producerNames") or [],
        "vsingerNames": vsinger_names,
        "tagNames": tag_names,
    }
    return meta


def iter_song_files(json_dir: Path, max_songs: int = 0) -> List[Path]:
    files = sorted(json_dir.glob("song_*.json"))
    if max_songs > 0:
        files = files[:max_songs]
    return files


def load_song(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def embed_batch(client: OpenAI, texts: List[str]) -> List[List[float]]:
    """调用 OpenAI embeddings API，对一批文本做 embedding。"""
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        dimensions=EMBEDDING_DIM,
    )
    return [item.embedding for item in resp.data]


def flush_batch_to_qdrant(
    openai_client: OpenAI,
    qdrant_client: QdrantClient,
    collection_name: str,
    batch_ids: List[int],
    batch_docs: List[str],
    batch_metas: List[Dict[str, Any]],
    label: str,
) -> int:
    """
    将当前 batch 写入指定的 Qdrant collection，返回写入数量。
    payload 中会包含：
      - 原来的 metadata
      - "document": 文本内容

    Qdrant 的 point id 使用自增整数（int），业务 ID 放在 payload 里：
      - song_id, chunk_index, chunk_id 等
    """
    if not batch_ids:
        return 0

    try:
        embeddings = embed_batch(openai_client, batch_docs)
        points: List[PointStruct] = []

        for _id, doc, meta, vec in zip(batch_ids, batch_docs, batch_metas, embeddings):
            payload = dict(meta)
            if label == "chunk":
                payload["lyrics"] = doc
            points.append(
                PointStruct(
                    id=_id,       # int，自增
                    vector=vec,
                    payload=payload,
                )
            )

        qdrant_client.upsert(
            collection_name=collection_name,
            points=points,
        )

        logging.info(
            "[%s] 写入 batch：%d 条（示例 id: %s）",
            label, len(batch_ids), batch_ids[0]
        )
        return len(batch_ids)
    except Exception as e:
        logging.error("[%s] 批量写入或 embedding 失败：%s", label, e)
        return 0


def main():
    setup_logger()
    args = parse_args()

    json_dir = Path(args.json_dir)

    if not json_dir.exists():
        raise RuntimeError(f"json-dir 不存在: {json_dir}")

    if not args.song_level and not args.chunk_level:
        raise RuntimeError("必须至少启用 song-level 或 chunk-level 之一。")

    if args.song_level:
        song_collection_name = SONG_COLLECTION_NAME
    else:
        song_collection_name = None
    if args.chunk_level:
        chunk_collection_name = CHUNK_COLLECTION_NAME
    else:
        chunk_collection_name = None

    openai_client = init_openai_client()
    qdrant_client = init_qdrant_client_and_collections(
        args.qdrant_dir,
        EMBEDDING_DIM,
        song_collection_name=song_collection_name,
        chunk_collection_name=chunk_collection_name,
        on_disk=args.on_disk,
    )

    song_files = iter_song_files(json_dir, args.max_songs)
    logging.info("即将处理 JSON 文件数：%d", len(song_files))

    # song-level 批缓存
    song_ids: List[int] = []
    song_docs: List[str] = []
    song_metas: List[Dict[str, Any]] = []

    # chunk-level 批缓存
    chunk_ids: List[int] = []
    chunk_docs: List[str] = []
    chunk_metas: List[Dict[str, Any]] = []

    total_song_added = 0
    total_chunk_added = 0

    # 为 Qdrant 的 point id 准备自增计数器（各 collection 独立即可）
    song_point_id = 1
    chunk_point_id = 1

    for path in tqdm(song_files, desc="Building embeddings"):
        song = load_song(path)

        lyrics = song.get("originalLyrics")
        if not lyrics or not str(lyrics).strip():
            continue

        meta_common = build_common_metadata(song)
        song_id = meta_common.get("song_id")
        if song_id is None:
            continue

        # ---------- song-level ----------
        if args.song_level:
            doc_song = build_song_document(song)
            if doc_song.strip():
                # Qdrant 的 point id 用自增整数
                song_ids.append(song_point_id)
                song_docs.append(doc_song)
                song_metas.append(meta_common)
                song_point_id += 1

        # ---------- chunk-level ----------
        if args.chunk_level:
            chunks = chunk_lyrics(
                lyrics=str(lyrics),
                min_lines=args.min_lines
            )
            for idx, ch_text in enumerate(chunks):
                if not ch_text.strip():
                    continue

                meta_chunk = dict(meta_common)
                meta_chunk["chunk_index"] = idx

                chunk_ids.append(chunk_point_id)
                chunk_docs.append(ch_text)
                chunk_metas.append(meta_chunk)
                chunk_point_id += 1

        # 批量写入控制
        if args.song_level and len(song_ids) >= args.batch_size:
            added = flush_batch_to_qdrant(
                openai_client,
                qdrant_client,
                SONG_COLLECTION_NAME,
                song_ids,
                song_docs,
                song_metas,
                label="song",
            )
            total_song_added += added
            song_ids, song_docs, song_metas = [], [], []

        if args.chunk_level and len(chunk_ids) >= args.batch_size:
            added = flush_batch_to_qdrant(
                openai_client,
                qdrant_client,
                CHUNK_COLLECTION_NAME,
                chunk_ids,
                chunk_docs,
                chunk_metas,
                label="chunk",
            )
            total_chunk_added += added
            chunk_ids, chunk_docs, chunk_metas = [], [], []

    # 处理尾巴
    if args.song_level and song_ids:
        added = flush_batch_to_qdrant(
            openai_client,
            qdrant_client,
            SONG_COLLECTION_NAME,
            song_ids,
            song_docs,
            song_metas,
            label="song-final",
        )
        total_song_added += added

    if args.chunk_level and chunk_ids:
        added = flush_batch_to_qdrant(
            openai_client,
            qdrant_client,
            CHUNK_COLLECTION_NAME,
            chunk_ids,
            chunk_docs,
            chunk_metas,
            label="chunk-final",
        )
        total_chunk_added += added

    logging.info(
        "构建完成：song-level 向量条目=%d, chunk-level 向量条目=%d",
        total_song_added, total_chunk_added
    )


if __name__ == "__main__":
    main()
