#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 VocaDB 歌曲 JSON 构建双向量 Chroma 数据库（使用 OpenAI embeddings）：

1. song-level collection: vocadb_songs
   - 每首歌 1 条向量
   - document: 标题 + 空行 + 全部歌词

2. chunk-level collection: vocadb_chunks
   - 每首歌若干条向量
   - 使用 “自然段 + 长度控制” 的 chunk 策略
   - document: 单个 chunk 文本

metadata 包含：
   song_id, defaultName, year, primaryCultureCode,
   ratingScore, favoritedTimes, lengthSeconds,
   producerNames, tagNames, (chunk-level 还有 chunk_index)

使用模型：
   text-embedding-3-small （max 1536 维，多语言，便宜）

用法示例：
   python build_chroma_dual.py \
      --json-dir ./vocadb_songs \
      --chroma-dir ./chroma_vdb \
      --batch-size 64 \
      --max-songs 0
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

import chromadb
from chromadb.config import Settings
from openai import OpenAI
from dotenv import load_dotenv


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
        description="Build dual Chroma vector DB (song-level + chunk-level) from VocaDB JSON."
    )
    parser.add_argument(
        "--json-dir",
        type=str,
        required=True,
        help="VocaDB 歌曲 JSON 文件目录（song_*.json）",
    )
    parser.add_argument(
        "--chroma-dir",
        type=str,
        required=True,
        help="Chroma 持久化目录（会自动创建）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="一次发送给 OpenAI 的文本数量（默认 64）",
    )
    parser.add_argument(
        "--max-songs",
        type=int,
        default=0,
        help="最多处理多少首歌（0 表示全部）",
    )
    parser.add_argument(
        "--song-level",
        action="store_true",
        help="构建 song-level",
    )
    parser.add_argument(
        "--chunk-level",
        action="store_true",
        help="构建 chunk-level",
    )
    return parser.parse_args()


def init_openai_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("环境变量 OPENAI_API_KEY 未设置。")
    return OpenAI(api_key=api_key)

def init_chroma_collections(chroma_dir: Path):
    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )

    # song-level collection
    try:
        song_col = client.get_collection(name=SONG_COLLECTION_NAME)
        logging.info("使用已有 song-level collection：%s", SONG_COLLECTION_NAME)
    except Exception:
        song_col = client.create_collection(
            name=SONG_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logging.info("创建新的 song-level collection：%s", SONG_COLLECTION_NAME)

    # chunk-level collection
    try:
        chunk_col = client.get_collection(name=CHUNK_COLLECTION_NAME)
        logging.info("使用已有 chunk-level collection：%s", CHUNK_COLLECTION_NAME)
    except Exception:
        chunk_col = client.create_collection(
            name=CHUNK_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logging.info("创建新的 chunk-level collection：%s", CHUNK_COLLECTION_NAME)

    return song_col, chunk_col


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

    meta = {
        "song_id": song.get("id"),
        "defaultName": song.get("defaultName"),
        "year": song.get("year"),
        "primaryCultureCode": song.get("primaryCultureCode"),
        "ratingScore": song.get("ratingScore"),
        "favoritedTimes": song.get("favoritedTimes"),
        "lengthSeconds": song.get("lengthSeconds"),
        "producerNames": song.get("producerNames") or [],
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


def flush_batch_to_chroma(
    client: OpenAI,
    collection,
    batch_ids: List[str],
    batch_docs: List[str],
    batch_metas: List[Dict[str, Any]],
    label: str,
) -> int:
    """将当前 batch 写入指定的 Chroma collection，返回写入数量。"""
    if not batch_ids:
        return 0

    try:
        embeddings = embed_batch(client, batch_docs)
        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas,
            embeddings=embeddings,
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
    chroma_dir = Path(args.chroma_dir)

    if not json_dir.exists():
        raise RuntimeError(f"json-dir 不存在: {json_dir}")

    if not args.song_level and not args.chunk_level:
        raise RuntimeError("必须至少启用 song-level 或 chunk-level 之一。")

    client = init_openai_client()
    song_col, chunk_col = init_chroma_collections(chroma_dir)

    song_files = iter_song_files(json_dir, args.max_songs)
    logging.info("即将处理 JSON 文件数：%d", len(song_files))

    # song-level 批缓存
    song_ids: List[str] = []
    song_docs: List[str] = []
    song_metas: List[Dict[str, Any]] = []

    # chunk-level 批缓存
    chunk_ids: List[str] = []
    chunk_docs: List[str] = []
    chunk_metas: List[Dict[str, Any]] = []

    total_song_added = 0
    total_chunk_added = 0

    for path in tqdm(song_files, desc="Building embeddings"):
        song = load_song(path)

        lyrics = song.get("originalLyrics")
        if not lyrics or not str(lyrics).strip():
            continue

        meta_common = build_common_metadata(song)
        song_id = meta_common.get("song_id")
        if song_id is None:
            continue
        song_id_str = str(song_id)

        # ---------- song-level ----------
        if args.song_level:
            doc_song = build_song_document(song)
            if doc_song.strip():
                song_ids.append(song_id_str)
                song_docs.append(doc_song)
                song_metas.append(meta_common)

        # ---------- chunk-level ----------
        if args.chunk_level:
            chunks = chunk_lyrics(
                lyrics=str(lyrics)
            )
            for idx, ch_text in enumerate(chunks):
                if not ch_text.strip():
                    continue
                chunk_id = f"{song_id_str}:{idx}"
                meta_chunk = dict(meta_common)
                meta_chunk["chunk_index"] = idx
                chunk_ids.append(chunk_id)
                chunk_docs.append(ch_text)
                chunk_metas.append(meta_chunk)

        # 批量写入控制
        if args.song_level and len(song_ids) >= args.batch_size:
            added = flush_batch_to_chroma(
                client, song_col, song_ids, song_docs, song_metas, label="song"
            )
            total_song_added += added
            song_ids, song_docs, song_metas = [], [], []

        if args.chunk_level and len(chunk_ids) >= args.batch_size:
            added = flush_batch_to_chroma(
                client, chunk_col, chunk_ids, chunk_docs, chunk_metas, label="chunk"
            )
            total_chunk_added += added
            chunk_ids, chunk_docs, chunk_metas = [], [], []

    # 处理尾巴
    if args.song_level and song_ids:
        added = flush_batch_to_chroma(
            client, song_col, song_ids, song_docs, song_metas, label="song-final"
        )
        total_song_added += added

    if args.chunk_level and chunk_ids:
        added = flush_batch_to_chroma(
            client, chunk_col, chunk_ids, chunk_docs, chunk_metas, label="chunk-final"
        )
        total_chunk_added += added

    logging.info(
        "构建完成：song-level 向量条目=%d, chunk-level 向量条目=%d",
        total_song_added, total_chunk_added
    )


if __name__ == "__main__":
    main()
