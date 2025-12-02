import os
import logging
from pathlib import Path
from typing import Union

from openai import OpenAI
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.models import Distance, VectorParams

SONG_COLLECTION_NAME = "vocadb_songs"
CHUNK_COLLECTION_NAME = "vocadb_chunks"

FIELD_SCHEMA_MAP = {
    "year": rest.PayloadSchemaType.INTEGER,
    "month": rest.PayloadSchemaType.INTEGER,
    "primaryCultureCode": rest.PayloadSchemaType.KEYWORD,
    "ratingScore": rest.PayloadSchemaType.INTEGER,
    "favoritedTimes": rest.PayloadSchemaType.INTEGER,
    "lengthSeconds": rest.PayloadSchemaType.INTEGER,
    "producerNames": rest.PayloadSchemaType.KEYWORD,
    "producerNum": rest.PayloadSchemaType.INTEGER,
    "vsingerNames": rest.PayloadSchemaType.KEYWORD,
    "vsingerNum": rest.PayloadSchemaType.INTEGER,
    "tagNames": rest.PayloadSchemaType.KEYWORD,
}


def init_openai_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1")
    if not api_key:
        raise RuntimeError("环境变量 OPENAI_API_KEY 未设置。")
    return OpenAI(
        api_key=api_key,
        base_url=base_url
    )


def init_qdrant_client_and_collections(
        embedding_dim: int,
        song_collection_name: str = None,
        chunk_collection_name: str = None,
        create_payload_indexes: bool = False,
        on_disk: bool = False,
    ) -> QdrantClient:
    """
    初始化Qdrant，并确保 song-level 和 chunk-level 两个 collection 存在。
    """

    logger = logging.getLogger("QdrantInit")

    assert song_collection_name or chunk_collection_name, \
        "Need at least one of song_collection_name or chunk_collection_name."
    load_dotenv()
    qdrant_dir = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT__SERVICE__API_KEY")
    if qdrant_dir.startswith("http://") or qdrant_dir.startswith("https://"):
        client = QdrantClient(url=qdrant_dir, api_key=api_key)
        logger.debug("使用远程 Qdrant 服务：%s", qdrant_dir)
    else:
        qdrant_dir = Path(qdrant_dir)
        qdrant_dir.mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=str(qdrant_dir), api_key=api_key)
        logger.debug("使用本地嵌入式 Qdrant，目录：%s", qdrant_dir)

    def ensure_collection(name: str) -> None:
        try:
            client.get_collection(collection_name=name)
            logger.debug("使用已有 collection：%s", name)
        except Exception:
            client.recreate_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE,
                    on_disk=on_disk,
                ),
                on_disk_payload=on_disk,
            )
            logger.debug("创建新的 collection：%s", name)
    
    def ensure_payload_indexes(name: str, field_schema_map: dict[str, rest.PayloadSchemaType]) -> None:
        for field_name, schema in field_schema_map.items():
            try:
                client.create_payload_index(
                    collection_name=name,
                    field_name=field_name,
                    field_schema=schema,
                )
                logger.debug(
                    "为 collection %s 创建字段 %s 的索引。",
                    name,
                    field_name,
                )
            except Exception as e:
                logger.warning(
                    "无法为 collection %s 创建字段 %s 的索引，可能已存在。错误信息：%s",
                    name,
                    field_name,
                    str(e),
                )

    if song_collection_name:
        ensure_collection(song_collection_name)
        if create_payload_indexes:
            ensure_payload_indexes(
                song_collection_name,
                FIELD_SCHEMA_MAP,
            )
    if chunk_collection_name:
        ensure_collection(chunk_collection_name)
        if create_payload_indexes:
            ensure_payload_indexes(
                chunk_collection_name,
                FIELD_SCHEMA_MAP,
            )

    return client
