import os
import logging
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

SONG_COLLECTION_NAME = "vocadb_songs"
CHUNK_COLLECTION_NAME = "vocadb_chunks"

def init_openai_client(base_url) -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("环境变量 OPENAI_API_KEY 未设置。")
    return OpenAI(
        api_key=api_key,
        base_url=base_url
    )


def init_qdrant_client_and_collections(
        qdrant_dir: Path,
        embedding_dim: int,
        song_collection_name: str,
        chunk_collection_name: str
    ) -> QdrantClient:
    """
    初始化本地嵌入式 Qdrant，并确保 song-level 和 chunk-level 两个 collection 存在。
    """
    qdrant_dir.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(qdrant_dir))

    def ensure_collection(name: str) -> None:
        try:
            client.get_collection(collection_name=name)
            logging.info("使用已有 collection：%s", name)
        except Exception:
            client.recreate_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            logging.info("创建新的 collection：%s", name)

    ensure_collection(song_collection_name)
    ensure_collection(chunk_collection_name)

    return client
