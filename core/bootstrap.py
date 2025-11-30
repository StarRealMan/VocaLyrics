"""Helper utilities to initialize the orchestrator and default agents."""

from __future__ import annotations

from typing import Dict, List, Tuple

from openai import OpenAI
from qdrant_client import QdrantClient

from agents import (
    AnalystAgent,
    ComposerAgent,
    GeneralAgent,
    ParserAgent,
    PlannerAgent,
    RetrieverAgent,
    WriterAgent,
)
from core.context import ConversationContext
from core.orchestrator import Orchestrator
from core.task import TaskManager
from utils import client as client_utils
from utils import query as query_utils


def init_services() -> Tuple[OpenAI, QdrantClient]:
    """Initialize OpenAI and Qdrant clients using env configuration."""

    openai_client = client_utils.init_openai_client()
    qdrant_client = client_utils.init_qdrant_client_and_collections(
        embedding_dim=query_utils.EMBEDDING_DIM,
        song_collection_name=client_utils.SONG_COLLECTION_NAME,
        chunk_collection_name=client_utils.CHUNK_COLLECTION_NAME,
        create_payload_indexes=True,
    )
    return openai_client, qdrant_client


def build_default_agents(
    openai_client: OpenAI,
    qdrant_client: QdrantClient,
) -> List:
    """Instantiate the standard agent roster."""

    return [
        PlannerAgent(openai_client=openai_client),
        RetrieverAgent(openai_client=openai_client, qdrant_client=qdrant_client),
        AnalystAgent(openai_client=openai_client),
        ParserAgent(openai_client=openai_client),
        ComposerAgent(openai_client=openai_client),
        WriterAgent(openai_client=openai_client),
        GeneralAgent(openai_client=openai_client),
    ]


def create_default_orchestrator() -> Tuple[Orchestrator, Dict[str, object]]:
    """Create an orchestrator with all default agents registered."""

    openai_client, qdrant_client = init_services()
    context = ConversationContext()
    task_manager = TaskManager()
    orchestrator = Orchestrator(context=context, task_manager=task_manager)
    orchestrator.register_agents(build_default_agents(openai_client, qdrant_client))

    resources: Dict[str, object] = {
        "openai": openai_client,
        "qdrant": qdrant_client,
        "context": context,
        "task_manager": task_manager,
    }
    return orchestrator, resources
