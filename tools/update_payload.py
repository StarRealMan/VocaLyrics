#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility script to update payload counts in Qdrant collection."""

from utils.client import	init_qdrant_client_and_collections
from utils.query import EMBEDDING_DIM

def update_payload_counts(client, collection_name):
    page_size = 1000
    offset = 0 

    total_updated = 0

    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            offset=offset,
            limit=page_size,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            break

        for point in points:
            client.set_payload(
                collection_name=collection_name,
                payload={
                    # update payload
                },
                points=[point.id],
            )

            total_updated += 1

        print(f"Updated {total_updated} points so far...")

        if offset is None:
            break

    print(f"Update completed! Total updated: {total_updated}")


if __name__ == "__main__":
    collection_name = "vocadb_chunks"
    client = init_qdrant_client_and_collections(
        embedding_dim=EMBEDDING_DIM,
        chunk_collection_name=collection_name,
        create_payload_indexes=True
    )

    update_payload_counts(client, collection_name)
