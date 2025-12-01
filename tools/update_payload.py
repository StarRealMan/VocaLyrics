#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility script to update payload counts in Qdrant collection."""

import os
import json
from utils.client import	init_qdrant_client_and_collections
from utils.query import EMBEDDING_DIM


def update_payload_counts(client, collection_name, json_base):
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
            payload = point.payload or {}
            # song_id = payload.get("song_id")
            # with open(os.path.join(json_base, f"song_{song_id}.json"), "r", encoding="utf-8") as json_file:
            #     json_data = json.load(json_file)
            
            vsingerNames = payload.get("vsingerNames")
            renewedNames = []
            for name in vsingerNames:
                name_clean = name.split(" ")[0]
                renewedNames.append(name_clean)
            
            renewedNames = list(set(renewedNames))
                
            client.set_payload(
                collection_name=collection_name,
                payload={
                    "vsingerNames": renewedNames,
                    "vsingerNum": len(renewedNames)
                },
                points=[point.id],
            )

            total_updated += 1

        print(f"Updated {total_updated} points so far...")

        if offset is None:
            break

    print(f"Update completed! Total updated: {total_updated}")


if __name__ == "__main__":
    collection_name = "vocadb_songs"
    json_base = "/root/Data/vocadb_raw_json/"
    client = init_qdrant_client_and_collections(
        embedding_dim=EMBEDDING_DIM,
        chunk_collection_name=collection_name,
        create_payload_indexes=True
    )

    update_payload_counts(client, collection_name, json_base)
