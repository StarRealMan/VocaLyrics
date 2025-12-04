# VocaLyrics

A multi-agent system that ignites your imagination and guides you through the creation of VOCALOID lyrics.

## 🔧 Setup

```bash
conda create -n vocalyrics python=3.10 -y
conda activate vocalyrics
pip install -r requirements.txt
```

You can use the provided qdrant database for easy setup: https://qdrant.starydy.xyz/

Optionally, you can deploy your own Qdrant instance using Docker. 
In this case, you need to crawl and build the database yourself:

```bash
# set up Qdrant with Docker
cd docker
docker compose up -d

# crawl and build the database
python -m tools.crawl_vocadb_data --output_dir /path/to/json
python -m tools.build_database --json_dir /path/to/json --song_level --chunk_level

# analyze stats
python -m tools.analyze_stats

# run retrieve test
python -m tools.retrieve_test

# update payload if needed
python -m tools.update_payload
```

## 🌐 Environment

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_BASE_URL` | OpenAI API base url for third-party services |
| `OPENAI_API_KEY` | OpenAI API key |
| `OPENAI_API_MODEL` | OpenAI API model |
| `QDRANT_URL` | Qdrant instance address |
| `QDRANT__SERVICE__API_KEY` | Required if authentication is enabled |

Create a `.env` file in the root directory and populate it with environment variables above.

## 🚀 Running

Run it with a single query:

```bash
python main.py --query "your request"
```

Or start an interactive session:

```bash
python main.py
```
