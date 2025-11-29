# VocaLyrics
An agent that ignites your imagination and guides you through the creation of VOCALOID lyrics.

## 🔧 Setup

```bash
conda create -n vocalyrics python=3.10 -y
conda activate vocalyrics
pip install -r requirements.txt

cd docker
docker compose up -d
```

Optionally, run `crawl_vocadb_data.py` and `build_database.py` to update the Qdrant database with the latest data from VocaDB.

## 🌐 Environment

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_BASE_URL` | OpenAI API base url for third-party services |
| `OPENAI_API_KEY` | OpenAI API key |
| `QDRANT_URL` | Qdrant instance address |
| `QDRANT__SERVICE__API_KEY` | Required if authentication is enabled |

We recommend save these variables in a `.env` file for local development.

## 🧠 Multi-agent workflow

## ▶️ Run

single query:

```bash
python main.py --query "please help me write a VOCALOID song lyrics about summer and friendship"
```

interactive mode:

```bash
python main.py --interactive
```