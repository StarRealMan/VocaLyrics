# VocaLyrics
An agent that ignites your imagination and guides you through the creation of VOCALOID lyrics.

## Installation

```bash
conda create -n vocalyrics python=3.10 -y
conda activate vocalyrics
pip install -r requirements.txt
```

## Usage

1. crawl lyrics from vocaDB 
```bash
python data/crawl_vocadb_data.py
```

2. build vector database
```bash
python data/build_database.py
```
