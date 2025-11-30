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

Planner 会输出严格的 JSON 步骤列表，Orchestrator 再依次调用各代理。每个步骤对象包含：

```json
{
	"id": "step-1",
	"agent": "retriever",
	"goal": "检索 sad songs featuring Hatsune Miku",
	"inputs": {
		"query_text": "sad songs featuring Hatsune Miku",
		"filters": {"vsingers_any": ["初音未来"]},
		"level": "song",
		"top_k": 5
	}
}
```

`retriever` 建议提供 `query_text/filters/level/top_k`；`analyst` 传递 `references/focus`；`composer`/`writer` 应写明 `brief/style/format/seed/midi_summary` 等字段。Finisher 也会读取这些 `inputs`，所有代理都能在 trace JSON 中看到自己的参数。

### Agent I/O contracts

| Agent | Expected `inputs` 字段 | Output 约定 |
| --- | --- | --- |
| planner | `id/agent/goal/inputs` JSON 列表（仅 JSON，无 markdown） | `plan` artifact (list) |
| retriever | `query_text`, `filters`, `level`, `top_k`, `collection`, `pure_payload` | 文本 summary + `documents` artifact；若 `filters` 为空，会通过 LLM 自动推断并统一成官方 VocaDB 名称 |
| analyst | `focus`, `references`（可为空） | 三段式文本：Summary / Style Markers / Actionable Ideas |
| parser | `midi_path` | 解析摘要 + `midi` artifact |
| composer | `brief`, `style`, `references`, `seed_lyrics`, `midi_summary` | 分段歌词（Verse/Chorus/Bridge）+ `Notes` 段 |
| writer | `brief`, `format_hint` | 1 句 logline → 正文 → `Next steps` 列表 |
| general | `brief`, `prompt` | 2-4 段回答，可含 bullet |

#### 引用上游输出

Planner 可以用 `*_from` 字段把上游结果喂给下游。例如：

```json
{
	"agent": "analyst",
	"inputs": {
		"focus": "对比 step-1 的歌曲",
		"references_from": {"step": "step-1", "artifact": "documents"}
	}
}
```

- `references_from` 默认会寻找上一阶段的 `documents` artifact；`midi_summary_from` 默认读取 `midi` artifact。
- 文本字段（如 `brief_from`, `focus_from`, `prompt_from`）若未指定 `artifact`，则自动引用该 step 的 `content`。
- 只允许引用已经在列表中出现过的 step，Orchestrator 会在运行时展开这些引用并传给对应代理。

## ▶️ Run

single query:

```bash
python main.py --query "please help me write a VOCALOID song lyrics about summer and friendship"
```

interactive mode:

```bash
python main.py --interactive
```

attach MIDI and filters:

```bash
python main.py --query "Fill lyrics for this melody" --midi demo/midi_example.mid --payload-filters '{"tags_any": ["happy"]}'
```

control the finisher agent and stylistic hints:

```bash
python main.py --interactive --finisher composer --style-hint "shimmering future bass" --seed-lyrics "melody of neon rain"
```

save agent traces to JSON:

```bash
python main.py --query "推荐几首匹诺曹P的欢快歌曲" --trace-file traces/pinocchio.json
```
Interactive mode automatically appends `-turnN` to the filename so each turn is preserved separately.

While chatting interactively you can attach a new MIDI file at any time without restarting:

```
:attach midi /absolute/path/to/song.mid
```
The parser/composer agents will automatically use the most recently attached MIDI.

You can also调整 finisher 与提示信息：

```
:set finisher composer
:set style neon cyberpunk euphoria
:set format "light novel outline"
:set seed 『夜に溶ける青い願い』
```
这些指令会即时写入会话上下文，随后的代理调用都会读取最新设置。

### 自动推断 filters + 语义查询

当 Planner 未显式提供 `filters` 时，retriever 会根据 `query_text` 向 LLM 询问结构化过滤条件，并同步生成一个去除过滤信息后的 `semantic_query`（仅保留情绪/主题等语义信息）来进行语义检索。支持字段：`name`、`producers_*`、`vsingers_*`、`tags_*`、`year/month/favorite/length` 范围，以及 `rating`、`culture`。LLM 会自动把别称映射成官方 VocaDB 名称，例如：

```json
{
	"semantic_query": "lonely glitch pop ballads about empty cities",
	"filters": {
		"vsingers_any": ["初音ミク"],
		"producers_any": ["ピノキオピー"],
		"tags_any": ["lonely", "glitch pop"],
		"year_min": 2018
	}
}
```

因此只要 Planner 写好 `query_text`，retriever 就能拆分“语义 vs 过滤条件”，再用统一规范向 Qdrant 发起查询；若确实需要手动指定 filter，也务必直接填官方名称。

Run tests:

```bash
pytest
```