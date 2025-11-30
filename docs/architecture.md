# VocaLyrics Multi-Agent Architecture

## Overview
VocaLyrics is built as a flexible multi-agent orchestration layer on top of OpenAI and Qdrant. The orchestrator decomposes user intents into actionable tasks, invokes specialized agents, and aggregates their results back into a single conversational response. Every agent is powered by a shared OpenAI client for reasoning and generation, while specialized tools (Qdrant search, MIDI parsing, etc.) are wrapped in small utility helpers for deterministic behavior.

```
User → Orchestrator → Planner → (Retriever | Analyst | Parser | Composer | Writer | General) → Response
                         ↓                                           ↑
                      Context store ← Task manager ←── Agent outputs ─┘
```

## Core building blocks

### Task manager (`core.task`)
* Creates unique task and sub-task identifiers.
* Tracks lifecycle (`pending`, `running`, `succeeded`, `failed`).
* Captures structured results, including references to retrieved documents or generated assets.

### Conversation context (`core.context`)
* Maintains ordered message history, scratchpad notes, and agent artifacts (tables, lyrics drafts, JSON snippets, etc.).
* Provides helpers for truncating history before sending to models, and for attaching auxiliary payloads (filters, MIDI digests, etc.).

### Orchestrator (`core.orchestrator`)
* Entry point for every user turn.
* Delegates planning to the planner agent, then executes sub-tasks sequentially with adaptive retries.
* Merges agent outputs and chooses a finisher agent (composer/writer/general) based on task metadata.
* Surfaces reasoning traces for debugging when verbose mode is enabled.

## Agent roster

| Agent | Purpose | Tooling |
| --- | --- | --- |
| `PlannerAgent` | Breaks user goals into structured steps (JSON plan) and selects responsible agents. | LLM only |
| `RetrieverAgent` | Queries Qdrant on song-level (`vocadb_songs`) or chunk-level (`vocadb_chunks`), applies payload filters, reranks, and returns normalized documents. | `utils.query.query` |
| `AnalystAgent` | Interprets lyrical themes, stylistic traits, producer signatures, and composes analytical reports referencing retrieved artifacts. | LLM + retrieved context |
| `ParserAgent` | Parses uploaded / local MIDI files into quantized note timelines that are easy for downstream writing agents to consume. | `utils.midi.parse_midi` |
| `ComposerAgent` | Generates, transforms, and continues lyrics given stylistic briefs, MIDI scaffolds, or prompt snippets. | LLM |
| `WriterAgent` | Handles non-lyric creative writing (worldbuilding, summaries, narrative framing) tied to VOCALOID themes. | LLM |
| `GeneralAgent` | Catch-all assistant for unsupported or very broad queries; ensures the chatbot always has an answer. | LLM |

All LLM-enabled agents share a common `_chat()` helper defined on the base class so that temperature, max tokens, and safety settings remain consistent.

## Control flow

1. **User turn received** – orchestrator stores the request and optional attachments (filters, MIDI path, prior artifacts).
2. **Planning** – planner returns JSON steps (`[{"id":"step-1","agent":"retriever","goal":"Find energetic PinocchioP songs",...}]`).
3. **Execution** – for each step the orchestrator:
   * creates a sub-task via the task manager,
   * calls the designated agent with the current conversation slice and task metadata,
   * captures structured `AgentResult` (text + artifacts + citations).
4. **Aggregation** – once all blocking steps finish, orchestrator picks a finisher agent (usually composer for lyric tasks, writer for prose, general otherwise) to craft the user-facing reply.
5. **Context update** – conversation context records the final response along with supporting evidence so that the next user turn can reference it.

## Error handling & resilience

* Every agent returns both natural-language output and a machine-friendly `AgentResult`. Failures surface as structured errors so the orchestrator can retry or pick a fallback agent.
* The retriever gracefully handles missing OpenAI clients by falling back to pure payload filters.
* Planner can downshift to a single-step plan if the query is simple, ensuring low latency.

## Extensibility

* New agents can be registered by subclassing `BaseAgent` and adding them to the orchestrator registry.
* Additional tools (e.g., automatic Vocaloid taggers) can be exposed through the shared `toolset` interface inside `AgentContext` without touching existing agents.

This document is a living reference—update it whenever the orchestration logic or agent roster meaningfully changes.
