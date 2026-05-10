# tiny-boss

**Cheap LLM does the reading. Smart LLM makes the decisions.**

A cost-saving protocol where a small (free) LLM processes your full context and a larger LLM asks targeted questions to synthesize the final answer. The expensive model never processes the long context.

```
  ┌──────────────────┐
  │   SUPERVISOR     │  "What's the methodology?"
  │   (smart LLM)    │  "Got it. And the results?"
  │   DeepSeek V4    │  → synthesizes final answer
  └──────┬───────▲───┘
    asks  │       │  answers
  ┌──────▼───────┴───┐
  │     WORKER       │
  │   (cheap LLM)    │  Reads entire document.
  │   Groq 8B (free) │  Answers supervisor's questions.
  └──────────────────┘
```

Inspired by [HazyResearch/minions](https://github.com/HazyResearch/minions). No local GPU needed — both models run on cloud APIs.

## Install

```bash
pip install git+https://github.com/arshkumarsingh/tiny-boss.git
```

Or from source:

```bash
git clone https://github.com/arshkumarsingh/tiny-boss.git
cd tiny-boss
pip install -e .
```

## Quick Start

### CLI

```bash
tiny-boss \
  --worker groq/llama-3.1-8b-instant \
  --supervisor deepseek/deepseek-v4-pro \
  --task "Summarize the key findings" \
  --context paper.txt
```

### Python

```python
from tiny_boss import TinyBoss, get_client

worker = get_client("groq", "llama-3.1-8b-instant")
supervisor = get_client("deepseek", "deepseek-v4-pro")

boss = TinyBoss(worker, supervisor, max_rounds=3)
result = boss(task="Extract entities", context=open("doc.txt").read())

print(result.final_answer)
print(f"Rounds: {result.rounds_used}, Time: {result.timing['total']:.1f}s")
```

### API Keys

Set one or more. Auto-loaded from `~/.hermes/.env` for Hermes users.

```bash
export GROQ_API_KEY=gsk_...       # https://console.groq.com/keys
export DEEPSEEK_API_KEY=sk-...     # https://platform.deepseek.com/api_keys
export GEMINI_API_KEY=...          # https://aistudio.google.com/apikey
export OPENAI_API_KEY=sk-...       # https://platform.openai.com
export OPENROUTER_API_KEY=...      # https://openrouter.ai/keys
```

## Hermes Agent Integration

Run as a cost-saving provider inside Hermes Agent.

### 1. Start the proxy

```bash
tiny-boss-proxy \
  --worker groq/llama-3.1-8b-instant \
  --supervisor deepseek/deepseek-v4-pro
```

### 2. Register in Hermes config

Add to `~/.hermes/config.yaml` under `custom_providers`:

```yaml
custom_providers:
  tiny-boss:
    name: Tiny Boss (cost-saving)
    base_url: http://localhost:8765/v1
    api_key: no-key-needed
    provider: openai
    models:
      - tiny-boss
```

### 3. Use in Hermes

```bash
hermes --provider custom:tiny-boss --model tiny-boss -q "Summarize this..."
```

Or use `/model` in interactive mode and select `custom:tiny-boss`.

### Auto-start (systemd)

```ini
# ~/.config/systemd/user/tiny-boss.service
[Unit]
Description=Tiny Boss Proxy
After=network-online.target

[Service]
ExecStart=/path/to/venv/bin/python -m tiny_boss.proxy \
  --worker groq/llama-3.1-8b-instant \
  --supervisor deepseek/deepseek-v4-pro
Restart=on-failure
Environment=HOME=%h

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now tiny-boss
```

## Supported Providers

| Provider | Key env var | Free tier? | Example models |
|----------|------------|------------|----------------|
| `groq` | `GROQ_API_KEY` | Yes | `llama-3.1-8b-instant`, `llama-3.3-70b-versatile` |
| `deepseek` | `DEEPSEEK_API_KEY` | No (~$0.27/M) | `deepseek-chat`, `deepseek-v4-pro` |
| `gemini` | `GEMINI_API_KEY` | Yes | `gemini-2.0-flash`, `gemini-2.5-pro-exp-03-25` |
| `openai` | `OPENAI_API_KEY` | No | `gpt-5.5`, `gpt-4o-mini` |
| `anthropic` | `ANTHROPIC_API_KEY` | No | `claude-sonnet-4-6-20250514`, `claude-opus-4-7-20250514` |
| `openrouter` | `OPENROUTER_API_KEY` | Some free | `google/gemma-3-4b-it:free`, `anthropic/claude-sonnet-4` |

Add more by using `get_client("openai", model, base_url="...", api_key="...")` for any OpenAI-compatible endpoint.

## API Reference

### `get_client(provider, model, **kwargs) → LLMClient`

```python
get_client("groq", "llama-3.1-8b-instant")
get_client("deepseek", "deepseek-v4-pro")
get_client("gemini", "gemini-2.0-flash")
get_client("openai", "gpt-4o", api_key="sk-...")
get_client("openai", "custom-model", base_url="https://api.example.com/v1")
```

### `TinyBoss(worker, supervisor, max_rounds=3, verbose=False)`

```python
boss = TinyBoss(worker, supervisor, max_rounds=3)
result = boss(task="...", context="...")
# result.final_answer, result.rounds_used, result.timing, ...
```

### `BossResult`

| Field | Type | Description |
|-------|------|-------------|
| `final_answer` | `str` | The synthesized answer |
| `rounds_used` | `int` | Number of worker/supervisor rounds |
| `timing` | `dict` | Per-step timing in seconds |
| `worker_tokens` | `int` | Total tokens used by worker |
| `supervisor_tokens` | `int` | Total tokens used by supervisor |
| `worker_messages` | `list` | Worker Q&A history |
| `supervisor_messages` | `list` | Supervisor decisions |

## Why

Every time you send a long document to a frontier model, you pay for every token of context — even the 90% that's just filler. Tiny Boss flips this: a free/cheap model reads the whole thing, and your expensive model only sees short, targeted answers.

The math is simple:
- **Without Tiny Boss**: 10,000 tokens of context → all billed to the expensive model
- **With Tiny Boss**: 10,000 tokens go to Groq (free) + ~2,000 tokens go to the supervisor

## When to use

| Task type | Use Tiny Boss? | Why |
|-----------|---------------|-----|
| Summarize a long document | Yes | Context-heavy, mechanical |
| Extract entities / key facts | Yes | Worker can scan, supervisor structures |
| "Find where the paper discusses X" | Yes | Search task, not reasoning |
| Compare two papers | Yes | Two long contexts, worker handles both |
| Write original analysis / code | No | Needs the smart model throughout |
| Creative writing / brainstorming | No | No context to offload |
| Math / complex reasoning | No | Supervisor needs the full problem |
| Simple factual questions | No | Just ask normally, context is tiny |

Rule of thumb: if the context is long and the task is extractive, use Tiny Boss. If the task is generative or requires deep reasoning, use your main model directly.

## Savings: 3 real scenarios

All scenarios use **Groq Llama 8B (free tier) + DeepSeek V4 Pro**. Pricing as of May 2026 (DeepSeek V4 Pro at 75% introductory discount — $0.435/M input, $0.87/M output).

### Scenario 1: Summarize a research paper (12K tokens)

| | Without Tiny Boss | With Tiny Boss |
|---|---|---|
| Context processing | 12,000 tok × $0.435/M = $0.00522 | Groq free tier (no cost) |
| Supervisor work | 500 tok output × $0.87/M = $0.00044 | 1,500 input + 500 output = $0.00109 |
| **Total** | **$0.00566** | **$0.00109** |
| **Savings** | — | **81%** ($0.00457 saved per run) |

### Scenario 2: Extract all named entities from a legal document (8K tokens)

| | Without Tiny Boss | With Tiny Boss |
|---|---|---|
| Context processing | 8,000 tok × $0.435/M = $0.00348 | Groq free tier (no cost) |
| Supervisor work | 300 tok output × $0.87/M = $0.00026 | 1,000 input + 300 output = $0.00070 |
| **Total** | **$0.00374** | **$0.00070** |
| **Savings** | — | **81%** ($0.00304 saved per run) |

### Scenario 3: "What does this log file say about the error?" (20K tokens)

| | Without Tiny Boss | With Tiny Boss |
|---|---|---|
| Context processing | 20,000 tok × $0.435/M = $0.00870 | Groq free tier (no cost) |
| Supervisor work | 200 tok output × $0.87/M = $0.00017 | 800 input + 200 output = $0.00052 |
| **Total** | **$0.00887** | **$0.00052** |
| **Savings** | — | **94%** ($0.00835 saved per run) |

### At scale (1,000 runs/month)

| Scenario | Direct cost | Tiny Boss cost | Monthly savings |
|----------|------------|---------------|-----------------|
| Paper summaries | $5.66 | $1.09 | **$4.57** |
| Entity extraction | $3.74 | $0.70 | **$3.04** |
| Log analysis | $8.87 | $0.52 | **$8.35** |

## Savings across different model combos

Same scenario: summarize a 12K-token paper (2 protocol rounds, 500 tok final output).

### With a free worker (Groq Llama 8B)

Worker costs nothing. Supervisor only sees ~1,500 tokens of distilled answers.

| Supervisor | Pricing (per M tok) | Direct cost | Tiny Boss cost | Savings |
|------------|-------------------|------------|---------------|---------|
| **DeepSeek V4 Pro** | $0.435 in / $0.87 out | $0.0057 | $0.0011 | **81%** |
| **GPT-5.5** | $5.00 in / $30.00 out | $0.0750 | $0.0225 | **70%** |
| **Claude Sonnet 4.6** | $3.00 in / $15.00 out | $0.0435 | $0.0120 | **72%** |

The more expensive your supervisor, the more Tiny Boss saves. With GPT-5.5 at scale (1,000 papers/month): $75.00 → $22.50, saving $52.50/month.

### All-Claude stack (Opus 4.7 supervisor)

For teams locked into the Anthropic ecosystem. Worker isn't free, but Opus never reads the full context.

| Worker | Worker cost | Supervisor cost | Total | vs Direct Opus |
|--------|-----------|----------------|-------|----------------|
| — *(direct)* | — | Opus 4.7: 12K in + 500 out | **$0.073** | — |
| **Sonnet 4.6** ($3/$15) | 12K in + 400 out = $0.042 | Opus: 1.5K in + 500 out = $0.020 | **$0.062** | **14%** save |
| **Haiku 4.5** ($1/$5) | 12K in + 400 out = $0.014 | Opus: 1.5K in + 500 out = $0.020 | **$0.034** | **53%** save |

> **Pick the cheapest worker that can do the job.** Haiku handles extraction fine; Sonnet if the worker needs more reasoning. Free workers (Groq, Gemini Flash) give the best margins — 70–81%.

> **The pattern:** free workers (Groq, Gemini Flash) give the biggest savings. Paid workers still help — the expensive supervisor avoids reading the full context — but the worker's own cost eats into the margin. Pick the cheapest worker that can do the job.

> **Pricing sources (May 2026):** [DeepSeek](https://api-docs.deepseek.com/quick_start/pricing) (V4 Pro: 75% intro discount until May 31), [OpenAI](https://openai.com/api/pricing/) (GPT-5.5: Apr 2026), [Anthropic](https://platform.claude.com/docs/en/about-claude/pricing) (Opus 4.7: $5/$25, Sonnet 4.6: $3/$15, Haiku 4.5: $1/$5). Groq free tier: 30 RPM, 500K tokens/day — [limits](https://console.groq.com/settings/limits).

> **What about after the discount?** DeepSeek V4 Pro's regular pricing ($1.74/M input, $3.48/M output) would make savings even larger — 85–95% per run. The worker (Groq) has a generous free tier (30 requests/min, 500K tokens/day). You'll hit the free tier ceiling at ~40 paper summaries per day — well beyond individual use.

## License

MIT
