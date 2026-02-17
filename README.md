# CausalArmor + LangGraph Demo

A demonstration of [CausalArmor](https://github.com/prashantkul/causal-armor) protecting a LangGraph-based travel agent from **indirect prompt injection attacks**.

## Architecture

```
Agent LLM:      Gemini 2.0 Flash (Google API)  — drives tool-calling decisions
CausalArmor:
  Proxy:        VLLMProxyProvider + Gemma-3-12B-IT (local vLLM) — LOO log-prob scoring
  Sanitizer:    Gemini 2.5 Flash (Google API) — rewrites flagged spans
  Action regen: Gemini 2.5 Flash (Google API) — regenerates safe actions
```

All three CausalArmor model roles use real providers — the proxy scores log-probabilities via vLLM, while the sanitizer and action regenerator use Gemini 2.5 Flash.

## The Attack Scenario

A user asks the travel agent to review a PDF itinerary and book the best flight. The PDF contains a realistic travel plan with an embedded injection payload disguised as an airline security protocol that instructs the agent to call `send_money($5000)`.

### Without CausalArmor

```mermaid
graph LR
    U[User: Review my travel plan] --> LLM[Gemini 2.0 Flash]
    LLM -->|read_travel_plan| T1[Tool: Read PDF]
    T1 -->|Itinerary + INJECTION| LLM
    LLM -->|send_money 5000| T2[Tool: Send Money]
    T2 -->|Attack succeeds!| LLM

    style T2 fill:#ff6b6b,color:#fff
    style U fill:#4ecdc4,color:#fff
```

The agent follows the injected instruction and calls `send_money($5000)`.

### With CausalArmor

```mermaid
graph LR
    U[User: Review my travel plan] --> LLM[Gemini 2.0 Flash]
    LLM -->|read_travel_plan| T1[Tool: Read PDF]
    T1 -->|Itinerary + INJECTION| LLM
    LLM -->|send_money 5000| G[CausalArmor Guard]
    G -->|BLOCKED! book_flight| T3[Tool: Book Flight]
    T3 -->|Booking confirmed| LLM

    style G fill:#f9c74f,color:#333
    style T3 fill:#06d6a0,color:#fff
    style U fill:#4ecdc4,color:#fff
```

CausalArmor's vLLM proxy computes LOO attribution scores and detects that `send_money` is causally driven by the untrusted PDF content (not the user's request), blocks it, and regenerates a safe `book_flight` action.

## How It Works

CausalArmor uses **Leave-One-Out (LOO) causal attribution** via a proxy model (Gemma 3 12B on vLLM) to determine whether a proposed tool call is driven by the user's request or by untrusted tool output:

1. **Score** `P(action | full context)` — base probability
2. **Ablate user request** — if probability stays high, user isn't driving the action
3. **Ablate tool result** — if probability drops, the tool result is driving the action
4. **Detect dominance** — tool delta >> user delta means injection attack
5. **Defend** — sanitize, mask chain-of-thought, regenerate safe action

See [docs/how-it-works.md](docs/how-it-works.md) for the full walkthrough and [docs/example-attack-scenario.md](docs/example-attack-scenario.md) for a detailed real-trace walkthrough of both scenarios.

## Quick Start

### 1. Start vLLM (proxy model)

See [docs/vllm-setup.md](docs/vllm-setup.md) for detailed instructions.

```bash
docker run --runtime nvidia --gpus '"device=0"' \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
  -p 8000:8000 --ipc=host \
  vllm/vllm-openai:latest \
  --model google/gemma-3-12b-it --dtype auto --max-model-len 8192
```

### 2. Install dependencies

```bash
git clone https://github.com/prashantkul/causal-armor-langgraph-demo.git
cd causal-armor-langgraph-demo
uv sync
```

### 3. Generate the poisoned PDF

```bash
uv sync --extra dev
uv run python scripts/generate_pdf.py
```

### 4. Set environment variables

```bash
cp .env.example .env
# Edit .env with your keys:
#   GOOGLE_API_KEY=...          (Gemini 2.0 Flash)
#   VLLM_BASE_URL=http://...   (vLLM endpoint)
#   LANGCHAIN_API_KEY=lsv2_... (optional, for LangSmith tracing)
```

### 5. Run the demo

```bash
uv run python run_demo.py
```

The demo runs the agent twice — once with CausalArmor disabled (pass-through) and once enabled (real LOO scoring via vLLM) — then prints a comparison.

## LangSmith Tracing

Both runs are traced to [LangSmith](https://smith.langchain.com) when `LANGCHAIN_TRACING_V2=true` is set. Each run shows the full agent execution including LLM calls, tool invocations, and guard decisions.

Set these in your `.env`:

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_PROJECT=causal-armor-demo
```

## Project Structure

```
├── pyproject.toml              # Dependencies
├── run_demo.py                 # CLI runner — runs agent twice for comparison
├── scripts/
│   └── generate_pdf.py         # Generates poisoned travel_plan.pdf
├── data/
│   └── travel_plan.pdf         # Generated PDF with embedded injection
├── src/demo/
│   ├── tools.py                # read_travel_plan, book_flight, send_money
│   ├── state.py                # AgentState + GraphConfig (Studio toggle)
│   ├── adapters.py             # LangChain ↔ CausalArmor message converters
│   ├── guard.py                # CausalArmor guard node (vLLM + config toggle)
│   └── agent.py                # Single graph: Gemini LLM + configurable guard
└── docs/
    ├── how-it-works.md         # Detailed technical walkthrough
    └── vllm-setup.md           # DGX / vLLM setup instructions
```

## Graph Architecture

A single graph with the guard always present. The guard reads `causal_armor_enabled` from the configurable namespace — when `False` it passes through, when `True` it runs real LOO attribution.

```
START → LLM (Gemini 2.0 Flash) → should_continue → Guard → Tools → LLM → ... → END
```

In LangGraph Studio, `causal_armor_enabled` appears as a toggle in the UI configuration panel.

## Links

- [CausalArmor](https://github.com/prashantkul/causal-armor) — the core library
- [Paper: arXiv:2602.07918](https://arxiv.org/abs/2602.07918) — the research behind CausalArmor
