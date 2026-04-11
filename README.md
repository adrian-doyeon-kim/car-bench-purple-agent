# CAR-bench Purple Agent

Purple agent for the **AgentX-AgentBeats** competition — **CAR-bench** track.

Single-pass A2A agent with a domain-agnostic, policy-agnostic system prompt.
All rules the agent follows come from the instructions the green agent sends;
the agent itself has no hardcoded task knowledge, tool names, or policy
content.

## Architecture

```
Green Agent (Evaluator)  ◄──A2A──►  Purple Agent
  - Sends policies + tools           - Single LLM call per turn
  - Sends user messages              - Reasoning model (e.g. gpt-5-mini)
  - Executes tool calls              - 6 general agent rules only
  - Scores results                   - No hardcoded policies or tools
```

The agent is intentionally minimal:

- **No multi-stage pipeline**. Multi-agent / planner-executor pipelines hurt
  sequential, state-dependent tool-calling tasks (arXiv 2601.12307, 2604.02460).
- **No policy content in the prompt**. The six `CRITICAL RULES` describe *how*
  to follow instructions, never *what* those instructions contain.
- **No tool-name or domain-specific strings**. The prompt says "AI assistant"
  and "INSTRUCTIONS:", not "in-car voice assistant" or "vehicle policies".

## Quick Start

```bash
# Set your API key
export OPENAI_API_KEY="sk-..."

# Run locally
uv run src/server.py --host 0.0.0.0 --port 8080

# Or with Docker (linux/amd64 required for agentbeats.dev)
docker build --platform linux/amd64 -t car-bench-purple-agent .
docker run -p 8080:8080 -e OPENAI_API_KEY=$OPENAI_API_KEY car-bench-purple-agent
```

## Configuration

| Env Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / ... | — | API key for the chosen model provider (required) |
| `AGENT_LLM` | `openai/gpt-5-mini` | LiteLLM-compatible model identifier |
| `AGENT_THINKING` | `true` | Enable reasoning / extended thinking |
| `AGENT_REASONING_EFFORT` | `medium` | `low / medium / high` (reasoning models) or integer budget |
| `AGENT_TEMPERATURE` | `1.0` | Sampling temperature |
| `AGENT_INTERLEAVED_THINKING` | `true` | Interleaved thinking (Anthropic models) |
| `AGENT_TIMEOUT` | `120` | Per-LLM-call timeout in seconds |
| `AGENT_TRACE_DIR` | *(unset)* | If set, write deep per-turn trace markdown |

Any LiteLLM-compatible model works. With a reasoning model the `AGENT_THINKING`
path activates `reasoning_effort`; with Anthropic models it can also activate
`interleaved-thinking`.

## Layout

```
src/
  car_bench_agent.py   # Agent executor (single-pass, 6 rules)
  server.py            # A2A server entry point
  logging_utils.py     # Loguru + turn tracing
  tool_call_types.py   # Pydantic models for A2A tool calls
eval/
  run_pass3_eval.py    # Local evaluation harness
scenarios/
  scenario.toml        # Local dev scenario (purple+green on localhost)
Dockerfile             # linux/amd64 image for agentbeats.dev
.env.example           # Env var template
```

## Evaluation

Local evaluation uses the official CAR-bench green agent from
[CAR-bench/car-bench-agentbeats](https://github.com/CAR-bench/car-bench-agentbeats).

```bash
# 1. Clone the green agent repo next to this one
git clone https://github.com/CAR-bench/car-bench-agentbeats ../car-bench-agentbeats

# 2. Copy and fill the env template
cp .env.example .env  # then edit .env

# 3. Run smoke eval (~5 min, 6 subtypes × 1 trial)
uv run python eval/run_pass3_eval.py --smoke-test --start-purple

# 4. Run mini eval (12 tasks × 3 trials = 36 sessions, Pass^3)
uv run python eval/run_pass3_eval.py --mini-test
```

## Design Principles

The agent follows six rules, all domain-agnostic and policy-agnostic:

1. **Capability check** — verify a matching tool exists before acting; treat
   "similar" tools as non-substitutes; treat missing/unknown response fields
   as unavailable data.
2. **Policy compliance** — identify applicable rules from the instructions;
   verify prerequisites via information-gathering tools first.
3. **Resolve ambiguity** — follow whatever resolution procedure the
   instructions define; asking the user is a last resort, not a first choice.
4. **Gather then act** — information-gathering tools before state-changing
   tools; parallel calls when independent.
5. **Minimize state changes** — don't call a state-changing tool if the
   current state already matches.
6. **Output format** — follow any format rules in the instructions.

Plus an agent-persistence directive (keep going until the user's request is
fully resolved) drawn from the OpenAI GPT-5 prompting guide.

## License

MIT
