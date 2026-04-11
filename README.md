# CAR-bench Purple Agent

Purple agent for the **AgentX-AgentBeats** competition — **CAR-bench** track.

## Abstract

CAR-bench measures whether an in-car voice assistant stays consistent across
trials, obeys the policies it is given, resolves ambiguous requests through
the prescribed escalation path, and refuses actions it cannot perform. Its
headline metric, `Pass^k`, is the fraction of tasks that succeed on *all* `k`
independent trials, so a single stochastic slip collapses a task's score to
zero.

This submission is a deliberately minimal purple agent. It is a **single-pass
A2A executor**: one LLM call per turn, with the full tool catalogue and the
full policy text (as received from the green agent) visible to the model on
every call. There is no planner, no tool-selector, no separate policy
checker. The motivation is that the green agent already encodes every rule
the task needs; the purple agent's job is to obey those rules, and any
pipeline stage that re-summarises either the tool list or the policy text
risks dropping the one clause a later stage would have needed.

The system prompt contains six domain-agnostic rules plus an agent-persistence
directive: (1) capability check — never fabricate a tool or fake a result;
(2) policy compliance — verify prerequisites via information-gathering tools
before any state change; (3) resolve ambiguity via the procedure the
instructions define, treating a clarification question as a last resort; (4)
gather before act; (5) minimise state changes; (6) follow the output format
the instructions specify. The prompt contains no vehicle terminology, no
policy text, and no task identifiers; swapping in a different CAR-bench-style
benchmark would require changing nothing in the agent.

The agent runs on `gpt-5-mini` with `reasoning_effort=medium` and
`temperature=1.0`. Evaluated results are published on the
[CAR-bench leaderboard](https://github.com/RDI-Foundation/car-bench-agentbeats-leaderboard).

## Architecture

```
Green Agent (Evaluator)  ◄──A2A──►  Purple Agent
  - Sends policies + tools           - Single LLM call per turn
  - Sends user messages              - Reasoning model (e.g. gpt-5-mini)
  - Executes tool calls              - 6 general agent rules only
  - Scores results                   - No hardcoded policies or tools
```

## Quick Start

```bash
# Set your API key
export OPENAI_API_KEY="sk-..."

# Run locally (default port 9009 matches the agentbeats.dev runner)
uv run src/server.py --host 0.0.0.0 --port 9009

# Or pull the pre-built public image (linux/amd64)
docker run -p 9009:9009 -e OPENAI_API_KEY=$OPENAI_API_KEY \
  ghcr.io/adrian-doyeon-kim/car-bench-purple-agent:latest

# Or build the image locally
docker build --platform linux/amd64 -t car-bench-purple-agent .
docker run -p 9009:9009 -e OPENAI_API_KEY=$OPENAI_API_KEY car-bench-purple-agent
```

Verify the A2A agent card is reachable:

```bash
curl http://127.0.0.1:9009/.well-known/agent-card.json
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

Entry point: `src/server.py` (A2A server exposing `/.well-known/agent-card.json`).
Executor: `CARBenchAgentExecutor` in `src/car_bench_agent.py`.

```
src/
  car_bench_agent.py           # Agent executor (single-pass, 6 rules)
  server.py                    # A2A server entry point
  logging_utils.py             # Loguru + turn tracing
  tool_call_types.py           # Pydantic models for A2A tool calls
amber/
  amber-manifest-purple.json5  # Amber deployment manifest (agentbeats.dev)
eval/
  run_pass3_eval.py            # Local evaluation harness
scenarios/
  scenario.toml                # Local dev scenario
  scenario-leaderboard.toml    # Official leaderboard submission template
Dockerfile                     # linux/amd64 image for agentbeats.dev
.env.example                   # Env var template
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

## License

MIT
