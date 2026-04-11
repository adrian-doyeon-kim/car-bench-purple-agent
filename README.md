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

Running `gpt-5-mini` with `reasoning_effort=medium` and `temperature=1.0`,
the agent reaches **Pass¹ 86.7 %** on a 30-task subset of the public test
split and **Pass³ 83.3 %** (10 / 12) on a 12-task Pass³ mini split, with
Pass@3 = 100 %. The Pass@3 result means every remaining Pass³ miss came from
run-to-run variance rather than a task the model failed to understand.

## Overview

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

# Run locally (default port 9009 matches the agentbeats.dev runner)
uv run src/server.py --host 0.0.0.0 --port 9009

# Or with Docker (linux/amd64 required for agentbeats.dev)
docker build --platform linux/amd64 -t car-bench-purple-agent .
docker run -p 9009:9009 -e OPENAI_API_KEY=$OPENAI_API_KEY car-bench-purple-agent
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
