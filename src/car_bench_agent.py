"""
CAR-bench Purple Agent — Single-Pass Architecture

One LLM call per turn with all tools + policies. No pipeline.
Research (arxiv 2601.12307, 2604.02460) shows multi-stage pipelines hurt
sequential state-dependent tasks. The strong model + single-pass approach
achieved the previous 88.9% baseline.
"""

import asyncio
import copy
import json
import sys
import time
from pathlib import Path

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import DataPart, Part, TextPart
from a2a.utils import new_agent_parts_message
from litellm import acompletion

sys.path.insert(0, str(Path(__file__).parent))
from logging_utils import configure_logger, TurnTracer  # noqa: E402
from tool_call_types import ToolCall, ToolCallsData  # noqa: E402

logger = configure_logger(role="agent", context="-")

MAX_RETRIES = 3
DEFAULT_LLM_TIMEOUT = 120


# ═══ System Prompt ═══════════════════════════════════════════════════════

META_SYSTEM_PROMPT = """\
You are an AI assistant. Help the user using ONLY the tools provided.

CRITICAL RULES:

1. CAPABILITY CHECK — before ANY action, verify a matching tool exists in your
   available tool list. If no tool exactly matches the required action, the
   action is IMPOSSIBLE — tell the user honestly that this capability is not
   available. A tool with a similar name or adjacent purpose is NOT a
   substitute: do not call it hoping it will achieve the missing action.
   NEVER fabricate tool results, pretend to perform actions, or claim an
   action was done when you only called a different tool.
   If a tool response contains missing fields, null values, or explicit
   uncertainty markers (e.g., "unknown") for data that a prerequisite check
   requires, treat that data as UNAVAILABLE. Do NOT assume default values
   and do NOT proceed with the action. Report the uncertainty to the user.

2. POLICY COMPLIANCE — before any state-changing action:
   - Identify ALL rules from the instructions below that apply.
   - Call the required information-gathering tools to verify ALL prerequisites FIRST.
   - Only proceed once ALL prerequisites are confirmed.
   - If a rule requires user confirmation under specific conditions, check
     those conditions first, then ask BEFORE executing.
   - NEVER skip a prerequisite check, even if you believe you know the state.

3. RESOLVE AMBIGUITY — when any tool parameter is unspecified or ambiguous,
   follow the resolution procedure defined in the instructions below exactly.
   Do not skip any step the instructions specify. Before asking the user for
   clarification, you MUST first exhaust every resolution mechanism the
   instructions provide (e.g., stored data, context, defaults). Asking the
   user is a last resort, not a first choice.

4. GATHER THEN ACT — always call information-gathering tools to check current
   state before calling state-changing tools. Call multiple independent
   information-gathering tools in parallel when possible.

5. MINIMIZE STATE CHANGES — only call state-changing tools when necessary.
   If the current state already matches the desired state (confirmed via
   information-gathering), do NOT call the state-changing tool again.

6. OUTPUT FORMAT — follow any response format requirements specified in the
   instructions below.

Agent persistence: Keep going until the user's request is fully resolved
before ending your turn. Plan before each tool call and reflect on previous
results. Do not guess or assume.

INSTRUCTIONS:
"""


# ═══ Helpers ═════════════════════════════════════════════════════════════

def _is_retryable(error: Exception) -> bool:
    msg = str(error).lower()
    return any(p in msg for p in (
        "rate_limit", "rate limit", "429", "timeout", "timed out",
        "server_error", "500", "502", "503", "529",
        "overloaded", "capacity", "connection", "network",
    ))


def _validate_tool_calls(tool_calls, available_tools):
    available = {t["function"]["name"] for t in available_tools}
    return [tc for tc in tool_calls if tc["function"]["name"] not in available]


# ═══ Debug Logger ════════════════════════════════════════════════════════

class DebugLogger:
    def __init__(self):
        self.logs: dict[str, list] = {}
        self.dir = Path("output/debug")
        self.dir.mkdir(parents=True, exist_ok=True)

    def log(self, ctx_id: str, event: str, **data):
        key = ctx_id[:8]
        if key not in self.logs:
            self.logs[key] = []
        self.logs[key].append({"event": event, "ts": time.time(), **data})
        try:
            with open(self.dir / f"{key}.json", "w") as f:
                json.dump(self.logs[key], f, indent=2, ensure_ascii=False)
        except Exception:
            pass


debug = DebugLogger()


# ═══ Agent Executor ══════════════════════════════════════════════════════

class CARBenchAgentExecutor(AgentExecutor):

    def __init__(
        self, model: str, temperature: float = 1.0,
        thinking: bool = True, reasoning_effort: str = "high",
        interleaved_thinking: bool = True,
        timeout: int = DEFAULT_LLM_TIMEOUT,
        trace_dir: str | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.interleaved_thinking = interleaved_thinking
        self.timeout = timeout
        self.trace_dir = trace_dir
        self.ctx_msgs: dict[str, list] = {}
        self.ctx_tools: dict[str, list] = {}
        self.ctx_tracer: dict[str, TurnTracer] = {}

    def _get_tracer(self, ctx_id):
        if not self.trace_dir:
            return None
        if ctx_id not in self.ctx_tracer:
            self.ctx_tracer[ctx_id] = TurnTracer(
                base_dir=self.trace_dir, context_id=ctx_id)
        return self.ctx_tracer[ctx_id]

    def _thinking_kwargs(self):
        """Build model-specific reasoning/thinking kwargs."""
        kw = {}
        if not self.thinking:
            return kw
        if self.model and "opus" in self.model:
            kw["thinking"] = {"type": "adaptive"}
        elif self.model and ("gpt-5" in self.model or "o1" in self.model
                              or "o3" in self.model or "o4" in self.model):
            # OpenAI reasoning models use reasoning_effort
            if self.reasoning_effort in ("minimal", "low", "medium", "high"):
                kw["reasoning_effort"] = self.reasoning_effort
        else:
            # Anthropic Claude (non-opus)
            e = self.reasoning_effort
            if e in ("none", "disable", "low", "medium", "high"):
                kw["reasoning_effort"] = e
            else:
                kw["thinking"] = {"type": "enabled",
                                  "budget_tokens": int(e)}
            if self.interleaved_thinking:
                kw["extra_headers"] = {
                    "anthropic-beta": "interleaved-thinking-2025-05-14"}
        return kw

    async def _call(self, msgs, extra, ctx_logger):
        """Main LLM call with retry."""
        kw = {"model": self.model, "temperature": self.temperature,
              "timeout": self.timeout, **self._thinking_kwargs()}
        kw.update(extra)
        last = None
        for i in range(MAX_RETRIES):
            try:
                return await acompletion(messages=msgs, **kw)
            except Exception as e:
                last = e
                if i < MAX_RETRIES - 1 and _is_retryable(e):
                    await asyncio.sleep(2 ** i)
                    ctx_logger.warning(f"Retry {i+1}: {e}")
                else:
                    raise
        raise last

    # ── Main Execute ──────────────────────────────────────────────────

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        ctx_id = context.context_id
        cl = logger.bind(role="agent", context=f"ctx:{ctx_id[:8]}")
        tracer = self._get_tracer(ctx_id)

        if ctx_id not in self.ctx_msgs:
            self.ctx_msgs[ctx_id] = []
        msgs = self.ctx_msgs[ctx_id]
        tools = self.ctx_tools.get(ctx_id, [])

        if tracer:
            tracer.new_turn()

        # ── 1. Parse inbound ──────────────────────────────────────────
        user_text = None
        tool_results = None

        try:
            for part in context.message.parts:
                if isinstance(part.root, TextPart):
                    txt = part.root.text
                    if "System:" in txt and "\n\nUser:" in txt:
                        sp = txt.split("\n\nUser:", 1)
                        policies = sp[0].replace("System:", "").strip()
                        user_text = sp[1].strip()
                        if not msgs:
                            msgs.append({
                                "role": "system",
                                "content": META_SYSTEM_PROMPT + policies,
                            })
                    else:
                        user_text = txt
                elif isinstance(part.root, DataPart):
                    d = part.root.data
                    if "tools" in d:
                        tools = d["tools"]
                        self.ctx_tools[ctx_id] = tools
                    elif "tool_results" in d:
                        tool_results = d["tool_results"]

            if not user_text and not tool_results:
                user_text = context.get_user_input()
            if user_text is not None and not user_text.strip():
                user_text = "none"

            cl.info("In", turn=len(msgs) + 1,
                    preview=user_text[:80] if user_text else "")
        except Exception as e:
            cl.warning(f"Parse: {e}")
            user_text = context.get_user_input() or "none"

        if tracer:
            try:
                tracer.capture_phase1(context.message.parts)
            except Exception:
                pass

        # ── 2. Update message history ─────────────────────────────────
        if (msgs and msgs[-1].get("role") == "assistant"
                and msgs[-1].get("tool_calls")):
            prev = msgs[-1]["tool_calls"]
            if tool_results:
                by_name: dict[str, list] = {}
                for tc in prev:
                    by_name.setdefault(
                        tc["function"]["name"], []).append(tc)
                for tr in tool_results:
                    nm = tr.get("tool_name", "")
                    m = by_name.get(nm, [])
                    if m:
                        msgs.append({
                            "role": "tool",
                            "tool_call_id": m.pop(0)["id"],
                            "content": tr.get("content", ""),
                        })
                    else:
                        msgs.append({
                            "role": "tool",
                            "tool_call_id": tr.get(
                                "tool_call_id", f"u_{nm}"),
                            "content": tr.get("content", ""),
                        })
            else:
                for tc in prev:
                    msgs.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": user_text or "none",
                    })
        else:
            msgs.append({"role": "user", "content": user_text or "none"})

        # ── 3. Build LLM call ─────────────────────────────────────────
        exe_tools = copy.deepcopy(tools) if tools else None
        if exe_tools:
            exe_tools[-1]["cache_control"] = {"type": "ephemeral"}

        exe_msgs = copy.deepcopy(msgs)
        if exe_msgs:
            exe_msgs[0]["cache_control"] = {"type": "ephemeral"}
        for m in exe_msgs:
            m.pop("thinking_blocks", None)
            m.pop("reasoning_content", None)

        debug.log(ctx_id, "exec",
                  turn=len(msgs),
                  n_tools=len(exe_tools) if exe_tools else 0)

        if tracer:
            try:
                tracer.capture_phase2(exe_msgs, exe_tools, {})
            except Exception:
                pass

        # ── 4. Single LLM call ────────────────────────────────────────
        err = None
        try:
            t0 = time.time()
            resp = await self._call(
                exe_msgs, {"tools": exe_tools}, cl)
            ms = (time.time() - t0) * 1000
            out = resp.choices[0].message.model_dump(exclude_unset=True)
            tc = out.get("tool_calls")
            cl.info("LLM", tools=bool(tc),
                    n=len(tc) if tc else 0, ms=f"{ms:.0f}")
            debug.log(ctx_id, "llm",
                      calls=[c["function"]["name"] for c in (tc or [])],
                      resp=(out.get("content") or "")[:200])

            # ── 5. Verify tool calls exist ────────────────────────────
            if tc:
                bad = _validate_tool_calls(tc, tools)
                if bad:
                    bad_n = [c["function"]["name"] for c in bad]
                    cl.warning(f"Invalid tool calls removed: {bad_n}")
                    debug.log(ctx_id, "invalid", names=bad_n)
                    good = {c["id"] for c in tc} - {c["id"] for c in bad}
                    tc = [c for c in tc if c["id"] in good]
                    out["tool_calls"] = tc or None

            if tracer:
                try:
                    tracer.capture_phase3(out, ms)
                except Exception:
                    pass

            # ── 6. Build A2A response ─────────────────────────────────
            parts: list[Part] = []
            tc = out.get("tool_calls")
            # When tool_calls present, suppress text (prevents internal reasoning leak)
            if out.get("content") and not tc:
                parts.append(Part(root=TextPart(
                    kind="text", text=out["content"])))
            if tc:
                tcl = []
                for c in tc:
                    fn = c["function"]["name"]
                    raw = c["function"]["arguments"]
                    try:
                        a = json.loads(raw) if isinstance(raw, str) else raw
                    except (json.JSONDecodeError, TypeError):
                        try:
                            fx = raw.rstrip().rstrip(",") + "}"
                            if not fx.startswith("{"):
                                fx = "{" + fx
                            a = json.loads(fx)
                        except Exception:
                            continue
                    tcl.append(ToolCall(tool_name=fn, arguments=a))
                if tcl:
                    parts.append(Part(root=DataPart(
                        kind="data",
                        data=ToolCallsData(tool_calls=tcl).model_dump())))
            if out.get("reasoning_content"):
                parts.append(Part(root=DataPart(
                    kind="data",
                    data={"reasoning_content":
                          out["reasoning_content"]})))
            if not parts:
                parts.append(Part(root=TextPart(
                    kind="text", text=out.get("content", ""))))

        except Exception as e:
            cl.error(f"LLM: {e}")
            err = str(e)
            parts = [Part(root=TextPart(
                kind="text", text="Could you repeat that?"))]
            out = {"content": "Could you repeat that?"}

        if tracer:
            try:
                tracer.capture_phase4(parts, error=err)
            except Exception:
                pass

        # ── 7. Save history + Send ────────────────────────────────────
        entry = {"role": "assistant",
                 "content": out.get("content") or ""}
        if out.get("tool_calls"):
            entry["tool_calls"] = out["tool_calls"]
        if out.get("thinking_blocks"):
            entry["thinking_blocks"] = out["thinking_blocks"]
        if out.get("reasoning_content"):
            entry["reasoning_content"] = out["reasoning_content"]
        msgs.append(entry)

        await event_queue.enqueue_event(
            new_agent_parts_message(parts=parts, context_id=ctx_id))
        if tracer:
            asyncio.create_task(tracer.flush_turn())

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        cid = context.context_id
        t = self.ctx_tracer.get(cid)
        if t:
            await t.flush_meta()
        for s in (self.ctx_msgs, self.ctx_tools, self.ctx_tracer):
            s.pop(cid, None)
