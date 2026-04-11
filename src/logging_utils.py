"""
Centralized logging + deep turn-level tracing for CAR-bench.

TurnTracer writes per-turn markdown trace files:
  logs/traces/{context_id}/turn_001_trace.md

Each trace contains 4 phases:
  Phase 1: Green Agent Request (inbound A2A message)
  Phase 2: LLM Context Input (messages + tools sent to Sonnet 4.6)
  Phase 3: LLM Raw Response (thinking blocks + content + tool calls)
  Phase 4: Purple Agent Response (final A2A message to Green Agent)
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger


# ── Standard logger ────────────────────────────────────────────────────

def configure_logger(role: str, context: str = None, serialize: bool = False):
    """Configure loguru logger for structured logging."""
    logger.remove()

    if serialize or os.getenv("LOG_FORMAT") == "json":
        logger.add(
            sys.stderr,
            format="{message}",
            level=os.getenv("LOGURU_LEVEL", "INFO"),
            serialize=True,
        )
    else:
        def format_with_extras(record):
            time_str = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green>"
            level_str = "<level>{level: <8}</level>"

            if "context" in record["extra"]:
                base = (
                    f"{time_str} | {level_str} | "
                    f"<cyan>{{extra[role]}}</cyan> | "
                    f"<cyan>{{extra[context]}}</cyan> | "
                    f"<level>{{message}}</level>"
                )
            else:
                base = (
                    f"{time_str} | {level_str} | "
                    f"<cyan>{{extra[role]}}</cyan> | "
                    f"<level>{{message}}</level>"
                )

            if record["level"].name == "DEBUG":
                extra_fields = {
                    k: v for k, v in record["extra"].items()
                    if k not in ("role", "context")
                }
                if extra_fields:
                    extras = []
                    for k, v in extra_fields.items():
                        if isinstance(v, str):
                            v_safe = v.replace("{", "{{").replace("}", "}}")
                            extras.append(f"{k}={v_safe}")
                        elif isinstance(v, (dict, list)):
                            v_str = json.dumps(v)
                            v_safe = v_str.replace("{", "{{").replace("}", "}}")
                            extras.append(f"{k}={v_safe}")
                        else:
                            extras.append(f"{k}={v}")
                    return base + " | " + " | ".join(extras) + "\n"
            return base + "\n"

        logger.add(
            sys.stderr,
            format=format_with_extras,
            level=os.getenv("LOGURU_LEVEL", "INFO"),
            colorize=True,
        )

    if context:
        return logger.bind(role=role, context=context)
    return logger.bind(role=role)


# ── Deep Turn Tracer ───────────────────────────────────────────────────

def _truncate(text: str, max_len: int = 500) -> str:
    if not text or len(text) <= max_len:
        return text or ""
    return text[:max_len] + f"... ({len(text)} chars total)"


def _json_pretty(obj, max_len: int = 5000) -> str:
    try:
        raw = json.dumps(obj, indent=2, ensure_ascii=False, default=str)
        if len(raw) > max_len:
            return raw[:max_len] + f"\n... (truncated, {len(raw)} chars total)"
        return raw
    except Exception:
        return str(obj)[:max_len]


def _serialize_parts(parts) -> list[dict]:
    """Serialize A2A Part objects to plain dicts for logging."""
    from a2a.types import TextPart, DataPart

    result = []
    for part in parts:
        root = part.root if hasattr(part, "root") else part
        if isinstance(root, TextPart):
            result.append({"type": "text", "text": root.text})
        elif isinstance(root, DataPart):
            result.append({"type": "data", "data": root.data})
        else:
            result.append({"type": "unknown", "repr": str(root)[:200]})
    return result


class TurnTracer:
    """Writes per-turn markdown trace files for deep post-mortem analysis.

    Usage:
        tracer = TurnTracer(base_dir="logs", context_id="abc123...")
        tracer.new_turn()
        tracer.capture_phase1(inbound_parts)
        tracer.capture_phase2(messages, tools, completion_kwargs)
        tracer.capture_phase3(assistant_content, duration_ms)
        tracer.capture_phase4(response_parts, error=None)
        await tracer.flush_turn()
    """

    def __init__(self, base_dir: str | Path, context_id: str):
        self.base_dir = Path(base_dir)
        self.context_id = context_id
        self.short_id = context_id[:12]
        self.session_dir = self.base_dir / "traces" / self.short_id
        self.turn = 0
        self._phases: dict[str, str] = {}
        self._turn_start: float = 0
        self._llm_start: float = 0

    def new_turn(self):
        """Start a new turn. Call before phase captures."""
        self.turn += 1
        self._phases = {}
        self._turn_start = time.time()

    # ── Phase captures (sync, memory-only) ─────────────────────────

    def capture_phase1(self, raw_parts: list) -> None:
        """Phase 1: Green Agent Request — the inbound A2A message."""
        try:
            serialized = _serialize_parts(raw_parts)
            lines = [
                "## Phase 1: Green Agent Request\n",
            ]

            for i, part in enumerate(serialized):
                if part["type"] == "text":
                    text = part["text"]
                    if "System:" in text and "User:" in text:
                        # Split system/user for readability
                        split = text.split("\n\nUser:", 1)
                        sys_part = split[0].replace("System:", "").strip()
                        usr_part = split[1].strip() if len(split) > 1 else ""
                        lines.append(
                            f"### System Prompt\n\n"
                            f"<details>\n<summary>System prompt "
                            f"({len(sys_part)} chars)</summary>\n\n"
                            f"```\n{sys_part}\n```\n\n</details>\n"
                        )
                        lines.append(
                            f"### User Message\n\n> {_truncate(usr_part, 1000)}\n"
                        )
                    else:
                        lines.append(f"### Text Part {i+1}\n\n> {_truncate(text, 1000)}\n")

                elif part["type"] == "data":
                    data = part["data"]
                    if "tools" in data:
                        tools = data["tools"]
                        names = [t.get("function", {}).get("name", "?") for t in tools]
                        lines.append(
                            f"### Tools ({len(tools)} available)\n\n"
                            f"`{', '.join(names[:20])}`"
                            f"{'...' if len(names) > 20 else ''}\n"
                        )
                    elif "tool_results" in data:
                        results = data["tool_results"]
                        lines.append(f"### Tool Results ({len(results)})\n\n")
                        for tr in results:
                            name = tr.get("tool_name", "?")
                            content = _truncate(tr.get("content", ""), 500)
                            lines.append(
                                f"**{name}**:\n```json\n{content}\n```\n"
                            )
                    else:
                        lines.append(
                            f"### Data Part {i+1}\n\n"
                            f"```json\n{_json_pretty(data, 2000)}\n```\n"
                        )

            self._phases["phase1"] = "\n".join(lines)
        except Exception as e:
            self._phases["phase1"] = f"## Phase 1: Green Agent Request\n\n*Capture error: {e}*\n"

    def capture_phase2(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        completion_kwargs: dict,
    ) -> None:
        """Phase 2: LLM Context Input — what we send to Sonnet 4.6."""
        self._llm_start = time.time()
        try:
            model = completion_kwargs.get("model", "?")
            temp = completion_kwargs.get("temperature", "?")
            thinking = completion_kwargs.get("reasoning_effort") or completion_kwargs.get("thinking", "off")
            timeout = completion_kwargs.get("timeout", "?")
            n_tools = len(tools) if tools else 0

            lines = [
                "## Phase 2: LLM Context (Input)\n",
                f"| Param | Value |",
                f"|-------|-------|",
                f"| Model | `{model}` |",
                f"| Temperature | {temp} |",
                f"| Thinking | {thinking} |",
                f"| Timeout | {timeout}s |",
                f"| Messages | {len(messages)} |",
                f"| Tools | {n_tools} |",
                "",
            ]

            # Messages summary table
            lines.append("### Messages\n")
            lines.append("| # | Role | Preview |")
            lines.append("|---|------|---------|")

            for i, msg in enumerate(messages):
                role = msg.get("role", "?")
                content = msg.get("content") or ""
                tc = msg.get("tool_calls")

                if tc:
                    tc_names = [c["function"]["name"] for c in tc]
                    preview = f"[{len(tc)} tool calls: {', '.join(tc_names[:3])}]"
                elif role == "system":
                    preview = _truncate(content, 80)
                else:
                    preview = _truncate(content, 120)

                # Escape pipes for markdown table
                preview = preview.replace("|", "\\|").replace("\n", " ")
                lines.append(f"| {i+1} | {role} | {preview} |")

            # Full system prompt in collapsible
            if messages and messages[0].get("role") == "system":
                sys_content = messages[0].get("content", "")
                lines.append(
                    f"\n<details>\n"
                    f"<summary>Full system prompt ({len(sys_content)} chars)</summary>\n\n"
                    f"```\n{sys_content}\n```\n\n</details>\n"
                )

            self._phases["phase2"] = "\n".join(lines)
        except Exception as e:
            self._phases["phase2"] = f"## Phase 2: LLM Context\n\n*Capture error: {e}*\n"

    def capture_phase3(self, assistant_content: dict, duration_ms: float = 0) -> None:
        """Phase 3: LLM Raw Response — thinking blocks, content, tool calls."""
        try:
            lines = [
                "## Phase 3: LLM Raw Response\n",
                f"**Duration**: {duration_ms:.0f}ms\n",
            ]

            # Thinking / reasoning blocks
            thinking = assistant_content.get("thinking_blocks")
            reasoning = assistant_content.get("reasoning_content")

            if thinking:
                lines.append("### Thinking Blocks\n")
                if isinstance(thinking, list):
                    for i, block in enumerate(thinking):
                        text = block.get("thinking", "") if isinstance(block, dict) else str(block)
                        lines.append(
                            f"<details>\n<summary>Thinking block {i+1} "
                            f"({len(text)} chars)</summary>\n\n"
                            f"```\n{text}\n```\n\n</details>\n"
                        )
                elif isinstance(thinking, str):
                    lines.append(f"```\n{thinking}\n```\n")

            if reasoning:
                if isinstance(reasoning, str):
                    lines.append(
                        f"### Reasoning Content\n\n"
                        f"<details>\n<summary>Reasoning "
                        f"({len(reasoning)} chars)</summary>\n\n"
                        f"```\n{reasoning}\n```\n\n</details>\n"
                    )
                elif isinstance(reasoning, list):
                    for i, item in enumerate(reasoning):
                        text = item.get("thinking", "") if isinstance(item, dict) else str(item)
                        if text:
                            lines.append(
                                f"<details>\n<summary>Reasoning block {i+1}</summary>\n\n"
                                f"```\n{text}\n```\n\n</details>\n"
                            )

            # Text content
            content = assistant_content.get("content")
            if content:
                lines.append(f"### Content\n\n{content}\n")
            else:
                lines.append("### Content\n\n*(no text content)*\n")

            # Tool calls
            tool_calls = assistant_content.get("tool_calls")
            if tool_calls:
                lines.append(f"### Tool Calls ({len(tool_calls)})\n")
                lines.append("| # | Tool | Arguments |")
                lines.append("|---|------|-----------|")
                for i, tc in enumerate(tool_calls):
                    fn = tc.get("function", {})
                    name = fn.get("name", "?")
                    args_raw = fn.get("arguments", "{}")
                    try:
                        args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                        args_str = json.dumps(args, ensure_ascii=False)
                    except Exception:
                        args_str = str(args_raw)
                    args_preview = _truncate(args_str, 200).replace("|", "\\|")
                    lines.append(f"| {i+1} | `{name}` | `{args_preview}` |")
                lines.append("")
            else:
                lines.append("### Tool Calls\n\n*(none — text-only response)*\n")

            self._phases["phase3"] = "\n".join(lines)
        except Exception as e:
            self._phases["phase3"] = f"## Phase 3: LLM Raw Response\n\n*Capture error: {e}*\n"

    def capture_phase4(self, response_parts: list, error: str | None = None) -> None:
        """Phase 4: Purple Agent Response — final A2A message to Green Agent."""
        try:
            lines = ["## Phase 4: Purple Agent Response\n"]

            if error:
                lines.append(
                    f'<div style="background:#2d0000; padding:8px; '
                    f'border-left:4px solid red; margin:8px 0;">\n\n'
                    f"**ERROR**\n\n"
                    f"```\n{error}\n```\n\n</div>\n"
                )

            serialized = _serialize_parts(response_parts)
            for i, part in enumerate(serialized):
                if part["type"] == "text":
                    lines.append(f"### Text Response\n\n> {part['text']}\n")
                elif part["type"] == "data":
                    data = part["data"]
                    if "tool_calls" in data:
                        tc_list = data["tool_calls"]
                        lines.append(f"### Tool Calls Sent ({len(tc_list)})\n\n")
                        lines.append(f"```json\n{_json_pretty(tc_list, 3000)}\n```\n")
                    elif "reasoning_content" in data:
                        lines.append("### Reasoning (debug, not scored)\n\n*(included as DataPart)*\n")
                    else:
                        lines.append(f"### Data Part {i+1}\n\n```json\n{_json_pretty(data, 2000)}\n```\n")

            elapsed = (time.time() - self._turn_start) * 1000 if self._turn_start else 0
            lines.append(f"\n---\n*Turn total: {elapsed:.0f}ms*\n")

            self._phases["phase4"] = "\n".join(lines)
        except Exception as e:
            self._phases["phase4"] = f"## Phase 4: Purple Agent Response\n\n*Capture error: {e}*\n"

    # ── Flush to disk (async) ──────────────────────────────────────

    async def flush_turn(self) -> None:
        """Write the accumulated trace to a markdown file (non-blocking)."""
        md = self._render_markdown()
        try:
            await asyncio.to_thread(self._sync_write_turn, md)
        except Exception as e:
            # NEVER crash the main evaluation flow
            logger.warning(f"Trace write failed for turn {self.turn}: {e}")

    async def flush_meta(self) -> None:
        """Write session metadata JSON."""
        meta = {
            "context_id": self.context_id,
            "short_id": self.short_id,
            "total_turns": self.turn,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            await asyncio.to_thread(self._sync_write_meta, meta)
        except Exception:
            pass

    def _render_markdown(self) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        header = (
            f"# Turn {self.turn} — `ctx:{self.short_id}`\n\n"
            f"**Timestamp**: {ts}\n\n---\n\n"
        )
        body = "\n---\n\n".join(
            self._phases.get(f"phase{i}", f"## Phase {i}\n\n*(not captured)*\n")
            for i in range(1, 5)
        )
        return header + body

    def _sync_write_turn(self, content: str) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        path = self.session_dir / f"turn_{self.turn:03d}_trace.md"
        path.write_text(content, encoding="utf-8")

    def _sync_write_meta(self, meta: dict) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        path = self.session_dir / "_meta.json"
        path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
