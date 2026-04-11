"""Server entry point for CAR-bench purple agent."""

import argparse
import os
import sys
import warnings
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings(
    "ignore",
    message=".*Pydantic serializer warnings.*",
    category=UserWarning,
    module="pydantic.main",
)

from a2a.server.apps import A2AStarletteApplication  # noqa: E402
from a2a.server.request_handlers import DefaultRequestHandler  # noqa: E402
from a2a.server.tasks import InMemoryTaskStore  # noqa: E402
from a2a.types import AgentCapabilities, AgentCard, AgentSkill  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from car_bench_agent import CARBenchAgentExecutor  # noqa: E402
from logging_utils import configure_logger  # noqa: E402

logger = configure_logger(role="agent", context="server")


def prepare_agent_card(url: str) -> AgentCard:
    skill = AgentSkill(
        id="car_assistant",
        name="In-Car Voice Assistant",
        description=(
            "In-car voice assistant with policy compliance, "
            "limit awareness, and disambiguation capabilities"
        ),
        tags=["benchmark", "car-bench", "voice-assistant"],
        examples=[],
    )
    return AgentCard(
        name="car_bench_purple_agent",
        description=(
            "CAR-bench purple agent. Single-pass architecture with a "
            "reasoning-capable LLM and a minimal, domain-agnostic policy "
            "compliance prompt."
        ),
        url=url,
        version="2.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run the CAR-bench purple agent."
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9009)
    parser.add_argument("--card-url", type=str, help="External URL for agent card")
    parser.add_argument("--agent-llm", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--thinking", action="store_true", default=None)
    parser.add_argument("--reasoning-effort", type=str, default=None)
    parser.add_argument("--interleaved-thinking", action="store_true", default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument(
        "--trace-dir",
        type=str,
        default=None,
        help="Directory for deep trace logs (also via AGENT_TRACE_DIR env var). "
             "Disabled if not set.",
    )
    args = parser.parse_args()

    # Resolve: CLI > env > defaults
    agent_llm = args.agent_llm or os.getenv("AGENT_LLM", "openai/gpt-5-mini")
    thinking = (
        args.thinking if args.thinking is not None
        else os.getenv("AGENT_THINKING", "true").lower() == "true"
    )
    reasoning_effort = args.reasoning_effort or os.getenv(
        "AGENT_REASONING_EFFORT", "medium"
    )
    temperature = (
        args.temperature if args.temperature is not None
        else float(os.getenv("AGENT_TEMPERATURE", "1.0"))
    )
    interleaved_thinking = (
        args.interleaved_thinking if args.interleaved_thinking is not None
        else os.getenv("AGENT_INTERLEAVED_THINKING", "true").lower() == "true"
    )
    timeout = (
        args.timeout if args.timeout is not None
        else int(os.getenv("AGENT_TIMEOUT", "120"))
    )
    trace_dir = args.trace_dir or os.getenv("AGENT_TRACE_DIR", None)

    logger.info(
        "Starting CAR-bench purple agent",
        model=agent_llm,
        temperature=temperature,
        thinking=thinking,
        reasoning_effort=reasoning_effort,
        interleaved_thinking=interleaved_thinking,
        timeout=timeout,
        trace_dir=trace_dir or "(disabled)",
        host=args.host,
        port=args.port,
    )

    card = prepare_agent_card(args.card_url or f"http://{args.host}:{args.port}/")

    request_handler = DefaultRequestHandler(
        agent_executor=CARBenchAgentExecutor(
            model=agent_llm,
            temperature=temperature,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            interleaved_thinking=interleaved_thinking,
            timeout=timeout,
            trace_dir=trace_dir,
        ),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(agent_card=card, http_handler=request_handler)
    uvicorn.run(app.build(), host=args.host, port=args.port, timeout_keep_alive=1000)


if __name__ == "__main__":
    main()
