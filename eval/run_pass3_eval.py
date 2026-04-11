#!/usr/bin/env python3
"""
Pass^3 Evaluation Harness for CAR-bench Purple Agent

Uses the OFFICIAL CAR-bench Green Agent for evaluation.
Runs 10 Base + 10 Hallucination + 10 Disambiguation tasks × 3 trials = 90 sessions.

Usage:
  # 1. Start Purple Agent in one terminal:
  cd /path/to/agentx_trial && uv run src/server.py --port 9010

  # 2. Run evaluation in another terminal:
  python eval/run_pass3_eval.py

  # Or let the script start Purple Agent automatically:
  python eval/run_pass3_eval.py --start-purple

  # Use with Docker:
  docker compose -f docker-compose.test.yml up -d
  python eval/run_pass3_eval.py --purple-url http://localhost:9010
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAR_BENCH_AGENTBEATS = PROJECT_ROOT / "car-bench-agentbeats"
GREEN_SERVER = CAR_BENCH_AGENTBEATS / "src" / "green_car_bench_agent" / "server.py"
PURPLE_SERVER = PROJECT_ROOT / "src" / "server.py"

# ── Default 10/10/10 task IDs (train split) ────────────────────────────
TASK_IDS = {
    "base": [f"base_{i}" for i in range(0, 20, 2)],             # 10 tasks
    "hallucination": [f"hallucination_{i}" for i in range(0, 20, 2)],  # 10 tasks
    "disambiguation": [f"disambiguation_{i}" for i in range(0, 20, 2)],  # 10 tasks
}

# Smoke test: 1 per subtype, 1 trial — 6 sessions total
# h: missing_tool, missing_tool_parameter, missing_tool_response
# d: internal, user
SMOKE_TASK_IDS = {
    "base": ["base_0"],
    "hallucination": ["hallucination_0", "hallucination_6", "hallucination_10"],
    "disambiguation": ["disambiguation_0", "disambiguation_2"],
}

# Mini test: covers all subtypes, 1 trial — 12 sessions total
# hallucination: missing_tool(0,18), missing_tool_parameter(6,14), missing_tool_response(10)
# disambiguation: internal(0,10), user(2,20)
MINI_TASK_IDS = {
    "base": ["base_0", "base_2", "base_14"],
    "hallucination": ["hallucination_0", "hallucination_18",
                       "hallucination_6", "hallucination_14",
                       "hallucination_10"],
    "disambiguation": ["disambiguation_0", "disambiguation_10",
                        "disambiguation_2", "disambiguation_20"],
}


# ── Scenario TOML generation ──────────────────────────────────────────

def generate_scenario_toml(
    purple_url: str,
    green_port: int,
    num_trials: int,
    task_split: str,
    task_ids: dict[str, list[str]],
) -> str:
    """Generate a scenario TOML for the official evaluator."""
    def fmt_list(lst):
        return "[" + ", ".join(f'"{x}"' for x in lst) + "]"

    return f"""\
[green_agent]
endpoint = "http://127.0.0.1:{green_port}"

[[participants]]
role = "agent"
endpoint = "{purple_url}"

[config]
num_trials = {num_trials}
task_split = "{task_split}"
tasks_base_task_id_filter = {fmt_list(task_ids["base"])}
tasks_hallucination_task_id_filter = {fmt_list(task_ids["hallucination"])}
tasks_disambiguation_task_id_filter = {fmt_list(task_ids["disambiguation"])}
max_steps = 50
"""


# ── Process management ─────────────────────────────────────────────────

def wait_for_server(url: str, timeout: int = 120, label: str = "server") -> bool:
    """Poll a server's agent card endpoint until it responds."""
    import httpx

    start = time.time()
    while time.time() - start < timeout:
        try:
            # Try both old and new agent card paths
            for path in ["/.well-known/agent-card.json", "/.well-known/agent.json"]:
                r = httpx.get(url.rstrip("/") + path, timeout=5)
                if r.status_code == 200:
                    print(f"  [{label}] Ready at {url}")
                    return True
        except Exception:
            pass
        time.sleep(1)
    print(f"  [{label}] TIMEOUT after {timeout}s")
    return False


def start_process(cmd: list[str], cwd: str, label: str, show_logs: bool) -> subprocess.Popen:
    """Start a subprocess with optional log visibility."""
    env = os.environ.copy()
    sink = None if show_logs else subprocess.DEVNULL
    print(f"  [{label}] Starting: {' '.join(cmd[:4])}...")
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=sink,
        stderr=sink,
        start_new_session=True,
    )
    return proc


def kill_process(proc: subprocess.Popen, label: str):
    """Gracefully kill a process group."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    print(f"  [{label}] Stopped")


# ── Green Agent setup ──────────────────────────────────────────────────

def ensure_green_agent_deps():
    """Install car-bench-agentbeats dependencies if needed."""
    venv_marker = CAR_BENCH_AGENTBEATS / ".venv"
    car_bench_data = CAR_BENCH_AGENTBEATS / "scenarios" / "car-bench" / "car-bench"

    if not car_bench_data.exists():
        print("[setup] Cloning car-bench data...")
        subprocess.run(
            ["bash", "scenarios/car-bench/setup.sh"],
            cwd=str(CAR_BENCH_AGENTBEATS),
            check=True,
        )

    if not venv_marker.exists():
        print("[setup] Installing car-bench-agentbeats dependencies...")
        subprocess.run(
            ["uv", "sync", "--extra", "car-bench-agent", "--extra", "car-bench-evaluator"],
            cwd=str(CAR_BENCH_AGENTBEATS),
            check=True,
        )
    else:
        print("[setup] car-bench-agentbeats deps already installed")


# ── Evaluation runner ──────────────────────────────────────────────────

def run_evaluation(
    scenario_path: Path,
    output_path: Path,
    show_logs: bool,
) -> bool:
    """Run the official evaluator via agentbeats client_cli."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "uv", "run", "python", "-m", "agentbeats.client_cli",
        str(scenario_path),
        str(output_path),
    ]

    print(f"\n{'='*60}")
    print(f"[eval] Running evaluation...")
    print(f"[eval] Scenario: {scenario_path}")
    print(f"[eval] Output:   {output_path}")
    print(f"{'='*60}\n")

    result = subprocess.run(
        cmd,
        cwd=str(CAR_BENCH_AGENTBEATS),
        capture_output=not show_logs,
        text=True,
    )

    if result.returncode != 0:
        print(f"[eval] Evaluation failed (exit code {result.returncode})")
        if not show_logs and result.stderr:
            print(f"[eval] stderr:\n{result.stderr[-2000:]}")
        return False

    return True


# ── Results parsing & reporting ────────────────────────────────────────

def parse_results(output_path: Path) -> dict | None:
    """Parse the official evaluator's output JSON."""
    if not output_path.exists():
        print(f"[report] Results file not found: {output_path}")
        return None

    with open(output_path) as f:
        data = json.load(f)

    # The client_cli wraps results in {"participants": ..., "results": [...]}
    results_list = data.get("results", [])

    # Find the main result dict (contains "detailed_results_by_split")
    for item in results_list:
        if isinstance(item, dict) and "detailed_results_by_split" in item:
            return item

    # If results_list is the data itself
    if isinstance(data, dict) and "detailed_results_by_split" in data:
        return data

    print("[report] Could not find detailed results in output JSON")
    return None


def compute_per_task_pass(detailed_results: dict) -> dict:
    """Compute per-task pass/fail across trials.

    Returns: {
        split: {
            task_id: {
                "trials": [reward1, reward2, reward3],
                "pass_all": bool,   # Pass^3
                "pass_any": bool,   # Pass@3
                "failure_reasons": [str, ...]
            }
        }
    }
    """
    per_task = {}

    for split, tasks in detailed_results.items():
        per_task[split] = {}
        # Group by task_id
        task_groups: dict[str, list] = {}
        for task_result in tasks:
            tid = task_result["task_id"]
            task_groups.setdefault(tid, []).append(task_result)

        for tid, trials in task_groups.items():
            rewards = [t["reward"] for t in trials]
            pass_threshold = 0.99

            # Collect failure reasons from reward_info
            failure_reasons = []
            for t in trials:
                if t["reward"] < pass_threshold:
                    ri = t.get("reward_info", {})
                    reasons = []
                    for metric, val in ri.items():
                        if isinstance(val, (int, float)) and val < pass_threshold:
                            reasons.append(metric)
                    if reasons:
                        failure_reasons.append(", ".join(reasons))
                    else:
                        failure_reasons.append("unknown")

            per_task[split][tid] = {
                "trials": rewards,
                "pass_all": all(r >= pass_threshold for r in rewards),
                "pass_any": any(r >= pass_threshold for r in rewards),
                "failure_reasons": failure_reasons,
            }

    return per_task


def print_report(results: dict, per_task: dict):
    """Print formatted Pass^3 / Pass@3 report."""
    W = 70

    print(f"\n{'='*W}")
    print(f"{'CAR-bench Pass^3 Evaluation Report':^{W}}")
    print(f"{'='*W}")

    # Official aggregate scores
    print(f"\n--- Official Aggregate Scores ---")
    pk = results.get("pass_power_k_scores", {})
    ak = results.get("pass_at_k_scores", {})
    max_k = results.get("max_trials", 3)
    for k in range(1, max_k + 1):
        p = pk.get(f"Pass^{k}", 0)
        a = ak.get(f"Pass@{k}", 0)
        print(f"  Pass^{k}: {p*100:5.1f}%   Pass@{k}: {a*100:5.1f}%")

    # Per-category breakdown
    for split in ["base", "hallucination", "disambiguation"]:
        if split not in per_task:
            continue

        tasks = per_task[split]
        n = len(tasks)
        pass3 = sum(1 for t in tasks.values() if t["pass_all"])
        passa3 = sum(1 for t in tasks.values() if t["pass_any"])

        # Category scores from official evaluator
        pk_split = results.get("pass_power_k_scores_by_split", {}).get(split, {})
        ak_split = results.get("pass_at_k_scores_by_split", {}).get(split, {})

        print(f"\n{'─'*W}")
        print(f"  {split.upper()}: Pass^3 = {pass3}/{n}  |  Pass@3 = {passa3}/{n}")
        if pk_split:
            for k in range(1, max_k + 1):
                p = pk_split.get(f"Pass^{k}", 0)
                a = ak_split.get(f"Pass@{k}", 0)
                print(f"    Official Pass^{k}: {p*100:5.1f}%  |  Pass@{k}: {a*100:5.1f}%")
        print()

        # Per-task detail
        for tid in sorted(tasks.keys(), key=lambda x: int(x.split("_")[-1])):
            t = tasks[tid]
            trials_str = " ".join(
                f"{'P' if r >= 0.99 else 'F'}({r:.2f})" for r in t["trials"]
            )
            status = "PASS" if t["pass_all"] else "FAIL"
            marker = " " if t["pass_all"] else "X"
            print(f"  [{marker}] {tid:.<30s} {trials_str:.<30s} {status}")

    # Failure analysis
    print(f"\n{'─'*W}")
    print(f"  FAILURE ANALYSIS")
    print(f"{'─'*W}")

    reason_counts: dict[str, int] = {}
    failed_tasks = []

    for split, tasks in per_task.items():
        for tid, t in tasks.items():
            if not t["pass_all"]:
                failed_tasks.append(f"{tid}")
                for reason in t["failure_reasons"]:
                    for r in reason.split(", "):
                        r = r.strip()
                        if r:
                            reason_counts[r] = reason_counts.get(r, 0) + 1

    if reason_counts:
        print(f"\n  Failure reasons (across all failed trials):")
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            bar = "#" * min(count, 30)
            print(f"    {reason:.<40s} {count:>3d}  {bar}")
    else:
        print(f"\n  No failures detected!")

    if failed_tasks:
        print(f"\n  Failed tasks ({len(failed_tasks)}):")
        print(f"    {', '.join(failed_tasks)}")

    # Summary
    total = sum(len(t) for t in per_task.values())
    total_pass3 = sum(
        sum(1 for t in tasks.values() if t["pass_all"])
        for tasks in per_task.values()
    )
    total_passa3 = sum(
        sum(1 for t in tasks.values() if t["pass_any"])
        for tasks in per_task.values()
    )

    print(f"\n{'='*W}")
    print(f"  TOTAL:  Pass^3 = {total_pass3}/{total} ({total_pass3/total*100:.1f}%)"
          f"  |  Pass@3 = {total_passa3}/{total} ({total_passa3/total*100:.1f}%)")
    print(f"  Time:   {results.get('time_used', 0):.0f}s")
    print(f"{'='*W}\n")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pass^3 Evaluation Harness for CAR-bench Purple Agent"
    )
    parser.add_argument(
        "--purple-url",
        default="http://127.0.0.1:9010",
        help="Purple Agent URL (default: http://127.0.0.1:9010)",
    )
    parser.add_argument(
        "--green-port",
        type=int,
        default=9011,
        help="Port for the Green Agent (default: 9011)",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=3,
        help="Trials per task for Pass^k (default: 3)",
    )
    parser.add_argument(
        "--task-split",
        default="train",
        choices=["train", "test"],
        help="Task split to use (default: train)",
    )
    parser.add_argument(
        "--start-purple",
        action="store_true",
        help="Auto-start the Purple Agent",
    )
    parser.add_argument(
        "--purple-port",
        type=int,
        default=9010,
        help="Port for Purple Agent if --start-purple (default: 9010)",
    )
    parser.add_argument(
        "--output",
        default="output/pass3_results.json",
        help="Output JSON path (default: output/pass3_results.json)",
    )
    parser.add_argument(
        "--show-logs",
        action="store_true",
        help="Show agent stdout/stderr",
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip dependency installation checks",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Quick sanity check: 1 task per category, 1 trial (3 sessions total)",
    )
    parser.add_argument(
        "--mini-test",
        action="store_true",
        help="Mini test: 3 tasks per category, 3 trials (27 sessions total)",
    )
    parser.add_argument(
        "--callback-url",
        type=str,
        default=None,
        help="URL to POST results JSON when evaluation completes",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["base", "hallucination", "disambiguation"],
        help="Run only one task split (for parallel execution)",
    )
    args = parser.parse_args()

    # Smoke test overrides
    if args.smoke_test:
        args.num_trials = 1

    procs: list[tuple[subprocess.Popen, str]] = []

    try:
        # ── 1. Setup ───────────────────────────────────────────────────
        if not args.skip_setup:
            print("[1/5] Setting up car-bench-agentbeats...")
            ensure_green_agent_deps()
        else:
            print("[1/5] Skipping setup (--skip-setup)")

        # ── 2. Start Purple Agent (optional) ───────────────────────────
        if args.start_purple:
            print(f"\n[2/5] Starting Purple Agent on port {args.purple_port}...")
            purple_proc = start_process(
                ["uv", "run", "python", str(PURPLE_SERVER),
                 "--host", "0.0.0.0", "--port", str(args.purple_port)],
                cwd=str(PROJECT_ROOT),
                label="purple",
                show_logs=args.show_logs,
            )
            procs.append((purple_proc, "purple"))
            args.purple_url = f"http://127.0.0.1:{args.purple_port}"
        else:
            print(f"\n[2/5] Expecting Purple Agent at {args.purple_url}")

        # ── 3. Start Green Agent ───────────────────────────────────────
        print(f"\n[3/5] Starting Green Agent on port {args.green_port}...")
        green_proc = start_process(
            ["uv", "run", "python", str(GREEN_SERVER),
             "--host", "127.0.0.1", "--port", str(args.green_port)],
            cwd=str(CAR_BENCH_AGENTBEATS),
            label="green",
            show_logs=args.show_logs,
        )
        procs.append((green_proc, "green"))

        # ── Wait for agents to be ready ────────────────────────────────
        print(f"\n[3/5] Waiting for agents to be ready...")
        if not wait_for_server(args.purple_url, timeout=60, label="purple"):
            print("FATAL: Purple Agent not reachable.")
            sys.exit(1)
        if not wait_for_server(
            f"http://127.0.0.1:{args.green_port}", timeout=120, label="green"
        ):
            print("FATAL: Green Agent not reachable.")
            sys.exit(1)

        # ── 4. Generate scenario & run evaluation ──────────────────────
        if args.smoke_test:
            active_task_ids = SMOKE_TASK_IDS
        elif args.mini_test:
            active_task_ids = MINI_TASK_IDS
        else:
            active_task_ids = TASK_IDS

        # Filter to single split if requested
        if args.split:
            active_task_ids = {
                k: v if k == args.split else []
                for k, v in active_task_ids.items()
            }
        mode_label = (
            "SMOKE TEST (1/1/1 x 1 trial = 3 sessions)" if args.smoke_test
            else f"MINI TEST (3/3/3 x 3 trials = 27 sessions)" if args.mini_test
            else f"FULL ({'/'.join(str(len(v)) for v in active_task_ids.values())} x {args.num_trials} trials)"
        )
        print(f"\n[4/5] Generating scenario TOML... [{mode_label}]")
        scenario_content = generate_scenario_toml(
            purple_url=args.purple_url,
            green_port=args.green_port,
            num_trials=args.num_trials,
            task_split=args.task_split,
            task_ids=active_task_ids,
        )

        scenario_path = PROJECT_ROOT / "eval" / "scenario-pass3-generated.toml"
        scenario_path.write_text(scenario_content)
        print(f"  Scenario written to {scenario_path}")

        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path

        success = run_evaluation(
            scenario_path=scenario_path,
            output_path=output_path,
            show_logs=args.show_logs,
        )

        if not success:
            print("\n[eval] Evaluation failed. Check logs with --show-logs.")
            sys.exit(1)

        # ── 5. Parse results & print report ────────────────────────────
        print(f"\n[5/5] Parsing results...")
        results = parse_results(output_path)
        if not results:
            print("[report] No results to report.")
            sys.exit(1)

        detailed = results.get("detailed_results_by_split", {})
        per_task = compute_per_task_pass(detailed)
        print_report(results, per_task)

        # POST results to callback URL if provided
        if args.callback_url:
            try:
                import httpx
                full_results = json.loads(Path(output_path).read_text())
                resp = httpx.post(
                    args.callback_url,
                    json=full_results,
                    timeout=10,
                )
                print(f"[callback] POST {args.callback_url} → {resp.status_code}")
            except Exception as e:
                print(f"[callback] Failed: {e}")

        # Also save to /shared if available (Docker volume)
        shared_results = Path("/shared/results.json")
        if shared_results.parent.exists():
            try:
                import shutil
                shutil.copy2(output_path, str(shared_results))
                print(f"[shared] Results copied to {shared_results}")
            except Exception as e:
                print(f"[shared] Copy failed: {e}")

    except KeyboardInterrupt:
        print("\n[eval] Interrupted by user.")

    finally:
        # ── Cleanup ────────────────────────────────────────────────────
        print("\n[cleanup] Stopping agents...")
        for proc, label in reversed(procs):
            kill_process(proc, label)
        print("[cleanup] Done.")


if __name__ == "__main__":
    main()
