"""Tiny Boss — CLI."""

import sys
import json
import argparse
from pathlib import Path

from .clients import get_client
from .protocol import TinyBoss


def main():
    parser = argparse.ArgumentParser(
        description="Tiny Boss — cheap worker + smart supervisor LLM protocol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  tiny-boss --worker groq/llama-3.1-8b-instant \\
             --supervisor deepseek/deepseek-v4-pro \\
             --task "Summarize" --context doc.txt

  cat paper.txt | tiny-boss --worker groq/llama-3.1-8b-instant \\
             --supervisor deepseek/deepseek-v4-pro \\
             --task "What methodology?"

Providers: groq, deepseek, openai, gemini, openrouter
Env vars: GROQ_API_KEY, DEEPSEEK_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY
(Keys auto-loaded from ~/.hermes/.env for Hermes users)
        """,
    )
    parser.add_argument("--worker", "-w", required=True, help="provider/model (e.g. groq/llama-3.1-8b-instant)")
    parser.add_argument("--supervisor", "-s", required=True, help="provider/model (e.g. deepseek/deepseek-v4-pro)")
    parser.add_argument("--task", "-t", required=True, help="Task/question")
    parser.add_argument("--context", "-c", nargs="*", help="Context files (or stdin)")
    parser.add_argument("--max-rounds", "-r", type=int, default=3)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--json", action="store_true", help="Output full result as JSON")

    args = parser.parse_args()

    wp, wm = args.worker.split("/", 1)
    sp, sm = args.supervisor.split("/", 1)

    worker = get_client(wp, wm)
    supervisor = get_client(sp, sm)

    # Context
    if args.context:
        ctx = "\n\n".join(Path(p).read_text() for p in args.context if Path(p).is_file())
    else:
        ctx = sys.stdin.read()

    print(f"Worker: {worker}  |  Supervisor: {supervisor}", file=sys.stderr)
    print(f"Context: {len(ctx)} chars", file=sys.stderr)

    boss = TinyBoss(worker, supervisor, max_rounds=args.max_rounds, verbose=args.verbose)
    result = boss(task=args.task, context=ctx)

    if args.json:
        out = {
            "final_answer": result.final_answer,
            "rounds_used": result.rounds_used,
            "timing": result.timing,
            "worker_tokens": result.worker_tokens,
            "supervisor_tokens": result.supervisor_tokens,
        }
        print(json.dumps(out, indent=2))
    else:
        print(f"\n{'='*70}")
        print(result.final_answer)
        print(f"{'='*70}")
        print(f"\n[{result.rounds_used} rounds, {result.timing['total']:.1f}s, "
              f"w:{result.worker_tokens} tok, s:{result.supervisor_tokens} tok]")
