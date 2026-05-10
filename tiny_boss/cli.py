"""Tiny Boss — CLI with config file, progress, and quiet mode."""

import sys
import json
import os
import argparse
from pathlib import Path

from .clients import get_client
from .protocol import TinyBoss

CONFIG_DIR = Path.home() / ".config" / "tiny-boss"
CONFIG_FILE = CONFIG_DIR / "config.toml"

DEFAULT_CONFIG = """# Tiny Boss configuration
# worker = provider/model for the cheap reader
# supervisor = provider/model for the smart decision-maker

[defaults]
worker = "groq/llama-3.1-8b-instant"
supervisor = "deepseek/deepseek-v4-pro"
max_rounds = 3
"""


def _load_config() -> dict:
    """Load TOML config, returning defaults if missing."""
    if not CONFIG_FILE.exists():
        return {"defaults": {}}
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            return {"defaults": {}}
    with open(CONFIG_FILE) as f:
        return tomllib.loads(f.read())


def _ensure_config():
    """Create config file if it doesn't exist."""
    if not CONFIG_FILE.exists():
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(DEFAULT_CONFIG)


def _resolve_spec(arg_val: str | None, config_val: str | None, name: str) -> str:
    """Resolve worker/supervisor from arg, config, or error."""
    if arg_val:
        return arg_val
    if config_val:
        return config_val
    print(f"\n  No {name} specified.", file=sys.stderr)
    print(f"  Run 'tiny-boss init' to create a config file, or use:", file=sys.stderr)
    print(f"    tiny-boss --{name} provider/model ...", file=sys.stderr)
    print(f"\n  Example:", file=sys.stderr)
    print(f"    tiny-boss --{name} groq/llama-3.1-8b-instant ...\n", file=sys.stderr)
    sys.exit(1)


def _progress_callback(role: str, message: str):
    """Show protocol progress — one clean line per event."""
    if role == "WORKER":
        # message starts with "Q: ..." or actual response
        if message.startswith("Q:"):
            print(f"  {message[:100]}", file=sys.stderr)
        else:
            print(f"    → {message[:120].strip()}", file=sys.stderr)
    elif role == "FINAL":
        print(f"  ✓ Done.", file=sys.stderr)
    # SUPERVISOR messages are verbose JSON — skip


def main():
    parser = argparse.ArgumentParser(
        description="tiny-boss — cheap worker + smart supervisor LLM protocol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With config file defaults:
  tiny-boss --task "Summarize this paper" --context doc.txt

  # Explicit models:
  tiny-boss --worker groq/llama-3.1-8b-instant --supervisor deepseek/deepseek-v4-pro \\
            --task "Extract entities" --context data.json

  # Pipe context, quiet output:
  cat paper.txt | tiny-boss --task "What methodology?" --quiet

  # Setup:
  tiny-boss init         Create config file with defaults
  tiny-boss config       Show current configuration

Providers: groq, deepseek, openai, gemini, anthropic, openrouter
Keys auto-loaded from ~/.hermes/.env
        """,
    )
    parser.add_argument("--task", "-t", help="Task/question")
    parser.add_argument("--context", "-c", nargs="*", help="Context files (or stdin)")
    parser.add_argument("--worker", "-w", help="provider/model (uses config default if omitted)")
    parser.add_argument("--supervisor", "-s", help="provider/model (uses config default if omitted)")
    parser.add_argument("--max-rounds", "-r", type=int, help="Max rounds (default: 3)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Full protocol trace")
    parser.add_argument("--progress", "-p", action="store_true", help="Real-time round progress")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only print final answer (no stats)")
    parser.add_argument("--json", action="store_true", help="Output full result as JSON")

    # Setup commands
    parser.add_argument("--init", action="store_true", help="Create config file with defaults")
    parser.add_argument("--config", action="store_true", help="Show current configuration and exit")

    args = parser.parse_args()

    # --init
    if args.init:
        _ensure_config()
        print(f"Config created at {CONFIG_FILE}")
        print(f"\nEdit it to set your defaults:\n")
        with open(CONFIG_FILE) as f:
            print(f.read())
        return

    # --config
    if args.config:
        _ensure_config()
        cfg = _load_config()
        print(f"Config file: {CONFIG_FILE}")
        print(f"Exists: {CONFIG_FILE.exists()}\n")
        defaults = cfg.get("defaults", {})
        print(f"  worker:     {defaults.get('worker', '(not set)')}")
        print(f"  supervisor: {defaults.get('supervisor', '(not set)')}")
        print(f"  max_rounds: {defaults.get('max_rounds', 3)}")
        return

    # Load config for defaults
    cfg = _load_config()
    defaults = cfg.get("defaults", {})

    worker_spec = _resolve_spec(args.worker, defaults.get("worker"), "worker")
    supervisor_spec = _resolve_spec(args.supervisor, defaults.get("supervisor"), "supervisor")
    max_rounds = args.max_rounds or defaults.get("max_rounds", 3)

    wp, wm = worker_spec.split("/", 1)
    sp, sm = supervisor_spec.split("/", 1)

    try:
        worker = get_client(wp, wm)
    except ValueError as e:
        print(f"\n  Worker error: {e}", file=sys.stderr)
        print(f"  Provider '{wp}' not recognized. Available: groq, deepseek, openai, gemini, anthropic, openrouter\n", file=sys.stderr)
        sys.exit(1)

    try:
        supervisor = get_client(sp, sm)
    except ValueError as e:
        print(f"\n  Supervisor error: {e}", file=sys.stderr)
        print(f"  Provider '{sp}' not recognized. Available: groq, deepseek, openai, gemini, anthropic, openrouter\n", file=sys.stderr)
        sys.exit(1)

    # Context
    if args.context:
        ctx = "\n\n".join(Path(p).read_text() for p in args.context if Path(p).is_file())
    else:
        ctx = sys.stdin.read()

    if not args.quiet:
        print(f"Worker: {worker}  |  Supervisor: {supervisor}", file=sys.stderr)
        print(f"Context: {len(ctx)} chars  |  Max rounds: {max_rounds}", file=sys.stderr)
        print(file=sys.stderr)

    callback = _progress_callback if args.progress else None
    boss = TinyBoss(worker, supervisor, max_rounds=max_rounds,
                    verbose=args.verbose, callback=callback)

    result = boss(task=args.task, context=ctx)

    if args.json:
        out = {
            "final_answer": result.final_answer,
            "rounds_used": result.rounds_used,
            "timing": result.timing,
            "worker_tokens": result.worker_tokens,
            "supervisor_tokens": result.supervisor_tokens,
            "errors": result.errors,
        }
        print(json.dumps(out, indent=2))
    elif args.quiet:
        print(result.final_answer)
    else:
        print(f"\n{'='*70}")
        print(result.final_answer)
        print(f"{'='*70}")
        stats = (f"\n[{result.rounds_used} rounds, {result.timing['total']:.1f}s, "
                 f"w:{result.worker_tokens} tok, s:{result.supervisor_tokens} tok]")
        if result.errors:
            stats += f"  errors: {result.errors}"
        print(stats)


if __name__ == "__main__":
    main()
