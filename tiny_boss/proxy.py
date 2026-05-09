#!/usr/bin/env python3
"""
Tiny Boss → Hermes Agent proxy.
Starts an OpenAI-compatible HTTP server. Each /v1/chat/completions request
goes through the tiny-boss protocol: worker reads context, supervisor guides.

Usage:
  tiny-boss-proxy --worker groq/llama-3.1-8b-instant --supervisor deepseek/deepseek-v4-pro

Then add to ~/.hermes/config.yaml:
  custom_providers:
    tiny-boss:
      name: Tiny Boss (cost-saving)
      base_url: http://localhost:8765/v1
      api_key: no-key-needed
      provider: openai
      models:
        - tiny-boss
"""

import json
import sys
import time
import uuid
import threading
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler

from tiny_boss.clients import get_client
from tiny_boss.protocol import TinyBoss


class BossProxy:
    """Thread-safe singleton wrapping TinyBoss."""

    def __init__(self, worker_spec: str, supervisor_spec: str):
        wp, wm = worker_spec.split("/", 1)
        sp, sm = supervisor_spec.split("/", 1)
        self.worker = get_client(wp, wm)
        self.supervisor = get_client(sp, sm)
        self._lock = threading.Lock()
        print(f"Worker:     {self.worker}", file=sys.stderr)
        print(f"Supervisor: {self.supervisor}", file=sys.stderr)

    def chat(self, messages: list) -> str:
        system_parts, user_parts = [], []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            (system_parts if msg.get("role") == "system" else user_parts).append(content)

        task = user_parts[-1] if user_parts else "Process."
        context = "\n\n".join(system_parts + user_parts[:-1]) if len(user_parts) > 1 else (system_parts[0] if system_parts else "")

        with self._lock:
            boss = TinyBoss(self.worker, self.supervisor, max_rounds=2)
            result = boss(task=task, context=context or "No context.")
        return result.final_answer


class Handler(BaseHTTPRequestHandler):
    proxy: BossProxy = None  # set by main()

    def _json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            self._json({"object": "list", "data": [{
                "id": "tiny-boss", "object": "model",
                "created": int(time.time()), "owned_by": "tiny-boss"
            }]})
        elif self.path == "/health":
            self._json({"status": "ok"})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            return self._json({"error": "not found"}, 404)

        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            return self._json({"error": "invalid JSON"}, 400)

        messages = req.get("messages", [])
        if not messages:
            return self._json({"error": "no messages"}, 400)

        rid = f"boss-{uuid.uuid4().hex[:8]}"
        try:
            start = time.time()
            answer = self.proxy.chat(messages)
            elapsed = time.time() - start

            self._json({
                "id": rid, "object": "chat.completion",
                "created": int(time.time()), "model": "tiny-boss",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })
        except Exception as e:
            self._json({"error": {"message": str(e), "type": "boss_error"}}, 500)

    def log_message(self, *args):
        pass  # quiet


def main():
    parser = argparse.ArgumentParser(description="Tiny Boss → Hermes proxy")
    parser.add_argument("--worker", required=True, help="Worker spec: provider/model (e.g. groq/llama-3.1-8b-instant)")
    parser.add_argument("--supervisor", required=True, help="Supervisor spec: provider/model (e.g. deepseek/deepseek-v4-pro)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    Handler.proxy = BossProxy(args.worker, args.supervisor)
    server = HTTPServer((args.host, args.port), Handler)
    print(f"\nListening on http://{args.host}:{args.port}", file=sys.stderr)
    print(f"  GET  /health", file=sys.stderr)
    print(f"  POST /v1/chat/completions", file=sys.stderr)
    print(f"\nAdd to ~/.hermes/config.yaml:", file=sys.stderr)
    print(f"  tiny-boss:", file=sys.stderr)
    print(f"    name: Tiny Boss", file=sys.stderr)
    print(f"    base_url: http://{args.host}:{args.port}/v1", file=sys.stderr)
    print(f"    api_key: no-key-needed", file=sys.stderr)
    print(f"    provider: openai", file=sys.stderr)
    print(f"    models:", file=sys.stderr)
    print(f"      - tiny-boss", file=sys.stderr)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.", file=sys.stderr)
        server.shutdown()


if __name__ == "__main__":
    main()
