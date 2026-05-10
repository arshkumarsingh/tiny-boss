#!/usr/bin/env python3
"""
Tiny Boss → Hermes Agent proxy.
Starts an OpenAI-compatible HTTP server. Each /v1/chat/completions request
goes through the tiny-boss protocol: worker reads context, supervisor guides.

Supports streaming (stream=true) — protocol runs synchronously,
then the final supervisor answer streams token-by-token via SSE.

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
        print("Worker:     {}".format(self.worker), file=sys.stderr)
        print("Supervisor: {}".format(self.supervisor), file=sys.stderr)

    def _parse_messages(self, messages: list) -> tuple[str, str]:
        """Extract task from last user message, context from all prior messages (including assistant)."""
        context_parts = []
        task = "Process."

        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            role = msg.get("role", "user")
            if role == "user":
                task = content  # last user message wins
            context_parts.append("{}: {}".format(role, content))

        # Build context from all messages except the final user task
        if len(context_parts) > 1 and context_parts[-1].startswith("user:"):
            context = "\n".join(context_parts[:-1])
        elif len(context_parts) == 1:
            context = ""
        else:
            context = "\n".join(context_parts)
        return task, context or "No context."

    def chat(self, messages: list) -> str:
        task, context = self._parse_messages(messages)
        boss = TinyBoss(self.worker, self.supervisor, max_rounds=2)
        result = boss(task=task, context=context)
        return result.final_answer

    def chat_stream(self, messages: list):
        """Run protocol, then stream the answer word-by-word (no extra API call)."""
        task, context = self._parse_messages(messages)
        boss = TinyBoss(self.worker, self.supervisor, max_rounds=2)
        result = boss(task=task, context=context)

        # Yield the answer already produced by the protocol
        # Break into word-sized chunks for a streaming feel
        words = result.final_answer.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")


class Handler(BaseHTTPRequestHandler):
    proxy: BossProxy = None  # set by main()

    def _json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _sse(self, generator, model: str, rid: str):
        """Send Server-Sent Events for streaming."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        try:
            for token in generator:
                chunk = json.dumps({
                    "id": rid,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": token},
                        "finish_reason": None,
                    }],
                })
                self.wfile.write(f"data: {chunk}\n\n".encode())
                self.wfile.flush()

            # Final chunk
            final = json.dumps({
                "id": rid,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
            })
            self.wfile.write(f"data: {final}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except Exception as e:
            err = json.dumps({"error": {"message": str(e), "type": "stream_error"}})
            self.wfile.write(f"data: {err}\n\n".encode())
            self.wfile.flush()

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

        stream = req.get("stream", False)
        rid = f"boss-{uuid.uuid4().hex[:8]}"

        try:
            if stream:
                self._sse(self.proxy.chat_stream(messages), "tiny-boss", rid)
            else:
                answer = self.proxy.chat(messages)

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

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print("\n  WARNING: Binding to {}. Anyone on the network can spend your API credits.".format(args.host), file=sys.stderr)
        print("  Use a reverse proxy with authentication for production deployments.\n", file=sys.stderr)

    print("\nListening on http://{}:{}".format(args.host, args.port), file=sys.stderr)
    print("  GET  /health", file=sys.stderr)
    print("  POST /v1/chat/completions  (stream=true supported)", file=sys.stderr)
    print("\nAdd to ~/.hermes/config.yaml:", file=sys.stderr)
    print("  tiny-boss:", file=sys.stderr)
    print("    name: Tiny Boss", file=sys.stderr)
    print("    base_url: http://{}:{}/v1".format(args.host, args.port), file=sys.stderr)
    print("    api_key: no-key-needed", file=sys.stderr)
    print("    provider: openai", file=sys.stderr)
    print("    models:", file=sys.stderr)
    print("      - tiny-boss", file=sys.stderr)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.", file=sys.stderr)
        server.shutdown()


if __name__ == "__main__":
    main()
