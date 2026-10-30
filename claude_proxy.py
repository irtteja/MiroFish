#!/usr/bin/env python3
"""
Claude Code LLM Proxy for MiroFish
-----------------------------------
Accepts OpenAI-compatible POST /v1/chat/completions requests and routes them
to the `claude` CLI, which is already authenticated in this Claude Code session.
No Anthropic API key required.

Usage:
  python claude_proxy.py             # listens on localhost:8765
  python claude_proxy.py --port 9000
"""

import argparse
import http.server
import json
import logging
import re
import subprocess
import sys
import time
import uuid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [claude-proxy] %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
CLAUDE_TIMEOUT = 180  # seconds per call


def messages_to_prompt(messages: list) -> str:
    """Flatten an OpenAI messages array into a single prompt for claude -p."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Handle structured content blocks (e.g. vision)
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )

        if role == "system":
            parts.append(f"<system>\n{content}\n</system>")
        elif role == "assistant":
            parts.append(f"<assistant>\n{content}\n</assistant>")
        else:
            parts.append(f"<user>\n{content}\n</user>")

    return "\n\n".join(parts)


JSON_SYSTEM_PROMPT = (
    "You are a JSON-only responder. "
    "Your entire response must be a single valid JSON object. "
    "Output only raw JSON — no markdown fences, no explanation, no text before or after."
)


def call_claude(prompt: str, json_mode: bool = False) -> str:
    """Invoke `claude -p` and return the text response."""
    cmd = [
        "claude", "-p", prompt,
        "--output-format", "text",
        # Prevent filesystem/bash tools — we want pure LLM responses
        "--disallowed-tools", "Bash,Edit,Write,Read,Glob,Grep,WebSearch,WebFetch",
    ]

    if json_mode:
        cmd += ["--append-system-prompt", JSON_SYSTEM_PROMPT]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=CLAUDE_TIMEOUT,
    )

    output = result.stdout.strip()
    if not output:
        output = result.stderr.strip()
    if not output:
        output = "(no response)"

    # Strip markdown code fences that Claude sometimes wraps responses in
    output = re.sub(r'^```(?:json)?\s*\n?', '', output, flags=re.IGNORECASE)
    output = re.sub(r'\n?```\s*$', '', output).strip()

    return output


class ProxyHandler(http.server.BaseHTTPRequestHandler):

    def log_message(self, format, *args):  # suppress default access log noise
        pass

    def _send_json(self, status: int, body: dict) -> None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/health", "/v1/health"):
            self._send_json(200, {"status": "ok", "backend": "claude-code-proxy"})
        elif self.path in ("/v1/models", "/models"):
            self._send_json(200, {
                "object": "list",
                "data": [{
                    "id": "claude-code",
                    "object": "model",
                    "created": 0,
                    "owned_by": "claude-code-proxy",
                }],
            })
        else:
            self._send_json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})

    def do_POST(self):
        normalized = self.path.split("?")[0].rstrip("/")
        if normalized not in ("/v1/chat/completions", "/chat/completions"):
            self._send_json(404, {
                "error": {"message": f"Unknown endpoint: {self.path}", "type": "invalid_request_error"}
            })
            return

        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)

        try:
            body = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._send_json(400, {
                "error": {"message": f"Invalid JSON body: {exc}", "type": "invalid_request_error"}
            })
            return

        messages = body.get("messages", [])
        response_format = body.get("response_format") or {}
        json_mode = (
            isinstance(response_format, dict)
            and response_format.get("type") == "json_object"
        )

        prompt = messages_to_prompt(messages)
        log.info("→ %d message(s), json_mode=%s", len(messages), json_mode)

        try:
            content = call_claude(prompt, json_mode=json_mode)
        except subprocess.TimeoutExpired:
            log.error("claude CLI timed out after %ds", CLAUDE_TIMEOUT)
            self._send_json(504, {
                "error": {"message": "claude CLI timed out", "type": "server_error"}
            })
            return
        except Exception as exc:
            log.error("claude CLI error: %s", exc)
            self._send_json(500, {
                "error": {"message": str(exc), "type": "server_error"}
            })
            return

        log.info("← %d chars", len(content))

        self._send_json(200, {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "claude-code"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        })


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Claude Code LLM proxy — serves OpenAI-compatible completions via claude CLI"
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    # Sanity-check that claude is available
    try:
        subprocess.run(["claude", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        log.error("'claude' CLI not found or not working. Are you inside a Claude Code session?")
        sys.exit(1)

    server = http.server.ThreadingHTTPServer((args.host, args.port), ProxyHandler)
    log.info("Claude proxy ready → http://%s:%d/v1/chat/completions", args.host, args.port)
    log.info("Set in .env:  LLM_BASE_URL=http://%s:%d/v1", args.host, args.port)
    log.info("              LLM_API_KEY=claude-code-proxy")
    log.info("              LLM_MODEL_NAME=claude-code")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Proxy stopped.")


if __name__ == "__main__":
    main()
