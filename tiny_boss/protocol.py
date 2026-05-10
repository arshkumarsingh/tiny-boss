"""
Tiny Boss Protocol
==================
Cheap LLM (worker) reads full context.
Smart LLM (supervisor) asks targeted questions, synthesizes final answer.

Multi-round: supervisor asks → worker answers → supervisor synthesizes → repeat.
Expensive model never processes the full context.
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

from .clients import LLMClient


# ── Prompts ──────────────────────────────────────────────────────

SUPERVISOR_INITIAL = """You are a supervisor. A worker LLM will process context and answer your questions.

<task>
{task}
</task>

You may ask the worker up to {max_rounds} questions across the entire session.
Each round: you ask ONE clear question → worker answers → you synthesize.

Output JSON:
{{
    "analysis": "your thinking",
    "question": "question for the worker — empty if answering directly",
    "final_answer": "complete answer if answering now, otherwise empty",
    "decision": "ask_worker or provide_final_answer"
}}

If you can answer directly without the worker, use provide_final_answer.
Otherwise use ask_worker."""

SUPERVISOR_SYNTHESIS = """You are a supervisor synthesizing worker responses.

<task>
{task}
</task>

<worker_response>
{worker_response}
</worker_response>

<rounds_remaining>
{rounds_remaining}
</rounds_remaining>

Output JSON:
{{
    "analysis": "what you learned, what's still missing",
    "question": "follow-up question (empty if done)",
    "final_answer": "complete answer if done, otherwise empty",
    "decision": "ask_worker or provide_final_answer"
}}"""

SUPERVISOR_FINAL = """FINAL round. You MUST answer now.

<task>
{task}
</task>

<worker_response>
{worker_response}
</worker_response>

Output JSON:
{{
    "analysis": "synthesis of all responses",
    "final_answer": "complete, well-structured answer",
    "decision": "provide_final_answer"
}}"""

WORKER_PROMPT = """Answer the question using ONLY the provided context.

<context>
{context}
</context>

<question>
{question}
</question>

Be thorough. Quote relevant parts. If the context lacks the answer, say so clearly."""


# ── Helpers ───────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown fences."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for pattern in [r"```(?:json)?\s*([\s\S]*?)```", r"\{[\s\S]*\}"]:
        m = re.search(pattern, text)
        if m:
            try:
                return json.loads(m.group(1) if "```" in pattern else m.group(0))
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Cannot extract JSON from: {text[:300]}...")


# ── Result ────────────────────────────────────────────────────────

@dataclass
class BossResult:
    final_answer: str
    supervisor_messages: list = field(default_factory=list)
    worker_messages: list = field(default_factory=list)
    rounds_used: int = 0
    timing: dict = field(default_factory=dict)
    worker_tokens: int = 0
    supervisor_tokens: int = 0
    errors: list = field(default_factory=list)


# ── Core ─────────────────────────────────────────────────────────

class TinyBoss:
    """
    Multi-round protocol: cheap worker processes context, smart supervisor guides.

        worker = get_client("groq", "llama-3.1-8b-instant")
        supervisor = get_client("deepseek", "deepseek-v4-pro")
        boss = TinyBoss(worker, supervisor, max_rounds=3)
        result = boss(task="Summarize", context=open("doc.txt").read())
        print(result.final_answer)
    """

    def __init__(
        self,
        worker: LLMClient,
        supervisor: LLMClient,
        max_rounds: int = 3,
        verbose: bool = False,
        callback: Optional[Callable] = None,
    ):
        self.worker = worker
        self.supervisor = supervisor
        self.max_rounds = max_rounds
        self.verbose = verbose
        self.callback = callback

    def _log(self, role: str, msg: str):
        if self.verbose:
            print(f"\n{'='*60}\n[{role}]\n{'='*60}\n{msg[:1500]}")
        if self.callback:
            self.callback(role, msg)

    def __call__(
        self,
        task: str,
        context: str | list[str],
        max_rounds: Optional[int] = None,
    ) -> BossResult:
        rounds = max_rounds or self.max_rounds
        ctx = "\n\n".join(context) if isinstance(context, list) else context

        s_msgs, w_msgs = [], []
        w_tok = s_tok = 0
        w_errors = 0
        t0 = time.time()
        timings = {}

        # Round 0: supervisor's opening move
        self._log("SUPERVISOR", "Analyzing task...")
        prompt = SUPERVISOR_INITIAL.format(task=task, max_rounds=rounds)
        t = time.time()
        try:
            resp, usage = self.supervisor(prompt)
        except Exception as e:
            return BossResult(
                final_answer=f"Supervisor unavailable: {e}",
                supervisor_messages=s_msgs, worker_messages=w_msgs,
                timing={"total": time.time() - t0, **timings},
                worker_tokens=w_tok, supervisor_tokens=s_tok,
                errors=[f"supervisor_init: {e}"],
            )
        timings["s_init"] = time.time() - t
        s_tok += usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        try:
            parsed = _extract_json(resp)
        except ValueError:
            parsed = {"decision": "provide_final_answer", "final_answer": resp}
        s_msgs.append(parsed)
        self._log("SUPERVISOR", json.dumps(parsed, indent=2))

        if parsed.get("decision") == "provide_final_answer":
            return BossResult(
                final_answer=parsed.get("final_answer") or parsed.get("question") or "No answer.",
                supervisor_messages=s_msgs, worker_messages=w_msgs,
                timing={"total": time.time() - t0, **timings},
                worker_tokens=w_tok, supervisor_tokens=s_tok,
            )

        # Rounds 1..N
        for r in range(1, rounds + 1):
            question = parsed.get("question", "Analyze the context.")

            self._log(f"WORKER r{r}", f"Q: {question[:200]}...")
            wp = WORKER_PROMPT.format(context=ctx, question=question)
            t = time.time()
            try:
                w_resp, wu = self.worker(wp)
            except Exception as e:
                w_errors += 1
                w_resp = f"[Worker unavailable: {e}]"
                wu = {"prompt_tokens": 0, "completion_tokens": 0}
                if w_errors >= 2:
                    # Worker failed twice — supervisor answers directly
                    w_resp = "[Worker unavailable after retries. Answer from your own knowledge.]"
            timings[f"w_r{r}"] = time.time() - t
            w_tok += wu.get("prompt_tokens", 0) + wu.get("completion_tokens", 0)
            w_msgs.append({"question": question, "response": w_resp})
            self._log(f"WORKER r{r}", w_resp)

            # Supervisor synthesizes
            remaining = rounds - r
            self._log("SUPERVISOR", f"Synthesizing ({remaining} rounds left)...")

            if remaining == 0:
                sp = SUPERVISOR_FINAL.format(task=task, worker_response=w_resp)
            else:
                sp = SUPERVISOR_SYNTHESIS.format(
                    task=task, worker_response=w_resp, rounds_remaining=remaining
                )

            t = time.time()
            try:
                s_resp, su = self.supervisor(sp)
            except Exception as e:
                # Supervisor failed mid-protocol — return what we have
                return BossResult(
                    final_answer=w_resp,
                    supervisor_messages=s_msgs, worker_messages=w_msgs,
                    rounds_used=r,
                    timing={"total": time.time() - t0, **timings},
                    worker_tokens=w_tok, supervisor_tokens=s_tok,
                    errors=[f"supervisor_r{r}: {e}"],
                )
            timings[f"s_r{r}"] = time.time() - t
            s_tok += su.get("prompt_tokens", 0) + su.get("completion_tokens", 0)
            try:
                parsed = _extract_json(s_resp)
            except ValueError:
                parsed = {"decision": "provide_final_answer", "final_answer": s_resp}
            s_msgs.append(parsed)
            self._log("SUPERVISOR", json.dumps(parsed, indent=2))

            if parsed.get("decision") == "provide_final_answer":
                answer = parsed.get("final_answer") or parsed.get("question") or w_resp
                self._log("FINAL", answer)
                return BossResult(
                    final_answer=answer,
                    supervisor_messages=s_msgs, worker_messages=w_msgs,
                    rounds_used=r,
                    timing={"total": time.time() - t0, **timings},
                    worker_tokens=w_tok, supervisor_tokens=s_tok,
                )

        # Exhausted rounds
        final = parsed.get("final_answer") or parsed.get("analysis", "No answer.")
        return BossResult(
            final_answer=final,
            supervisor_messages=s_msgs, worker_messages=w_msgs,
            rounds_used=rounds,
            timing={"total": time.time() - t0, **timings},
            worker_tokens=w_tok, supervisor_tokens=s_tok,
            errors=[f"max_rounds ({rounds}) exhausted"] if rounds > 1 else [],
        )
