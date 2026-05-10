"""Integration tests for the TinyBoss protocol using mocked clients."""

from tiny_boss.protocol import TinyBoss
from tiny_boss.clients import LLMClient


class MockClient(LLMClient):
    """Returns pre-programmed responses. Raises on exhaustion."""

    def __init__(self, responses: list, provider: str = "mock", model: str = "mock"):
        super().__init__(provider, model)
        self.responses = responses
        self.calls = []

    def __call__(self, prompt: str, system: str = "") -> tuple[str, dict]:
        self.calls.append(prompt)
        if not self.responses:
            raise RuntimeError("MockClient exhausted")
        resp = self.responses.pop(0)
        usage = {"prompt_tokens": 10, "completion_tokens": 20}
        return resp, usage


class TestTinyBoss:
    def test_supervisor_answers_directly(self):
        """Supervisor can answer without involving the worker."""
        supervisor = MockClient([
            '{"analysis": "trivial", "question": "", "final_answer": "42", "decision": "provide_final_answer"}',
        ])
        worker = MockClient([])

        boss = TinyBoss(worker, supervisor, max_rounds=3)
        result = boss(task="What is 6*7?", context="Math facts.")

        assert result.final_answer == "42"
        assert result.worker_rounds_used == 0
        assert len(worker.calls) == 0  # Worker never invoked

    def test_single_worker_round(self):
        """Supervisor asks one question, worker answers, supervisor finishes."""
        supervisor = MockClient([
            # Round 0: ask worker
            '{"analysis": "need context", "question": "What is the capital?", "final_answer": "", "decision": "ask_worker"}',
            # Round 1: synthesize and finish
            '{"analysis": "got it", "question": "", "final_answer": "Paris", "decision": "provide_final_answer"}',
        ])
        worker = MockClient([
            "The capital of France is Paris.",
        ])

        boss = TinyBoss(worker, supervisor, max_rounds=3)
        result = boss(task="What is the capital?", context="France info.")

        assert result.final_answer == "Paris"
        assert result.worker_rounds_used == 1
        assert len(worker.calls) == 1

    def test_multi_round_with_history(self):
        """Supervisor asks follow-up questions across multiple rounds."""
        supervisor = MockClient([
            # Round 0
            '{"analysis": "ask about methodology", "question": "What method was used?", "final_answer": "", "decision": "ask_worker"}',
            # Round 1: ask follow-up
            '{"analysis": "need results too", "question": "What were the results?", "final_answer": "", "decision": "ask_worker"}',
            # Round 2: synthesize
            '{"analysis": "complete", "question": "", "final_answer": "Method: survey. Results: 95% accuracy.", "decision": "provide_final_answer"}',
        ])
        worker = MockClient([
            "They used a survey.",
            "95% accuracy on the test set.",
        ])

        boss = TinyBoss(worker, supervisor, max_rounds=3)
        result = boss(task="Summarize the paper.", context="Paper text.")

        assert "survey" in result.final_answer
        assert "95%" in result.final_answer
        assert result.worker_rounds_used == 2
        assert len(worker.calls) == 2

    def test_worker_retry_on_failure(self):
        """Worker fails first attempt, succeeds on retry."""
        supervisor = MockClient([
            '{"analysis": "ask", "question": "What color?", "final_answer": "", "decision": "ask_worker"}',
            '{"analysis": "done", "question": "", "final_answer": "Blue", "decision": "provide_final_answer"}',
        ])

        call_count = [0]
        def flaky_worker(prompt, system=""):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Connection lost")
            return "The color is Blue.", {"prompt_tokens": 5, "completion_tokens": 10}

        # Use a real LLMClient subclass with custom __call__
        class FlakyClient(LLMClient):
            def __call__(self, prompt, system=""):
                return flaky_worker(prompt, system)

        worker = FlakyClient("mock", "mock")
        boss = TinyBoss(worker, supervisor, max_rounds=3)
        result = boss(task="What color?", context="Color info.")

        assert result.final_answer == "Blue"
        assert result.worker_rounds_used == 1
        assert call_count[0] == 2  # Failed once, succeeded on retry

    def test_worker_both_attempts_fail(self):
        """Worker fails both attempts, supervisor told to answer from knowledge."""
        supervisor = MockClient([
            '{"analysis": "ask", "question": "What color?", "final_answer": "", "decision": "ask_worker"}',
            '{"analysis": "worker down, answering myself", "question": "", "final_answer": "Red (from my knowledge)", "decision": "provide_final_answer"}',
        ])

        class AlwaysFailClient(LLMClient):
            def __call__(self, prompt, system=""):
                raise RuntimeError("API down")

        worker = AlwaysFailClient("mock", "mock")
        boss = TinyBoss(worker, supervisor, max_rounds=3)
        result = boss(task="What color?", context="Color info.")

        assert "answer from your own knowledge" in result.final_answer.lower() or "Red" in result.final_answer
        assert result.worker_rounds_used == 1

    def test_supervisor_failure_mid_protocol(self):
        """Supervisor fails during synthesis, returns worker response."""
        supervisor = MockClient([
            '{"analysis": "ask", "question": "What color?", "final_answer": "", "decision": "ask_worker"}',
            # Second call will raise (empty responses list)
        ])
        worker = MockClient(["The color is Green."])

        boss = TinyBoss(worker, supervisor, max_rounds=3)
        result = boss(task="What color?", context="Color info.")

        assert "Green" in result.final_answer
        assert len(result.errors) == 1
        assert "supervisor" in result.errors[0]
