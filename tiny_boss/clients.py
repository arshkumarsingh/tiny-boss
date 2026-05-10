"""
LLM clients for tiny-boss.
One interface: callable returns (text, usage_dict).
No presets, no defaults — always explicit provider + model.
Auto-loads API keys from ~/.hermes/.env.
"""

import os
from pathlib import Path
from typing import Optional, Generator

# Auto-load Hermes .env so Hermes users don't need to export keys
# Only loads *_API_KEY variables — ignores everything else
_ENV_FILE = Path.home() / ".hermes" / ".env"
if _ENV_FILE.exists():
    with open(_ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and val and key not in os.environ and key.endswith("_API_KEY"):
                    os.environ[key] = val


def _retry(fn, max_attempts: int = 3, base_delay: float = 1.0):
    """Call fn() with exponential backoff on transient errors."""
    import time as _time
    last = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            last = e
            if attempt < max_attempts - 1:
                delay = base_delay * (2 ** attempt)
                _time.sleep(delay)
    raise last


class LLMClient:
    """Callable that returns (text, usage_dict)."""

    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model

    def __call__(self, prompt: str, system: str = "") -> tuple[str, dict]:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.provider}/{self.model}"


# ── OpenAI-compatible (Groq, DeepSeek, OpenRouter, Together, etc.) ──

class OpenAIClient(LLMClient):
    """Any OpenAI-compatible endpoint."""

    def __init__(self, provider: str, model: str,
                 api_key: Optional[str] = None, base_url: Optional[str] = None,
                 temperature: float = 0.1, max_tokens: int = 4096):
        super().__init__(provider, model)
        self.api_key = api_key or os.environ.get(f"{provider.upper()}_API_KEY", "")
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        """Lazy-init and cache the OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def __call__(self, prompt: str, system: str = "") -> tuple[str, dict]:
        client = self._get_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = _retry(lambda: client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ))

        text = resp.choices[0].message.content or ""
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
            "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
        }
        return text, usage

    def stream(self, prompt: str, system: str = "") -> "Generator[str, None, None]":
        """Yield tokens one at a time via streaming."""
        client = self._get_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        stream = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content


# ── Google Gemini ──

class GeminiClient(LLMClient):
    """Google Gemini via google-generativeai SDK."""

    def __init__(self, model: str, api_key: Optional[str] = None,
                 temperature: float = 0.1, max_tokens: int = 4096):
        super().__init__("gemini", model)
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not set. Get one: https://aistudio.google.com/apikey"
            )

    def __call__(self, prompt: str, system: str = "") -> tuple[str, dict]:
        import google.generativeai as genai

        if self._client is None:
            genai.configure(api_key=self.api_key)
            self._client = genai

        model = self._client.GenerativeModel(
            model_name=self.model,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
            },
            system_instruction=system if system else None,
        )
        resp = _retry(lambda: model.generate_content(prompt))
        text = resp.text or ""
        usage = {
            "prompt_tokens": resp.usage_metadata.prompt_token_count if resp.usage_metadata else 0,
            "completion_tokens": resp.usage_metadata.candidates_token_count if resp.usage_metadata else 0,
        }
        return text, usage


# ── Convenience subclasses with pre-configured base URLs ──

class GroqClient(OpenAIClient):
    def __init__(self, model: str, api_key: Optional[str] = None):
        super().__init__("groq", model, api_key=api_key,
                         base_url="https://api.groq.com/openai/v1")


class DeepSeekClient(OpenAIClient):
    def __init__(self, model: str, api_key: Optional[str] = None):
        super().__init__("deepseek", model, api_key=api_key,
                         base_url="https://api.deepseek.com")


class OpenRouterClient(OpenAIClient):
    def __init__(self, model: str, api_key: Optional[str] = None):
        super().__init__("openrouter", model, api_key=api_key,
                         base_url="https://openrouter.ai/api/v1")


# ── Anthropic (native SDK) ──

class AnthropicClient(LLMClient):
    """Anthropic Claude via native SDK. Set ANTHROPIC_API_KEY."""

    def __init__(self, model: str, api_key: Optional[str] = None,
                 temperature: float = 0.1, max_tokens: int = 4096):
        super().__init__("anthropic", model)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Get one: https://console.anthropic.com/"
            )

    def __call__(self, prompt: str, system: str = "") -> tuple[str, dict]:
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)

        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        resp = _retry(lambda: self._client.messages.create(**kwargs))
        text = "".join(
            block.text for block in resp.content
            if hasattr(block, "text")
        )
        usage = {
            "prompt_tokens": resp.usage.input_tokens if resp.usage else 0,
            "completion_tokens": resp.usage.output_tokens if resp.usage else 0,
        }
        return text, usage


# ── Factory ──

_PROVIDER_MAP = {
    "openai": OpenAIClient,
    "groq": GroqClient,
    "deepseek": DeepSeekClient,
    "gemini": GeminiClient,
    "openrouter": OpenRouterClient,
    "anthropic": AnthropicClient,
}


def get_client(provider: str, model: str, **kwargs) -> LLMClient:
    """
    Create a client by provider name.

        worker = get_client("groq", "llama-3.1-8b-instant")
        supervisor = get_client("deepseek", "deepseek-v4-pro")
        supervisor = get_client("openai", "gpt-4o", api_key="sk-...")
    """
    p = provider.lower()
    if p not in _PROVIDER_MAP:
        raise ValueError(
            f"Unknown provider '{provider}'. Available: {list(_PROVIDER_MAP)}"
        )
    return _PROVIDER_MAP[p](model=model, **kwargs)
