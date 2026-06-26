"""
OpenRouter LLM integration for response generation.

OpenRouter provides access to 200+ models (GPT-4o, Claude, Gemini, Llama, Mistral, etc.)
through a single OpenAI-compatible API endpoint.

Get your API key at: https://openrouter.ai/keys
"""

import json
import logging
import requests
from utils.config import Config

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterLLM:
    """Manage OpenRouter LLM for response generation (OpenAI-compatible API)."""

    def __init__(
        self,
        api_key: str = None,
        model_name: str = None,
        temperature: float = None,
        max_tokens: int = None,
    ):
        self.api_key     = api_key    or Config.OPENROUTER_API_KEY
        self.model_name  = model_name or Config.OPENROUTER_MODEL_NAME
        self.temperature = temperature if temperature is not None else Config.LLM_TEMPERATURE
        self.max_tokens  = max_tokens or Config.LLM_MAX_TOKENS

        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not provided. "
                "Set it in .env or pass as argument. "
                "Get a free key at https://openrouter.ai/keys"
            )

        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "RAG Chatbot",
            "Content-Type": "application/json",
        }

        logger.info(
            f"OpenRouterLLM initialized with model: {self.model_name}, "
            f"temperature: {self.temperature}, max_tokens: {self.max_tokens}"
        )

    def _build_payload(self, prompt: str, stream: bool = False) -> dict:
        return {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }

    def generate_response(self, prompt: str) -> str:
        if not prompt or not prompt.strip():
            logger.warning("Empty prompt provided to OpenRouterLLM")
            return ""

        logger.info(f"Generating response via OpenRouter ({self.model_name})...")

        response = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=self._headers,
            json=self._build_payload(prompt),
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        logger.info("Response generated successfully")
        return text

    def generate_response_streaming(self, prompt: str):
        if not prompt or not prompt.strip():
            logger.warning("Empty prompt provided to OpenRouterLLM streaming")
            return

        logger.info(f"Generating streaming response via OpenRouter ({self.model_name})...")

        with requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=self._headers,
            json=self._build_payload(prompt, stream=True),
            stream=True,
            timeout=60,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    line = line[6:]
                if line == "[DONE]":
                    break
                try:
                    chunk = json.loads(line)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
                except (json.JSONDecodeError, KeyError):
                    continue

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    def get_model_info(self) -> dict:
        return {
            "provider": "openrouter",
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_key_configured": bool(self.api_key),
        }