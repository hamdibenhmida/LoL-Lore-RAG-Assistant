"""
Google Gemini LLM integration for response generation.

Provides the same interface as GroqLLM so it can be swapped in
transparently by the LLM factory without any changes to RAGChain.
"""

import logging
from utils.config import Config

logger = logging.getLogger(__name__)


class GeminiLLM:
    """Manage Google Gemini LLM for response generation."""

    def __init__(
        self,
        api_key: str = None,
        model_name: str = None,
        temperature: float = None,
        max_tokens: int = None
    ):
        """
        Initialize Gemini LLM.

        Args:
            api_key: Google AI API key. Defaults to Config.GEMINI_API_KEY
            model_name: Model name to use. Defaults to Config.GEMINI_MODEL_NAME
            temperature: Model temperature (0-1). Defaults to Config.LLM_TEMPERATURE
            max_tokens: Maximum tokens for response. Defaults to Config.LLM_MAX_TOKENS
        """
        self.api_key = api_key or Config.GEMINI_API_KEY
        self.model_name = model_name or Config.GEMINI_MODEL_NAME
        self.temperature = temperature if temperature is not None else Config.LLM_TEMPERATURE
        self.max_tokens = max_tokens or Config.LLM_MAX_TOKENS

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not provided. Set it in .env or pass as argument. "
                "Get a free key at https://aistudio.google.com/app/apikey"
            )

        # Lazy-import to avoid hard dependency if only Groq is used
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                "langchain-google-genai is not installed. "
                "Run: pip install langchain-google-genai google-generativeai"
            )

        self.model = ChatGoogleGenerativeAI(
            google_api_key=self.api_key,
            model=self.model_name,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )

        logger.info(
            f"GeminiLLM initialized with model: {self.model_name}, "
            f"temperature: {self.temperature}, max_tokens: {self.max_tokens}"
        )

    def generate_response(self, prompt: str) -> str:
        """
        Generate response from the LLM.

        Args:
            prompt: Input prompt for the model.

        Returns:
            Generated response text.
        """
        if not prompt or not prompt.strip():
            logger.warning("Empty prompt provided to GeminiLLM")
            return ""

        logger.info("Generating response from Gemini LLM...")

        try:
            response = self.model.invoke(prompt)
            response_text = response.content
            logger.info("Response generated successfully")
            return response_text

        except Exception as e:
            logger.error(f"Error generating response from Gemini LLM: {str(e)}")
            raise

    def generate_response_streaming(self, prompt: str):
        """
        Generate response with streaming (yields chunks).

        Args:
            prompt: Input prompt for the model.

        Yields:
            Response text chunks.
        """
        if not prompt or not prompt.strip():
            logger.warning("Empty prompt provided to GeminiLLM streaming")
            return

        logger.info("Generating streaming response from Gemini LLM...")

        try:
            for chunk in self.model.stream(prompt):
                yield chunk.content

        except Exception as e:
            logger.error(f"Error in Gemini streaming response: {str(e)}")
            raise

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text (approximation).

        Args:
            text: Text to count tokens for.

        Returns:
            Approximate token count.
        """
        return len(text) // 4

    def get_model_info(self) -> dict:
        """
        Get information about the configured model.

        Returns:
            Dictionary with model information.
        """
        return {
            "provider": "gemini",
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_key_configured": bool(self.api_key)
        }


def get_gemini_llm(
    api_key: str = None,
    model_name: str = None,
    temperature: float = None,
    max_tokens: int = None
) -> GeminiLLM:
    """
    Create a GeminiLLM instance.

    Args:
        api_key: Google AI API key (optional).
        model_name: Model name (optional).
        temperature: Temperature parameter (optional).
        max_tokens: Max tokens (optional).

    Returns:
        GeminiLLM instance.
    """
    return GeminiLLM(
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )
