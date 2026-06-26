"""
Groq LLM integration for response generation.

Manages connection to Groq API and provides inference capabilities
using the llama-3.3-70b-versatile model.
"""

import logging
from langchain_groq import ChatGroq
from utils.config import Config

logger = logging.getLogger(__name__)


class GroqLLM:
    """Manage Groq LLM for response generation."""

    _instance = None  # Singleton instance

    def __init__(
        self,
        api_key: str = None,
        model_name: str = None,
        temperature: float = None,
        max_tokens: int = None
    ):
        """
        Initialize Groq LLM.

        Args:
            api_key: Groq API key. Defaults to Config.GROQ_API_KEY
            model_name: Model name to use. Defaults to Config.LLM_MODEL_NAME
            temperature: Model temperature (0-1).
                        Defaults to Config.LLM_TEMPERATURE
            max_tokens: Maximum tokens for response.
                       Defaults to Config.LLM_MAX_TOKENS
        """
        self.api_key = api_key or Config.GROQ_API_KEY
        self.model_name = model_name or Config.LLM_MODEL_NAME
        self.temperature = temperature if temperature is not None else Config.LLM_TEMPERATURE
        self.max_tokens = max_tokens or Config.LLM_MAX_TOKENS

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not provided. Set it in .env or pass as argument."
            )

        # Initialize the ChatGroq model
        self.model = ChatGroq(
            groq_api_key=self.api_key,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        logger.info(
            f"GroqLLM initialized with model: {self.model_name}, "
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
            logger.warning("Empty prompt provided to LLM")
            return ""

        logger.info("Generating response from Groq LLM...")

        try:
            # Invoke the model
            response = self.model.invoke(prompt)

            # Extract text from the response
            response_text = response.content

            logger.info("Response generated successfully")
            return response_text

        except Exception as e:
            logger.error(f"Error generating response from Groq LLM: {str(e)}")
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
            logger.warning("Empty prompt provided to LLM streaming")
            return

        logger.info("Generating streaming response from Groq LLM...")

        try:
            for chunk in self.model.stream(prompt):
                yield chunk.content

        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            raise

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text (approximation).

        Args:
            text: Text to count tokens for.

        Returns:
            Approximate token count.
        """
        # Simple approximation: ~4 characters per token on average
        return len(text) // 4

    def get_model_info(self) -> dict:
        """
        Get information about the configured model.

        Returns:
            Dictionary with model information.
        """
        return {
            "provider": "groq",
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_key_configured": bool(self.api_key)
        }


def get_groq_llm(
    api_key: str = None,
    model_name: str = None,
    temperature: float = None,
    max_tokens: int = None
) -> GroqLLM:
    """
    Get or create singleton GroqLLM instance.

    Args:
        api_key: Groq API key (optional).
        model_name: Model name (optional).
        temperature: Temperature parameter (optional).
        max_tokens: Max tokens (optional).

    Returns:
        GroqLLM instance.
    """
    return GroqLLM(
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )


if __name__ == "__main__":
    # Example usage
    logger.info("=== Groq LLM Example ===")

    try:
        llm = get_groq_llm()

        # Get model info
        info = llm.get_model_info()
        logger.info("Model Information:")
        for key, value in info.items():
            logger.info(f"  {key}: {value}")

        # Generate a test response
        test_prompt = "What is Generative AI in one sentence?"
        response = llm.generate_response(test_prompt)
        logger.info(f"Response: {response}")

    except Exception as e:
        logger.error(f"Error in example: {str(e)}")
