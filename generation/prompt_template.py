"""
Prompt engineering and template management.

Constructs robust prompts that ensure the LLM answers only from
provided context and refuses to hallucinate information.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class PromptTemplate:
    """Manage prompts and context injection."""

    # Main RAG prompt template
    RAG_PROMPT_TEMPLATE = """You are an AI assistant that answers questions based exclusively on the provided context.

Context Information:
{context}

User Question:
{question}

Instructions:
1. Answer ONLY using information from the provided context.
2. Do not invent, assume, or use external knowledge.
3. If the answer is not available in the context, clearly state: "I do not have enough information in the documents to answer this question."
4. Be concise and direct in your response.
5. Cite the source document when relevant.

Answer:"""

    # System prompt for context
    SYSTEM_PROMPT = """You are a knowledgeable AI assistant specialized in answering questions based on provided documents. 
Your responses must be grounded exclusively in the provided context. 
Never hallucinate or provide information not found in the documents.
When information is not available, be honest and state that you don't have enough information."""

    def __init__(self, custom_template: str = None):
        """
        Initialize prompt template.

        Args:
            custom_template: Custom prompt template string.
                            Uses default RAG template if not provided.
        """
        self.template = custom_template or self.RAG_PROMPT_TEMPLATE
        logger.info("PromptTemplate initialized")

    def format_prompt(self, context: str, question: str) -> str:
        """
        Format a complete prompt with context and question.

        Args:
            context: Formatted context from retrieved documents.
            question: User's question.

        Returns:
            Complete formatted prompt.
        """
        if not context or not context.strip():
            logger.warning("Empty context provided to prompt template")
            context = "No relevant context found in the documents."

        if not question or not question.strip():
            logger.warning("Empty question provided to prompt template")
            return ""

        formatted_prompt = self.template.format(
            context=context,
            question=question
        )

        logger.info(f"Prompt formatted successfully")
        return formatted_prompt

    def validate_context(self, context: str, min_length: int = 10) -> bool:
        """
        Validate that context has sufficient content.

        Args:
            context: Context string to validate.
            min_length: Minimum required length.

        Returns:
            True if context is valid, False otherwise.
        """
        if not context or len(context) < min_length:
            logger.warning(f"Context too short or empty (length: {len(context) if context else 0})")
            return False
        return True

    def validate_question(self, question: str, min_length: int = 3) -> bool:
        """
        Validate that question is properly formatted.

        Args:
            question: Question string to validate.
            min_length: Minimum required length.

        Returns:
            True if question is valid, False otherwise.
        """
        if not question or len(question.strip()) < min_length:
            logger.warning(f"Question too short (length: {len(question.strip()) if question else 0})")
            return False
        return True

    @staticmethod
    def clean_response(response: str) -> str:
        """
        Clean and normalize LLM response.

        Args:
            response: Raw response from LLM.

        Returns:
            Cleaned response.
        """
        if not response:
            return ""

        # Remove extra whitespace
        cleaned = " ".join(response.split())

        # Remove common artifacts
        if cleaned.startswith("Answer:"):
            cleaned = cleaned[7:].strip()

        return cleaned

    def get_template_info(self) -> Dict[str, Any]:
        """
        Get information about the template.

        Returns:
            Dictionary with template information.
        """
        return {
            "template_length": len(self.template),
            "has_context_placeholder": "{context}" in self.template,
            "has_question_placeholder": "{question}" in self.template,
            "system_prompt_length": len(self.SYSTEM_PROMPT)
        }


class ResponseEvaluator:
    """Evaluate and validate LLM responses."""

    # Phrases indicating insufficient information
    INSUFFICIENT_INFO_PHRASES = [
        "i don't have",
        "i do not have",
        "insufficient information",
        "not available",
        "not found",
        "not mentioned",
        "not provided",
        "cannot find",
        "cannot determine"
    ]

    @staticmethod
    def is_insufficient_answer(response: str) -> bool:
        """
        Check if response indicates insufficient information.

        Args:
            response: LLM response to check.

        Returns:
            True if response indicates insufficient information.
        """
        if not response:
            return True

        response_lower = response.lower()

        for phrase in ResponseEvaluator.INSUFFICIENT_INFO_PHRASES:
            if phrase in response_lower:
                return True

        return False

    @staticmethod
    def check_hallucination_risk(response: str, context: str) -> bool:
        """
        Check if response might contain hallucinated information.

        Args:
            response: LLM response to check.
            context: Original context provided to LLM.

        Returns:
            True if hallucination risk is detected.
        """
        if not response or not context:
            return False

        # Simple heuristic: check for statements not in context
        # This is a basic check - in production, use more sophisticated methods
        response_words = set(response.lower().split())
        context_words = set(context.lower().split())

        # If response has many unique words not in context, might be hallucination
        unique_words = response_words - context_words
        if len(unique_words) > len(response_words) * 0.5:
            logger.warning("Potential hallucination detected in response")
            return True

        return False


if __name__ == "__main__":
    # Example usage
    logger.info("=== Prompt Template Example ===")

    prompt_template = PromptTemplate()

    # Example context and question
    context = "Machine learning is a subset of artificial intelligence."
    question = "What is machine learning?"

    # Format prompt
    formatted_prompt = prompt_template.format_prompt(context, question)
    logger.info("Formatted Prompt:")
    logger.info(formatted_prompt)

    # Get template info
    info = prompt_template.get_template_info()
    logger.info("Template Info:")
    for key, value in info.items():
        logger.info(f"  {key}: {value}")

    # Evaluate responses
    evaluator = ResponseEvaluator()
    test_responses = [
        "I do not have enough information to answer this.",
        "Machine learning is a subset of AI that learns from data.",
        "This process involves quantum computing and time travel."
    ]

    logger.info("Response Evaluation:")
    for resp in test_responses:
        insufficient = evaluator.is_insufficient_answer(resp)
        logger.info(f"  '{resp[:50]}...' - Insufficient: {insufficient}")
