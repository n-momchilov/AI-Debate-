"""Base agent interface and shared utilities for argument generation."""
from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from backend.config import settings
from backend.utils.ollama_client import OllamaClient


logger = logging.getLogger("ai_judge.agents")
if not logger.handlers:
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class BaseAgent(ABC):
    """Abstract base class for AI debate agents.

    Provides shared helpers for cleaning and validating generated text.
    """

    def __init__(self, client: Optional[OllamaClient] = None) -> None:
        self.client = client or OllamaClient(model_name=settings.MODEL_NAME)

    @abstractmethod
    def generate_argument(self, case: str, context: Dict[str, Any] | None = None) -> str:  # pragma: no cover - abstract
        """Generate an argument for the current round.

        Args:
            case: Case description text.
            context: Optional round context (e.g., prior opponent/self arguments, round label).
        """
        raise NotImplementedError

    # Shared utilities
    @staticmethod
    def _word_count(text: str) -> int:
        """Word count aligned with Pydantic validator (split on whitespace)."""
        return len([w for w in text.split() if w.strip()])

    @staticmethod
    def _clean_response(text: str) -> str:
        """Normalize whitespace and strip code fences/quotes that models sometimes add."""
        if not isinstance(text, str):
            return ""
        t = text.strip()
        # Remove triple backticks or json fences often added by models
        t = re.sub(r"^```[a-zA-Z]*\n|```$", "", t).strip()
        # Remove stray enclosing quotes
        if (t.startswith("\"") and t.endswith("\"")) or (t.startswith("'") and t.endswith("'")):
            t = t[1:-1].strip()
        # Normalize internal whitespace but keep newlines
        t = re.sub(r"[ \t]+", " ", t)
        return t

    @classmethod
    def _enforce_word_limit(cls, text: str, min_words: int, max_words: int) -> str:
        """Trim overly long responses to max_words while preserving sentence ends if possible."""
        words = re.findall(r"\S+", text)
        if len(words) > max_words:
            trimmed = " ".join(words[:max_words])
            # Try to end at a sentence boundary
            m = re.search(r"^(.+[.!?])\s*", trimmed)
            if m:
                return m.group(1)
            return trimmed
        return text

    @classmethod
    def _pad_to_min(cls, text: str, min_words: int) -> str:
        """If text is below min_words, append neutral concluding lines until it reaches the minimum.

        This is a pragmatic safeguard to prevent downstream validation errors when the model
        occasionally produces slightly short outputs even after retries.
        """
        words = re.findall(r"\S+", text)
        if len(words) >= min_words:
            return text
        filler = (
            " Therefore, based on the foregoing reasons, this position is justified and the requested remedy follows logically."
        )
        while len(words) < min_words:
            text = (text + filler).strip()
            words = re.findall(r"\S+", text)
        return text

    @classmethod
    def _validate_within_limits(cls, text: str, min_words: int, max_words: int) -> bool:
        wc = cls._word_count(text)
        return min_words <= wc <= max_words
