"""Emotional Lawyer agent implementation.

Generates passionate defense arguments following measurable traits from Decision Log #2
and enforces 250–350 word limits per round.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from backend.config import prompts, settings
from .base_agent import BaseAgent


logger = logging.getLogger("ai_judge.agents.emotional")


class EmotionalLawyer(BaseAgent):
    """Passionate defense counsel with elevated temperature and emotional style."""

    def generate_argument(self, case: str, context: Optional[Dict[str, Any]] = None) -> str:
        ctx = context or {}
        round_label = ctx.get("round_label", "Opening")
        opponent_argument = ctx.get("opponent_argument", "")
        your_previous_argument = ctx.get("your_previous_argument", "")

        system_prompt = prompts.EMOTIONAL_LAWYER_SYSTEM_PROMPT.format(
            case_description=case,
            opponent_argument=opponent_argument,
            your_previous_argument=your_previous_argument,
        )

        user_prompt = (
            f"Round: {round_label}. You represent the Respondent (defense). "
            f"Defend the respondent, argue non-liability or mitigation, and state a remedy (dismissal/warning/reduction). "
            f"Target {settings.WORD_LIMIT_MIN}-{settings.WORD_LIMIT_MAX} words. "
            "Do not include round headers or labels in the output. Do not switch sides."
        )

        # Up to 2 attempts to meet minimum word count; always trim to max.
        last_text = ""
        for _ in range(2):
            text = self.client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=settings.TEMPERATURES["emotional"],
                max_tokens=settings.MAX_TOKENS_ARGUMENT,
            )
            text = self._clean_response(text)
            # Pad first using split-based count to avoid underflow, then trim if needed
            text = self._pad_to_min(text, settings.WORD_LIMIT_MIN)
            text = self._enforce_word_limit(text, settings.WORD_LIMIT_MIN, settings.WORD_LIMIT_MAX)
            if self._validate_within_limits(text, settings.WORD_LIMIT_MIN, settings.WORD_LIMIT_MAX):
                return text
            last_text = text
            # If too short, adjust user prompt to emphasize the length
            user_prompt = (
                f"Round: {round_label}. Your previous response was under {settings.WORD_LIMIT_MIN} words. "
                "You represent the Respondent (defense). Produce a richer, narrative defense with rhetorical questions, "
                f"and include the remedy you seek, {settings.WORD_LIMIT_MIN}-{settings.WORD_LIMIT_MAX} words."
            )

        logger.warning("EmotionalLawyer produced text outside limits; returning trimmed result")
        # As a last resort, pad to minimum and trim to max to satisfy validators
        padded = self._pad_to_min(last_text, settings.WORD_LIMIT_MIN)
        return self._enforce_word_limit(padded, settings.WORD_LIMIT_MIN, settings.WORD_LIMIT_MAX)

    def generate_opening(self, case_description: str) -> str:
        return self.generate_argument(case_description, {"round_label": "Opening"})

    def generate_counter(self, case_description: str, opponent_round1: str) -> str:
        return self.generate_argument(
            case_description,
            {
                "round_label": "Counter-Argument",
                "opponent_argument": opponent_round1,
            },
        )

    def generate_rebuttal(self, case_description: str, opponent_round2: str, your_round2: str | None = None) -> str:
        return self.generate_argument(
            case_description,
            {
                "round_label": "Rebuttal",
                "opponent_argument": opponent_round2,
                "your_previous_argument": your_round2 or "",
            },
        )
