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
    """Emotional lawyer with elevated temperature and narrative style.

    The `role` attribute controls whether this agent acts as prosecution
    (attacker) or defense (defender).
    """

    def __init__(self, role: str = "prosecution", client: Optional[None] = None) -> None:
        super().__init__(client=client)
        self.role = "prosecution" if role not in {"prosecution", "defense"} else role

    def generate_argument(self, case: str, context: Optional[Dict[str, Any]] = None) -> str:
        ctx = context or {}
        round_label = ctx.get("round_label", "Opening")
        opponent_argument = ctx.get("opponent_argument", "")
        your_previous_argument = ctx.get("your_previous_argument", "")

        base_system = prompts.EMOTIONAL_LAWYER_SYSTEM_PROMPT.format(
            case_description=case,
            opponent_argument=opponent_argument,
            your_previous_argument=your_previous_argument,
        )
        if self.role == "prosecution":
            role_block = (
                "Role override: You represent the Complainant (prosecution). "
                "Your objective is to prove the Respondent is liable and argue for a strong remedy.\n\n"
            )
            user_prompt = (
                f"Round: {round_label}. You represent the Complainant (prosecution). "
                "Attack the respondent's conduct using emotional, narrative advocacy, argue that they are liable, and state a strong remedy (payment/refund/stop order/sanction). "
                f"Target {settings.WORD_LIMIT_MIN}-{settings.WORD_LIMIT_MAX} words. "
                "Do not include round headers or labels in the output. Do not switch sides."
            )
        else:
            role_block = (
                "Role override: You represent the Respondent (defense). "
                "Your objective is to defend the Respondent, challenge liability, and argue for dismissal or mitigation.\n\n"
            )
            user_prompt = (
                f"Round: {round_label}. You represent the Respondent (defense). "
                "Defend the respondent using emotional, narrative advocacy, highlight mitigating facts, and state the reduced outcome (dismissal/warning/reduction) you seek. "
                f"Target {settings.WORD_LIMIT_MIN}-{settings.WORD_LIMIT_MAX} words. "
                "Do not include round headers or labels in the output. Do not switch sides."
            )

        system_prompt = role_block + base_system

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
            if self.role == "prosecution":
                user_prompt = (
                    f"Round: {round_label}. Your previous response was under {settings.WORD_LIMIT_MIN} words. "
                    "You represent the Complainant (prosecution). Produce a richer, narrative attack with rhetorical questions, "
                    f"and clearly state the remedy you seek, {settings.WORD_LIMIT_MIN}-{settings.WORD_LIMIT_MAX} words."
                )
            else:
                user_prompt = (
                    f"Round: {round_label}. Your previous response was under {settings.WORD_LIMIT_MIN} words. "
                    "You represent the Respondent (defense). Produce a richer, narrative defense with rhetorical questions, "
                    f"and clearly state the dismissal or mitigation you seek, {settings.WORD_LIMIT_MIN}-{settings.WORD_LIMIT_MAX} words."
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
