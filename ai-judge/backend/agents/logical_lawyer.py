"""Logical Lawyer agent implementation.

Produces structured, low-emotion arguments with numbered points and if–then logic.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from backend.config import prompts, settings
from .base_agent import BaseAgent


logger = logging.getLogger("ai_judge.agents.logical")


class LogicalLawyer(BaseAgent):
    """Logical lawyer with low temperature and structured style.

    The `role` attribute controls whether this agent acts as prosecution
    or defense.
    """

    def __init__(self, role: str = "defense", client: Optional[None] = None) -> None:
        super().__init__(client=client)
        self.role = "defense" if role not in {"prosecution", "defense"} else role

    def generate_argument(self, case: str, context: Optional[Dict[str, Any]] = None) -> str:
        ctx = context or {}
        round_label = ctx.get("round_label", "Opening")
        opponent_argument = ctx.get("opponent_argument", "")
        your_previous_argument = ctx.get("your_previous_argument", "")

        base_system = prompts.LOGICAL_LAWYER_SYSTEM_PROMPT.format(
            case_description=case,
            opponent_argument=opponent_argument,
            your_previous_argument=your_previous_argument,
        )
        if self.role == "prosecution":
            role_block = (
                "Role override: You represent the Complainant (prosecution). "
                "Your objective is to prove the Respondent is liable using numbered, logical points.\n\n"
            )
            user_prompt = (
                f"Round: {round_label}. You represent the Complainant (prosecution). "
                "Present numbered points with explicit if-then logic to establish liability, and state a concrete remedy (e.g., stop order, refund, payment, fine). "
                f"Target {settings.WORD_LIMIT_MIN}-{settings.WORD_LIMIT_MAX} words; avoid emotional language. "
                "Do not include round headers or labels in the output. Do not switch sides."
            )
        else:
            role_block = (
                "Role override: You represent the Respondent (defense). "
                "Your objective is to challenge the complainant's case, highlight evidentiary gaps, and argue for dismissal or mitigation.\n\n"
            )
            user_prompt = (
                f"Round: {round_label}. You represent the Respondent (defense). "
                "Present numbered points with explicit if-then logic to challenge the complainant's case, highlight gaps in evidence, and argue for dismissal or mitigation. "
                f"Target {settings.WORD_LIMIT_MIN}-{settings.WORD_LIMIT_MAX} words; avoid emotional language. "
                "Do not include round headers or labels in the output. Do not switch sides."
            )

        system_prompt = role_block + base_system

        last_text = ""
        for _ in range(2):
            text = self.client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=settings.TEMPERATURES["logical"],
                max_tokens=settings.MAX_TOKENS_ARGUMENT,
            )
            text = self._clean_response(text)
            text = self._pad_to_min(text, settings.WORD_LIMIT_MIN)
            text = self._enforce_word_limit(text, settings.WORD_LIMIT_MIN, settings.WORD_LIMIT_MAX)
            if self._validate_within_limits(text, settings.WORD_LIMIT_MIN, settings.WORD_LIMIT_MAX):
                return text
            last_text = text
            if self.role == "prosecution":
                user_prompt = (
                    f"Round: {round_label}. Your previous response was under {settings.WORD_LIMIT_MIN} words. "
                    "You represent the Complainant (prosecution). Provide precise, numbered points with 2-3 if-then statements and a clear remedy, "
                    f"{settings.WORD_LIMIT_MIN}-{settings.WORD_LIMIT_MAX} words."
                )
            else:
                user_prompt = (
                    f"Round: {round_label}. Your previous response was under {settings.WORD_LIMIT_MIN} words. "
                    "You represent the Respondent (defense). Provide precise, numbered points with 2-3 if-then statements, stress reasonable doubt or necessity, and state the reduced outcome you seek, "
                    f"{settings.WORD_LIMIT_MIN}-{settings.WORD_LIMIT_MAX} words."
                )

        logger.warning("LogicalLawyer produced text outside limits; returning trimmed result")
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
