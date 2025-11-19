"""Pydantic schemas for AI Judge backend (Task 3.5).

Includes models:
- CaseInput
- Argument (with word-count validation and lawyer/round checks)
- Verdict (with score ranges and criteria validation)
- DebateTranscript (structure and status validation)
"""
from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field, field_validator, model_validator

from backend.config import settings


def _word_count(text: str) -> int:
    return len([w for w in text.split() if w.strip()])


class CaseInput(BaseModel):
    """Input payload describing a case."""

    description: str = Field(..., min_length=10, description="Detailed case description")
    title: str = Field(..., min_length=3, description="Short case title")


class Argument(BaseModel):
    """A single lawyer argument for a given debate round."""

    lawyer: str = Field(..., description='"emotional" or "logical"')
    round_number: int = Field(..., ge=1, le=settings.NUM_ROUNDS, description="Round index (1-3)")
    content: str = Field(..., min_length=20, description="Argument body text")
    word_count: int = Field(..., description="Word count of content; auto-corrected to actual count")

    @field_validator("lawyer")
    @classmethod
    def _lawyer_allowed(cls, v: str) -> str:
        lv = v.strip().lower()
        if lv not in {"emotional", "logical"}:
            raise ValueError('lawyer must be "emotional" or "logical"')
        return lv

    @model_validator(mode="after")
    def _check_word_count(self) -> "Argument":
        wc = _word_count(self.content)
        # If caller supplied word_count, ensure it matches; else set it.
        if self.word_count != wc:
            # Allow silent correction instead of raising to be more ergonomic
            object.__setattr__(self, "word_count", wc)
        if not (settings.WORD_LIMIT_MIN <= wc <= settings.WORD_LIMIT_MAX):
            raise ValueError(
                f"argument word_count {wc} out of range {settings.WORD_LIMIT_MIN}-{settings.WORD_LIMIT_MAX}"
            )
        return self


class Verdict(BaseModel):
    """Judge verdict with overall scores and rubric details."""

    emotional_score: int = Field(..., ge=0, le=100)
    logical_score: int = Field(..., ge=0, le=100)
    winner: str = Field(..., description='"emotional", "logical", or "tie"')
    reasoning: str = Field(..., min_length=30, description="Judge reasoning text")
    criteria_scores: Dict[str, int] = Field(..., description="Rubric component scores (0-20 each)")

    @field_validator("winner")
    @classmethod
    def _winner_allowed(cls, v: str) -> str:
        lv = v.strip().lower()
        if lv not in {"emotional", "logical", "tie"}:
            raise ValueError('winner must be "emotional", "logical", or "tie"')
        return lv

    @field_validator("criteria_scores")
    @classmethod
    def _criteria_ok(cls, v: Dict[str, int]) -> Dict[str, int]:
        required = {"relevance", "coherence", "evidence", "persuasiveness", "rebuttal"}
        missing = required - set(v.keys())
        if missing:
            raise ValueError(f"criteria_scores missing keys: {sorted(missing)}")
        for k in required:
            val = v[k]
            if not isinstance(val, int) or not (0 <= val <= 20):
                raise ValueError(f"criteria_scores[{k}] must be int in [0, 20]")
        return v

    @model_validator(mode="after")
    def _reasoning_length_hint(self) -> "Verdict":
        # Soft check: encourage 300–400 words (Decision Log #4), but do not hard-fail
        wc = _word_count(self.reasoning)
        # No strict exception to keep API tolerant; backend can enforce elsewhere if needed
        return self


class DebateTranscript(BaseModel):
    """Full debate transcript with arguments per round and final verdict."""

    debate_id: str
    case: CaseInput
    rounds: List[List[Argument]]  # Round 1: [arg1, arg2], Round 2: [...], Round 3: [...]
    verdict: Verdict
    timestamp: str
    status: str  # "in_progress", "complete", or "failed"

    @field_validator("status")
    @classmethod
    def _status_allowed(cls, v: str) -> str:
        lv = v.strip().lower()
        if lv not in {"in_progress", "complete", "failed"}:
            raise ValueError('status must be "in_progress", "complete", or "failed"')
        return lv


class DebateRoles(BaseModel):
    """Configuration for which side each lawyer represents in a debate."""

    emotional_role: str = Field("prosecution", description='"prosecution" or "defense" for Emotional lawyer')

    @field_validator("emotional_role")
    @classmethod
    def _role_allowed(cls, v: str) -> str:
        lv = v.strip().lower()
        if lv not in {"prosecution", "defense"}:
            raise ValueError('emotional_role must be "prosecution" or "defense"')
        return lv

    @field_validator("rounds", check_fields=False)
    @classmethod
    def _rounds_shape(cls, v: List[List[Argument]]) -> List[List[Argument]]:
        # Ensure 3 rounds present with up to 2 arguments each; remain permissive to allow streaming/in_progress
        if not isinstance(v, list) or len(v) != settings.NUM_ROUNDS:
            raise ValueError(f"rounds must contain exactly {settings.NUM_ROUNDS} lists (one per round)")
        for i, rnd in enumerate(v, start=1):
            if not isinstance(rnd, list):
                raise ValueError(f"rounds[{i}] must be a list of Argument")
            if len(rnd) > 2:
                raise ValueError(f"rounds[{i}] must have at most 2 arguments (emotional/logical)")
        return v
