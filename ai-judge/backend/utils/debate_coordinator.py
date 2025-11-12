"""Debate coordinator: orchestrates 3-round debate and invokes the judge."""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Callable, List, Tuple

from backend.config import settings
from backend.models.schemas import Argument, CaseInput, DebateTranscript, Verdict


logger = logging.getLogger("ai_judge.debate_coordinator")
if not logger.handlers:
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class DebateCoordinator:
    """Coordinates a full 3-round debate and returns a DebateTranscript."""

    def __init__(self, emotional_lawyer, logical_lawyer, judge) -> None:
        self.emotional_lawyer = emotional_lawyer
        self.logical_lawyer = logical_lawyer
        self.judge = judge

    # Public API
    def run_debate(self, case: CaseInput) -> DebateTranscript:
        start = time.time()
        debate_id = f"deb-{uuid.uuid4().hex[:8]}"
        logger.info("Starting debate %s for case: %s", debate_id, case.title)

        # Round 1: Opening arguments
        emo_open = self._retry(lambda: self.emotional_lawyer.generate_opening(case.description), label="Emotional R1")
        log_open = self._retry(lambda: self.logical_lawyer.generate_opening(case.description), label="Logical R1")
        r1 = [
            Argument(lawyer="emotional", round_number=1, content=emo_open, word_count=0),
            Argument(lawyer="logical", round_number=1, content=log_open, word_count=0),
        ]
        logger.info("Round 1 complete for %s", debate_id)

        # Round 2: Counter-arguments (each sees opponent's Round 1)
        emo_counter = self._retry(
            lambda: self.emotional_lawyer.generate_counter(case.description, log_open),
            label="Emotional R2",
        )
        log_counter = self._retry(
            lambda: self.logical_lawyer.generate_counter(case.description, emo_open),
            label="Logical R2",
        )
        r2 = [
            Argument(lawyer="emotional", round_number=2, content=emo_counter, word_count=0),
            Argument(lawyer="logical", round_number=2, content=log_counter, word_count=0),
        ]
        logger.info("Round 2 complete for %s", debate_id)

        # Round 3: Rebuttals (each sees opponent's Round 2 and may see own previous)
        emo_rebuttal = self._retry(
            lambda: self.emotional_lawyer.generate_rebuttal(case.description, log_counter, your_round2=emo_counter),
            label="Emotional R3",
        )
        log_rebuttal = self._retry(
            lambda: self.logical_lawyer.generate_rebuttal(case.description, emo_counter, your_round2=log_counter),
            label="Logical R3",
        )
        r3 = [
            Argument(lawyer="emotional", round_number=3, content=emo_rebuttal, word_count=0),
            Argument(lawyer="logical", round_number=3, content=log_rebuttal, word_count=0),
        ]
        logger.info("Round 3 complete for %s", debate_id)

        # Judge evaluation
        all_texts = [a.content for a in (r1 + r2 + r3)]
        verdict_dict = self._retry(lambda: self.judge.evaluate_debate(case.description, all_texts), label="Judge Verdict")
        verdict = Verdict(**verdict_dict)
        logger.info("Judge verdict ready for %s: winner=%s (E:%d L:%d)", debate_id, verdict.winner, verdict.emotional_score, verdict.logical_score)

        total_s = time.time() - start
        logger.info("Debate %s complete in %.1fs", debate_id, total_s)

        ts = datetime.now(timezone.utc).isoformat()
        transcript = DebateTranscript(
            debate_id=debate_id,
            case=case,
            rounds=[r1, r2, r3],
            verdict=verdict,
            timestamp=ts,
            status="complete",
        )
        return transcript

    # Internal helpers
    def _retry(self, func: Callable[[], str | dict], *, attempts: int = 3, label: str = "call") -> str | dict:
        """Retry wrapper for generation/evaluation calls with backoff and logging."""
        last_err: Exception | None = None
        for i in range(1, attempts + 1):
            try:
                return func()
            except Exception as e:
                last_err = e
                logger.error("%s failed (attempt %d/%d): %s", label, i, attempts, e)
                if i < attempts:
                    time.sleep(1.5 * i)
        assert last_err is not None
        raise last_err

