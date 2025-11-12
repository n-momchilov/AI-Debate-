"""Tests for single-argument generation and judge evaluation (Task 3.6).

Run with:
  python -m unittest ai-judge/tests/test_agents.py
"""
from __future__ import annotations

import json
import sys
import time
import unittest
from pathlib import Path
from typing import List, Tuple


# Ensure we can import the backend package
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.agents.emotional_lawyer import EmotionalLawyer
from backend.agents.logical_lawyer import LogicalLawyer
from backend.agents.judge import JudgeAgent
from backend.config import settings


CASE = (
    "My neighbor plays loud music every night until 2 AM despite multiple complaints. "
    "They claim 'their apartment, their rules.' I want them to stop after 10 PM."
)


def word_count(text: str) -> int:
    return len([w for w in text.split() if w.strip()])


def jaccard_difference(a: str, b: str) -> float:
    sa = set(w.lower() for w in a.split())
    sb = set(w.lower() for w in b.split())
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = max(1, len(sa | sb))
    return 1.0 - (inter / union)


class TestAgents(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.emo = EmotionalLawyer()
        cls.log = LogicalLawyer()
        cls.judge = JudgeAgent()

    def test_emotional_and_logical_opening_within_limits(self):
        start = time.time()
        emo_text = self.emo.generate_opening(CASE)
        log_text = self.log.generate_opening(CASE)
        emo_wc = word_count(emo_text)
        log_wc = word_count(log_text)
        self.assertTrue(
            settings.WORD_LIMIT_MIN <= emo_wc <= settings.WORD_LIMIT_MAX,
            msg=f"Emotional words {emo_wc} not within {settings.WORD_LIMIT_MIN}-{settings.WORD_LIMIT_MAX}",
        )
        self.assertTrue(
            settings.WORD_LIMIT_MIN <= log_wc <= settings.WORD_LIMIT_MAX,
            msg=f"Logical words {log_wc} not within {settings.WORD_LIMIT_MIN}-{settings.WORD_LIMIT_MAX}",
        )
        # Keep execution time bounded for CI-like runs
        self.assertLess(time.time() - start, 180.0)

    def test_temperature_variance_visible(self):
        # Emotional (higher temperature)
        emo_texts: List[str] = [self.emo.generate_opening(CASE) for _ in range(3)]
        # Logical (lower temperature)
        log_texts: List[str] = [self.log.generate_opening(CASE) for _ in range(3)]

        def avg_pairwise_diff(texts: List[str]) -> float:
            diffs: List[float] = []
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    diffs.append(jaccard_difference(texts[i], texts[j]))
            return sum(diffs) / max(1, len(diffs))

        emo_avg = avg_pairwise_diff(emo_texts)
        log_avg = avg_pairwise_diff(log_texts)

        # At least two unique outputs per agent
        self.assertGreaterEqual(len(set(emo_texts)), 2)
        self.assertGreaterEqual(len(set(log_texts)), 2)

        # Emotional variation should exceed Logical variation (visible difference)
        self.assertGreater(emo_avg, log_avg, msg=f"Expected emo variance > log variance (emo={emo_avg:.2f}, log={log_avg:.2f})")

    def test_judge_evaluates_mock_debate(self):
        # Generate one opening each
        emo_open = self.emo.generate_opening(CASE)
        log_open = self.log.generate_opening(CASE)
        # Create a mock 3-round transcript by reusing openings to keep test short
        all_args = [
            emo_open, log_open,
            emo_open, log_open,
            emo_open, log_open,
        ]
        verdict = self.judge.evaluate_debate(CASE, all_args)
        # Basic structure checks
        for key in ("emotional_score", "logical_score", "winner", "reasoning", "criteria_scores"):
            self.assertIn(key, verdict)
        self.assertIsInstance(verdict["emotional_score"], int)
        self.assertIsInstance(verdict["logical_score"], int)
        self.assertIn(verdict["winner"], {"emotional", "logical", "tie"})
        self.assertIsInstance(verdict["criteria_scores"], dict)


if __name__ == "__main__":
    unittest.main(verbosity=2)

