"""Full debate flow tests (Task 5.2).

Runs two cases end-to-end across three rounds, verifies cross-references,
and writes a results summary to `ai-judge/results/debate_flow_test_results.json`.

Run:
  python -m unittest ai-judge/tests/test_debate_flow.py -v
"""
from __future__ import annotations

import json
import sys
import time
import unittest
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.agents.emotional_lawyer import EmotionalLawyer
from backend.agents.logical_lawyer import LogicalLawyer
from backend.agents.judge import JudgeAgent
from backend.models.schemas import CaseInput
from backend.config import settings


RESULTS_FILE = ROOT / "results" / "debate_flow_test_results.json"
TEST_CASES_FILE = ROOT / "tests" / "test_cases.json"


STOPWORDS = set(
    "the a an and or but if then because therefore hence so to for of in on at from by with about as is are was were be been being this that these those it its it's i you he she we they our my your their".split()
)


def tokenize(text: str) -> List[str]:
    return [w.lower().strip(".,!?;:\"'()[]{}") for w in text.split() if w.strip()]


def overlap_ratio(text_a: str, text_b: str) -> float:
    a = [w for w in tokenize(text_a) if w not in STOPWORDS]
    b = [w for w in tokenize(text_b) if w not in STOPWORDS]
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def wc(text: str) -> int:
    return len([w for w in text.split() if w.strip()])


class TestDebateFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Load two sample cases
        data = json.loads(TEST_CASES_FILE.read_text(encoding="utf-8"))
        cls.cases = data["cases"][:2]
        cls.emo = EmotionalLawyer()
        cls.log = LogicalLawyer()
        cls.judge = JudgeAgent()
        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        cls.results: Dict[str, List[Dict[str, object]]] = {"debates": []}

    def _run_single_debate(self, title: str, description: str) -> Tuple[Dict[str, object], Dict[str, object]]:
        case = CaseInput(title=title, description=description)

        t0 = time.time()
        # Round 1
        r1_start = time.time()
        emo_open = self.emo.generate_opening(case.description)
        log_open = self.log.generate_opening(case.description)
        r1_time = time.time() - r1_start
        r1 = [emo_open, log_open]

        # Round 2 (reference R1)
        r2_start = time.time()
        emo_counter = self.emo.generate_counter(case.description, log_open)
        log_counter = self.log.generate_counter(case.description, emo_open)
        r2_time = time.time() - r2_start
        r2 = [emo_counter, log_counter]

        # Round 3 (reference R2)
        r3_start = time.time()
        emo_rebuttal = self.emo.generate_rebuttal(case.description, log_counter, your_round2=emo_counter)
        log_rebuttal = self.log.generate_rebuttal(case.description, emo_counter, your_round2=log_counter)
        r3_time = time.time() - r3_start
        r3 = [emo_rebuttal, log_rebuttal]

        # Judge
        judge_start = time.time()
        all_texts = r1 + r2 + r3
        verdict = self.judge.evaluate_debate(case.description, all_texts)
        judge_time = time.time() - judge_start

        total_time = time.time() - t0

        transcript = {
            "debate_id": f"test-{int(time.time())}",
            "case": {"title": case.title, "description": case.description},
            "rounds": [r1, r2, r3],
            "verdict": verdict,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "complete",
        }

        # Build results entry
        entry: Dict[str, object] = {
            "title": title,
            "times_sec": {
                "round1": round(r1_time, 2),
                "round2": round(r2_time, 2),
                "round3": round(r3_time, 2),
                "judge": round(judge_time, 2),
                "total": round(total_time, 2),
            },
            "word_counts": {
                "r1": [wc(r1[0]), wc(r1[1])],
                "r2": [wc(r2[0]), wc(r2[1])],
                "r3": [wc(r3[0]), wc(r3[1])],
            },
            "verdict": {
                "emotional_score": int(max(0, min(100, verdict.get("emotional_score", 0)))),
                "logical_score": int(max(0, min(100, verdict.get("logical_score", 0)))),
                "winner": verdict.get("winner", "tie"),
            },
        }

        return transcript, entry

    def test_full_debate_two_cases(self):
        debates_entries: List[Dict[str, object]] = []
        for case_obj in self.cases:
            title = case_obj["title"]
            description = case_obj["description"]
            transcript, entry = self._run_single_debate(title, description)

            # Assertions: all rounds present
            self.assertEqual(len(transcript["rounds"]), settings.NUM_ROUNDS)
            for rnd in transcript["rounds"]:
                self.assertEqual(len(rnd), 2)
                for arg_text in rnd:
                    self.assertTrue(len(arg_text.strip()) > 0)

            # Cross-references (Round 2 references Round 1; Round 3 references Round 2)
            emo_r2, log_r2 = transcript["rounds"][1]
            emo_ref = overlap_ratio(emo_r2, transcript["rounds"][0][1])
            log_ref = overlap_ratio(log_r2, transcript["rounds"][0][0])
            self.assertGreaterEqual(emo_ref, 0.03, msg=f"Emotional R2 should reference Logical R1 (overlap={emo_ref:.3f})")
            self.assertGreaterEqual(log_ref, 0.03, msg=f"Logical R2 should reference Emotional R1 (overlap={log_ref:.3f})")

            emo_r3, log_r3 = transcript["rounds"][2]
            emo_ref3 = overlap_ratio(emo_r3, log_r2)
            log_ref3 = overlap_ratio(log_r3, emo_r2)
            self.assertGreaterEqual(emo_ref3, 0.03, msg=f"Emotional R3 should reference Logical R2 (overlap={emo_ref3:.3f})")
            self.assertGreaterEqual(log_ref3, 0.03, msg=f"Logical R3 should reference Emotional R2 (overlap={log_ref3:.3f})")

            # Judge verdict basics
            v = transcript["verdict"]
            self.assertIn(v.get("winner"), {"emotional", "logical", "tie"})
            self.assertTrue(0 <= v.get("emotional_score", 0) <= 100)
            self.assertTrue(0 <= v.get("logical_score", 0) <= 100)

            # Performance bound: each debate < 5 minutes
            self.assertLessEqual(entry["times_sec"]["total"], 300.0)

            debates_entries.append(entry)

        # Write results file
        summary = {
            "status": "success",
            "debates": debates_entries,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        RESULTS_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        # Sanity check file wrote
        self.assertTrue(RESULTS_FILE.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
