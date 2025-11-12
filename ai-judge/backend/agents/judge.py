"""Judge agent implementation: evaluates both sides and returns a JSON verdict."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from backend.config import prompts, settings
from .base_agent import BaseAgent


logger = logging.getLogger("ai_judge.agents.judge")


class JudgeAgent(BaseAgent):
    def generate_argument(self, case: str, context=None) -> str:  # pragma: no cover - unused for judge
        return ""

    def evaluate_debate(self, case: str, all_arguments: List[str]) -> Dict[str, Any]:
        """Evaluate the full debate and return a structured verdict.

        Args:
            case: Case description string.
            all_arguments: Flattened list of arguments in order:
                [Emo R1, Log R1, Emo R2, Log R2, Emo R3, Log R3]

        Returns:
            Dict with keys: emotional_score, logical_score, winner, reasoning, criteria_scores
        """

        def _fmt_arg(role: str, round_n: int, text: str) -> str:
            return f"[{role} | Round {round_n}]\n{text.strip()}\n"

        if len(all_arguments) != 6:
            logger.warning("Judge received %d arguments; expected 6", len(all_arguments))

        lines: List[str] = []
        roles = [
            ("emotional", 1), ("logical", 1),
            ("emotional", 2), ("logical", 2),
            ("emotional", 3), ("logical", 3),
        ]
        for (role, rnd), text in zip(roles, all_arguments):
            lines.append(_fmt_arg("Emotional" if role == "emotional" else "Logical", rnd, text))

        debate_block = "\n".join(lines)

        system_prompt = prompts.JUDGE_SYSTEM_PROMPT.format(
            case_description=case,
            opponent_argument="",
            your_previous_argument="",
        )
        user_prompt = (
            "Evaluate the following debate transcript according to the rubric and output ONLY the JSON object described.\n\n"
            f"Transcript:\n{debate_block}\n"
        )

        raw = self.client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=settings.TEMPERATURES["judge"],
            max_tokens=settings.MAX_TOKENS_VERDICT,
            extra_options={"format": "json"},
        )
        verdict = self._parse_verdict_json(raw)
        if verdict.get("reasoning", "").startswith("Model did not return strict JSON"):
            # Try a one-shot repair by asking the model to reformat strictly as JSON
            repair_prompt = (
                "Reformat the following content as a STRICT JSON object using this schema with correct keys and types. "
                "Output exactly one JSON object and nothing else.\n\n"
                "Schema: {\n"
                "  \"emotional_score\": int 0-100,\n"
                "  \"logical_score\": int 0-100,\n"
                "  \"winner\": one of [\"emotional\", \"logical\", \"tie\"],\n"
                "  \"reasoning\": string (>= 30 words),\n"
                "  \"criteria_scores\": { \"relevance\":0-20, \"coherence\":0-20, \"evidence\":0-20, \"persuasiveness\":0-20, \"rebuttal\":0-20 }\n"
                "}\n\n"
                f"Content:\n{raw}\n"
            )
            repaired = self.client.generate(
                prompt=repair_prompt,
                system_prompt="Return ONLY strict JSON per schema.",
                temperature=0.0,
                max_tokens=settings.MAX_TOKENS_VERDICT,
                extra_options={"format": "json"},
            )
            repaired_verdict = self._parse_verdict_json(repaired)
            # Accept repaired verdict if it looks valid
            if isinstance(repaired_verdict.get("emotional_score"), int) and isinstance(repaired_verdict.get("logical_score"), int):
                verdict = repaired_verdict
        return verdict

    @staticmethod
    def _parse_verdict_json(text: str) -> Dict[str, Any]:
        """Parse a JSON verdict robustly.

        Strategy:
        1) Strip code fences and whitespace.
        2) Extract the first balanced JSON object { ... } respecting strings.
        3) Attempt json.loads.
        4) If fail, try a light "repair" (remove trailing commas) and load again.
        5) If still fail, attempt heuristic extraction of fields to avoid neutral ties.
        """

        def strip_fences(s: str) -> str:
            s = s.strip()
            s = re.sub(r"^```[a-zA-Z]*\n|```$", "", s).strip()
            return s

        def extract_first_object(s: str) -> Optional[str]:
            start = s.find("{")
            if start == -1:
                return None
            depth = 0
            in_str = False
            esc = False
            for i, ch in enumerate(s[start:], start=start):
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                else:
                    if ch == '"':
                        in_str = True
                        continue
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            return s[start : i + 1]
            return None

        def try_load(s: str) -> Optional[Dict[str, Any]]:
            try:
                return json.loads(s)
            except Exception:
                return None

        def repair_json(s: str) -> str:
            # Remove trailing commas before } or ]
            s = re.sub(r",\s*([}\]])", r"\1", s)
            return s

        cleaned = strip_fences(text)
        candidate = extract_first_object(cleaned) or cleaned

        data = try_load(candidate)
        if data is None:
            data = try_load(repair_json(candidate))

        if data is None:
            # Heuristic extraction to avoid neutral tie
            logger.error("Judge JSON parse failed; applying heuristic extraction")
            emo = _extract_int(cleaned, r"emotional[_\s-]*score\D{0,10}(\d{1,3})") or 50
            log = _extract_int(cleaned, r"logical[_\s-]*score\D{0,10}(\d{1,3})") or 50
            emo = int(max(0, min(100, emo)))
            log = int(max(0, min(100, log)))
            if emo > log:
                winner = "emotional"
            elif log > emo:
                winner = "logical"
            else:
                # Try to infer winner tokens
                w = _extract_winner(cleaned) or "tie"
                winner = w if w in {"emotional", "logical", "tie"} else "tie"
            return {
                "emotional_score": emo,
                "logical_score": log,
                "winner": winner,
                "reasoning": "Model did not return strict JSON; applied heuristic parse to extract scores and winner.",
                "criteria_scores": {
                    "relevance": 10,
                    "coherence": 10,
                    "evidence": 10,
                    "persuasiveness": 10,
                    "rebuttal": 10,
                },
            }

        # Normalize and validate expected keys when JSON parsed
        emo = int(max(0, min(100, data.get("emotional_score", 0))))
        log = int(max(0, min(100, data.get("logical_score", 0))))
        winner = data.get("winner")
        if winner not in {"emotional", "logical", "tie"}:
            winner = "emotional" if emo > log else "logical" if log > emo else "tie"
        reasoning = str(data.get("reasoning", ""))
        criteria = data.get("criteria_scores") or {}
        criteria = {
            "relevance": int(max(0, min(20, criteria.get("relevance", 0)))),
            "coherence": int(max(0, min(20, criteria.get("coherence", 0)))),
            "evidence": int(max(0, min(20, criteria.get("evidence", 0)))),
            "persuasiveness": int(max(0, min(20, criteria.get("persuasiveness", 0)))),
            "rebuttal": int(max(0, min(20, criteria.get("rebuttal", 0)))),
        }

        return {
            "emotional_score": emo,
            "logical_score": log,
            "winner": winner,
            "reasoning": reasoning,
            "criteria_scores": criteria,
        }


def _extract_int(s: str, pattern: str) -> Optional[int]:
    m = re.search(pattern, s, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _extract_winner(s: str) -> Optional[str]:
    m = re.search(r"winner\D{0,10}(emotional|logical|tie)", s, flags=re.IGNORECASE)
    return m.group(1).lower() if m else None
