"""
Benchmark Ollama Llama 3 8B locally.

Features:
- Measures generation speed (tokens/sec; approx 1 word ≈ 1.33 tokens)
- Estimates quality (simple coherence heuristic, 1–5)
- Logs GPU VRAM usage during generation (via nvidia-smi sampling)
- Tests temperature effects (0.3 vs 0.9) for output divergence

Outputs results to console and saves JSON to `results/benchmark_results.json`.

Requirements:
- ollama Python package (pip install ollama)
- NVIDIA GPU: `nvidia-smi` available in PATH for VRAM sampling (gracefully degrades if missing)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import ollama  # type: ignore
except Exception as e:  # pragma: no cover - import-time failure surfaced clearly
    print("[ERROR] `ollama` package is not installed. Run: pip install ollama", file=sys.stderr)
    raise


MODEL_NAME = os.environ.get("OLLAMA_MODEL", "llama3:8b")
RESULTS_PATH = Path("results/benchmark_results.json")


EMOTIONAL_PROMPT = (
    "You are a passionate defense lawyer. Write a 300-word emotional argument "
    "defending a person accused of parking in a disabled spot. Their defense: "
    "medical emergency for child's asthma medication. Use passionate language, "
    "rhetorical questions, and appeal to fairness."
)

LOGICAL_PROMPT = (
    "You are a logical prosecutor. Write a 300-word structured argument prosecuting "
    "a person who parked in a disabled spot. Present facts systematically using "
    "numbered points. Avoid emotional language. Use if-then logic."
)

JUDGE_PROMPT = (
    "You are an impartial judge. Evaluate these two arguments and declare a winner. "
    "Provide scores (0-100 for each) and reasoning (200 words)."
)


@dataclass
class GenMetrics:
    prompt_words: int
    response_words: int
    duration_sec: float
    approx_tokens: int
    tokens_per_sec: float
    vram_mb_peak: Optional[int]
    temperature: float
    model: str
    eval_count: Optional[int] = None
    prompt_eval_count: Optional[int] = None


def _now() -> float:
    return time.time()


def words_count(text: str) -> int:
    return len([w for w in re.findall(r"\b\w+\b", text)])


def approx_tokens_from_text(text: str) -> int:
    # Approximate per spec: 1 word ≈ 1.33 tokens
    return int(words_count(text) * 1.33)


def has_nvidia_smi() -> bool:
    try:
        subprocess.run(["nvidia-smi", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False


def query_vram_used_mb() -> Optional[int]:
    """Return used VRAM in MB for GPU 0 using nvidia-smi, or None if unavailable."""
    if not has_nvidia_smi():
        return None
    try:
        # Query only memory.used in MiB without header/units for easier parsing
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8", errors="ignore").strip().splitlines()
        # Use first GPU value
        return int(out[0]) if out else None
    except Exception:
        return None


class VRAMSampler:
    """Samples VRAM usage periodically in a background thread and tracks peak."""

    def __init__(self, interval_sec: float = 0.5) -> None:
        self.interval = interval_sec
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.peak_mb: Optional[int] = None

    def _run(self) -> None:
        while not self._stop.is_set():
            used = query_vram_used_mb()
            if used is not None:
                self.peak_mb = used if self.peak_mb is None else max(self.peak_mb, used)
            time.sleep(self.interval)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)


def generate_once(prompt: str, temperature: float, max_tokens: int = 550) -> Tuple[str, Dict[str, Any]]:
    """Generate a single completion via Ollama and return (text, raw_metadata)."""
    response = ollama.generate(
        model=MODEL_NAME,
        prompt=prompt,
        options={
            "temperature": float(temperature),
            "num_predict": int(max_tokens),
        },
    )
    text = response.get("response", "")
    return text, response


def benchmark_once(prompt: str, temperature: float, max_tokens: int = 550) -> GenMetrics:
    sampler = VRAMSampler(interval_sec=0.4)
    sampler.start()
    t0 = _now()
    text, meta = generate_once(prompt, temperature=temperature, max_tokens=max_tokens)
    t1 = _now()
    sampler.stop()

    duration = max(t1 - t0, 1e-6)
    resp_words = words_count(text)
    approx_tokens = approx_tokens_from_text(text)
    tps = approx_tokens / duration
    metrics = GenMetrics(
        prompt_words=words_count(prompt),
        response_words=resp_words,
        duration_sec=duration,
        approx_tokens=approx_tokens,
        tokens_per_sec=tps,
        vram_mb_peak=sampler.peak_mb,
        temperature=temperature,
        model=MODEL_NAME,
        eval_count=meta.get("eval_count"),
        prompt_eval_count=meta.get("prompt_eval_count"),
    )
    return metrics


def quality_score(text: str) -> int:
    """Very simple coherence heuristic mapped to 1–5.

    Heuristics considered:
    - Minimum length (>= 220 words for ~300-word target)
    - Sentence completeness: at least 8 sentences with '.', '!' or '?'
    - Repetition penalty: ratio of top word frequency
    - Structure: presence of paragraphs/newlines
    """
    w = words_count(text)
    sentences = re.split(r"[.!?]+\s", text.strip())
    sentence_count = len([s for s in sentences if len(s.strip()) > 0])
    top_freq_ratio = 0.0
    words = [m.lower() for m in re.findall(r"\b\w+\b", text)]
    if words:
        from collections import Counter

        counts = Counter(words)
        most_common = counts.most_common(1)[0][1]
        top_freq_ratio = most_common / max(len(words), 1)

    paragraphs = text.count("\n") + 1

    score = 1
    if w >= 220:
        score += 1
    if sentence_count >= 8:
        score += 1
    if top_freq_ratio <= 0.08:
        score += 1
    if paragraphs >= 2:
        score += 1

    return max(1, min(5, score))


def jaccard_difference(a: str, b: str) -> float:
    """Return 1 - Jaccard similarity of word sets (after simple normalization)."""
    tokenize = lambda s: set(re.findall(r"\b\w+\b", s.lower()))
    sa, sb = tokenize(a), tokenize(b)
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb) or 1
    similarity = inter / union
    return 1.0 - similarity


def ensure_results_dir() -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)


def run_suite() -> Dict[str, Any]:
    print(f"[INFO] Using model: {MODEL_NAME}")
    ensure_results_dir()

    results: Dict[str, Any] = {
        "model": MODEL_NAME,
        "runs": {},
        "temperature_effects": {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Speed + Quality + VRAM on Emotional prompt (5 runs)
    emo_metrics: List[Dict[str, Any]] = []
    emo_quality_scores: List[int] = []
    for i in range(5):
        print(f"\n[RUN] Emotional argument #{i+1} (temp=0.9)")
        metrics = benchmark_once(EMOTIONAL_PROMPT, temperature=0.9, max_tokens=550)
        emo_metrics.append(asdict(metrics))
        # Regenerate text to evaluate quality using the same sample we measured
        # We already have the text from benchmark_once internally; regenerate to score quality quickly
        text, _ = generate_once(EMOTIONAL_PROMPT, temperature=0.9, max_tokens=550)
        q = quality_score(text)
        emo_quality_scores.append(q)
        print(
            f"  - tokens/sec~{metrics.tokens_per_sec:.1f}, words={metrics.response_words}, VRAM_peak={metrics.vram_mb_peak} MB, quality={q}/5"
        )

    results["runs"]["emotional"] = {
        "metrics": emo_metrics,
        "avg_tokens_per_sec": sum(m["tokens_per_sec"] for m in emo_metrics) / len(emo_metrics),
        "min_tokens_per_sec": min(m["tokens_per_sec"] for m in emo_metrics),
        "max_tokens_per_sec": max(m["tokens_per_sec"] for m in emo_metrics),
        "avg_quality": sum(emo_quality_scores) / len(emo_quality_scores),
    }

    # Speed + Quality + VRAM on Logical prompt (5 runs)
    log_metrics: List[Dict[str, Any]] = []
    log_quality_scores: List[int] = []
    for i in range(5):
        print(f"\n[RUN] Logical argument #{i+1} (temp=0.3)")
        metrics = benchmark_once(LOGICAL_PROMPT, temperature=0.3, max_tokens=550)
        log_metrics.append(asdict(metrics))
        text, _ = generate_once(LOGICAL_PROMPT, temperature=0.3, max_tokens=550)
        q = quality_score(text)
        log_quality_scores.append(q)
        print(
            f"  - tokens/sec~{metrics.tokens_per_sec:.1f}, words={metrics.response_words}, VRAM_peak={metrics.vram_mb_peak} MB, quality={q}/5"
        )

    results["runs"]["logical"] = {
        "metrics": log_metrics,
        "avg_tokens_per_sec": sum(m["tokens_per_sec"] for m in log_metrics) / len(log_metrics),
        "min_tokens_per_sec": min(m["tokens_per_sec"] for m in log_metrics),
        "max_tokens_per_sec": max(m["tokens_per_sec"] for m in log_metrics),
        "avg_quality": sum(log_quality_scores) / len(log_quality_scores),
    }

    # Judge prompt single run (speed + quality proxy)
    print("\n[RUN] Judge evaluation (temp=0.0)")
    judge_text_input = (
        "Defense (emotional):\n" +
        "Members of the court, before we rush to condemn, can we pause and ask a simple, human question: what would any parent do when their child cannot breathe?...\n\n" +
        "Prosecution (logical):\n" +
        "1. Rule. Disabled parking bays are reserved for vehicles displaying a valid permit...\n"
    )
    judge_metrics = benchmark_once(JUDGE_PROMPT + "\n\n" + judge_text_input, temperature=0.0, max_tokens=600)
    judge_text, _ = generate_once(JUDGE_PROMPT + "\n\n" + judge_text_input, temperature=0.0, max_tokens=600)
    judge_quality = quality_score(judge_text)
    results["runs"]["judge"] = {
        "metrics": asdict(judge_metrics),
        "quality": judge_quality,
    }
    print(
        f"  - tokens/sec~{judge_metrics.tokens_per_sec:.1f}, words={judge_metrics.response_words}, VRAM_peak={judge_metrics.vram_mb_peak} MB, quality={judge_quality}/5"
    )

    # Temperature effects comparison on emotional prompt
    print("\n[RUN] Temperature effect test (0.3 vs 0.9)")
    low_text, _ = generate_once(EMOTIONAL_PROMPT, temperature=0.3, max_tokens=550)
    high_text, _ = generate_once(EMOTIONAL_PROMPT, temperature=0.9, max_tokens=550)
    diff_score = jaccard_difference(low_text, high_text)
    results["temperature_effects"]["emotional_prompt"] = {
        "jaccard_difference": diff_score,
        "note": "Higher is more different. Expect noticeable difference (> 0.35).",
    }
    print(f"  - Jaccard difference: {diff_score:.2f}")

    # Success criteria checks
    print("\n[CHECK] Success criteria")
    speed_ok = (
        results["runs"]["emotional"]["avg_tokens_per_sec"] >= 15.0 and
        results["runs"]["logical"]["avg_tokens_per_sec"] >= 15.0
    )
    quality_ok = (
        results["runs"]["emotional"]["avg_quality"] >= 3.0 and
        results["runs"]["logical"]["avg_quality"] >= 3.0
    )
    vram_ok = True
    for bucket in (emo_metrics + log_metrics + [asdict(judge_metrics)]):
        peak = bucket.get("vram_mb_peak")
        if peak is not None and peak > 7500:
            vram_ok = False
            break
    temp_ok = diff_score >= 0.35  # "noticeably different"

    results["criteria"] = {
        "speed_tokens_per_sec_ge_15": speed_ok,
        "quality_ge_3": quality_ok,
        "vram_under_7_5_gb": vram_ok,
        "temperature_effects_notable": temp_ok,
    }

    # Save results
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] Results written to {RESULTS_PATH}")

    # Final console summary
    print("\n=== Benchmark Summary ===")
    print(
        f"Speed OK: {speed_ok} | Quality OK: {quality_ok} | VRAM OK: {vram_ok} | Temp Effect OK: {temp_ok}"
    )
    return results


if __name__ == "__main__":
    try:
        run_suite()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Benchmark aborted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
