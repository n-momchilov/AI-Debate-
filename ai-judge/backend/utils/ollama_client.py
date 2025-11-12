"""Robust Ollama API client wrapper.

Implements:
- Streamed generation with a hard timeout (120s)
- 3-attempt retry with exponential backoff
- Logging of prompt/response length and latency
- Approximate token counting (1 word ≈ 1.33 tokens)
- Error handling for common failure modes

Usage:
    from backend.utils.ollama_client import OllamaClient
    client = OllamaClient("llama3:8b")
    text = client.generate(prompt, system_prompt, temperature=0.25, max_tokens=470)
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, Optional

import ollama  # type: ignore

try:  # Best-effort import for precise exception types
    import httpx
except Exception:  # pragma: no cover - fallback when httpx not importable from env
    httpx = None  # type: ignore


logger = logging.getLogger("ai_judge.ollama_client")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class OllamaClient:
    """Thin client over the `ollama` Python SDK with retries and safeguards.

    Parameters:
        model_name: Name of the local model in Ollama (e.g., "llama3:8b").
    """

    def __init__(self, model_name: str = "llama3:8b") -> None:
        self.model_name = model_name
        self._timeout_seconds = 120.0
        self._max_attempts = 3

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate text from the model with retries and a hard timeout.

        Args:
            prompt: The user/content prompt to send to the model.
            system_prompt: The system instruction (personality/role) to condition the model.
            temperature: Sampling temperature (creativity).
            max_tokens: Approximate maximum tokens to generate (`num_predict`).

        Returns:
            The model's generated text.

        Raises:
            TimeoutError: If generation exceeds the configured timeout (120 seconds).
            ConnectionError: If the Ollama server is not reachable.
            FileNotFoundError: If the requested model is not available.
            MemoryError: If a VRAM/Out-of-memory condition is detected.
            RuntimeError: For malformed or otherwise unclassified errors.
        """

        attempts = 0
        last_error: Optional[Exception] = None
        while attempts < self._max_attempts:
            attempts += 1
            start = time.time()
            try:
                text = self._stream_generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_options=extra_options,
                    start_time=start,
                )

                # Validate and log
                if not self._validate_response(text):
                    raise RuntimeError("Malformed or empty response from model")

                elapsed = max(time.time() - start, 1e-6)
                tokens = self._approx_tokens(text)
                logger.info(
                    "Ollama call ok | model=%s | prompt_words=%d | response_words=%d | tokens≈%d | time=%.2fs | tps≈%.1f",
                    self.model_name,
                    self._word_count(prompt),
                    self._word_count(text),
                    tokens,
                    elapsed,
                    tokens / elapsed,
                )
                return text

            except TimeoutError:
                logger.error("Ollama call timed out after %.0fs (attempt %d)", self._timeout_seconds, attempts)
                last_error = TimeoutError(f"Generation exceeded {self._timeout_seconds:.0f}s")
            except Exception as e:  # Map and log specific error types
                mapped = self._map_exception(e)
                logger.error("Ollama call failed (attempt %d/%d): %s", attempts, self._max_attempts, mapped)
                last_error = mapped

            # Backoff before retrying unless max attempts reached
            if attempts < self._max_attempts:
                sleep_s = 1.5 * attempts
                time.sleep(sleep_s)

        # Exhausted all attempts
        assert last_error is not None
        raise last_error

    # Internal helpers
    def _stream_generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        extra_options: Optional[Dict[str, Any]],
        start_time: float,
    ) -> str:
        """Stream tokens while enforcing a hard timeout by wall clock."""
        text_parts: list[str] = []

        # Use streamed generation to monitor elapsed time
        opts: Dict[str, Any] = {
            "temperature": float(temperature),
            "num_predict": int(max_tokens),
        }
        if extra_options:
            try:
                opts.update(dict(extra_options))
            except Exception:
                # Ignore malformed extra options to avoid crashing
                pass
        stream = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            system=system_prompt or None,
            options=opts,
            stream=True,
        )

        for chunk in stream:
            # Timeout check
            if (time.time() - start_time) > self._timeout_seconds:
                raise TimeoutError("Ollama generation timeout")

            piece = chunk.get("response")
            if piece:
                text_parts.append(piece)

            if chunk.get("done"):
                break

        return "".join(text_parts).strip()

    def _validate_response(self, response: str) -> bool:
        """Basic response validation."""
        if not isinstance(response, str):
            return False
        if len(response.strip()) == 0:
            return False
        # Heuristic: avoid trivially short/garbled outputs
        if self._word_count(response) < 10:
            return False
        return True

    @staticmethod
    def _word_count(text: str) -> int:
        return len([w for w in text.split() if w.strip()])

    @staticmethod
    def _approx_tokens(text: str) -> int:
        # 1 word ≈ 1.33 tokens
        return int(math.ceil(len([w for w in text.split() if w.strip()]) * 1.33))

    @staticmethod
    def _map_exception(exc: Exception) -> Exception:
        """Map raw exceptions to clearer categories per requirements."""
        msg = str(exc).lower()

        # Connection refused / server not running
        if httpx is not None and isinstance(
            exc, (getattr(httpx, "ConnectError", tuple()), getattr(httpx, "ReadTimeout", tuple()))
        ):
            return ConnectionError("Cannot reach Ollama server. Is it running?")
        if "connection refused" in msg or "failed to establish a new connection" in msg:
            return ConnectionError("Cannot reach Ollama server. Is it running?")

        # Model not found / 404
        if httpx is not None and isinstance(exc, getattr(httpx, "HTTPStatusError", tuple())):
            try:
                status = exc.response.status_code  # type: ignore[attr-defined]
            except Exception:
                status = None
            if status == 404:
                return FileNotFoundError("Model not found in Ollama: ensure it's pulled")
        if "model not found" in msg or "no such model" in msg:
            return FileNotFoundError("Model not found in Ollama: ensure it's pulled")

        # VRAM / memory
        if "out of memory" in msg or "cuda" in msg and "memory" in msg or "not enough vram" in msg:
            return MemoryError("VRAM exhausted during generation. Try a smaller model or reduce max tokens.")

        # Timeouts
        if "timeout" in msg:
            return TimeoutError("Ollama request timed out")

        # Fallback
        return RuntimeError(str(exc))
