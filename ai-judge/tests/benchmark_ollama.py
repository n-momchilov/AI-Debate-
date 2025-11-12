"""
Wrapper that delegates to the root benchmark if present, otherwise a minimal
standalone benchmark. Preferred location for Week 2 is ai-judge/tests.
"""
from __future__ import annotations

import importlib.util
import os
import runpy
import sys
from pathlib import Path


HERE = Path(__file__).parent
ROOT_BENCH = Path.cwd().parent / "tests" / "benchmark_ollama.py"

if ROOT_BENCH.exists():
    # Execute the root script to keep a single source of truth
    spec = importlib.util.spec_from_file_location("benchmark_ollama", str(ROOT_BENCH))
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules["benchmark_ollama"] = module
        spec.loader.exec_module(module)
    else:
        runpy.run_path(str(ROOT_BENCH), run_name="__main__")
else:
    # Fallback: Simple smoke test
    print("[WARN] Root benchmark not found; simple smoke test running.")
    try:
        import ollama  # type: ignore
        resp = ollama.generate(model=os.environ.get("OLLAMA_MODEL", "llama3:8b"), prompt="Hello from AI Judge!")
        print("[OK] Response length:", len(resp.get("response", "")))
    except Exception as e:
        print("[ERROR]", e)

