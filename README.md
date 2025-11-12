AI Judge – Multi‑Agent Legal Debate Platform

Summary

Two AI lawyers (Emotional vs Logical) debate Tier‑1 civil disputes across 3 rounds.
An AI Judge evaluates both sides and returns a JSON verdict with scores, winner, reasoning, and rubric breakdown.
Everything runs locally via Ollama (Llama 3 8B). Backend: FastAPI. Frontend: Vanilla HTML/CSS/JS.
Key Decisions (Design Contract)

3 Agents Only: Emotional Lawyer, Logical Lawyer, Judge. No shared memory beyond the case context.
Turn‑Based (Sequential): Round 1 openings → Round 2 counters → Round 3 rebuttals.
Word Limits: 250–350 words per lawyer per round; Judge reasoning 300–400 words.
Personalities:
Emotional (defense): emotional vocabulary >10%, personal pronouns >12%, 2–4 rhetorical questions, 1–3 exclamations, narrative style.
Logical (prosecution): 4–6 structural markers, 2–3 if‑then statements, >8% evidence words, emotional words <3%, 0–1 exclamation.
Temperatures: Emotional 0.8, Logical 0.25, Judge 0.0 (deterministic).
Scope: Tier‑1 civil disputes (neighbor noise/boundaries, small contracts, consumer rights, minor traffic/public).
Model: Llama 3 8B via Ollama; contingency to switch to paid API if speed/quality insufficient.
Features

3‑round debate with personality‑consistent agents and word‑limit enforcement.
Judge verdict in strict JSON (scores, winner, reasoning, and rubric criteria).
Live UI that streams round results and shows final scores and statistics.
Robust back end with retries, timeouts, VRAM sampling (in benchmark), and local JSON persistence.
Prerequisites

Python 3.10+
Ollama installed and running
Model pulled: ollama pull llama3:8b
Recommended GPU: 8GB VRAM (RTX 3050 works). CPU works but is slow.
Installation

Install dependencies: pip install -r ai-judge/requirements.txt
Run the App

Start backend (recommended port 8010):
cd ai-judge
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8010
Open the UI:
http://127.0.0.1:8010/
Using the UI

Case Input: title + description (no hard limit; longer texts increase latency). Live word/character counter.
Examples: includes a defense‑favored “Medical Emergency Parking” case to test Emotional wins.
Start Debate: rounds populate for Emotional 🔥 and Logical 🧊; Judge ⚖️ displays scores and reasoning.
Statistics: total debates + per‑side wins and win percentages.
API

POST /api/case/create
Body: { "title": str, "description": str }
Response: { "case_id": str }
POST /api/debate/start/{case_id}
Response: { "debate_id": str, "status": "in_progress" }
GET /api/debate/{debate_id}
Response: DebateTranscript (status: in_progress | complete | failed).
GET /api/statistics
Response: { "emotional_wins": int, "logical_wins": int, "total_debates": int }
Architecture

Agents (backend/agents)
EmotionalLawyer (defense) and LogicalLawyer (prosecution) generate opening, counter, and rebuttal.
JudgeAgent evaluates the transcript and returns a JSON verdict.
Debate Orchestration (backend/utils/debate_coordinator.py)
Runs the 3 rounds in order, passes opponent context, and calls the Judge.
Handles retries and timing; returns a DebateTranscript (Pydantic).
Ollama Client (backend/utils/ollama_client.py)
Streamed generation with timeout, retries, and logging; JSON mode supported via extra_options.
Persistence (backend/data/*.json)
debates.json: stores cases, debate status, rounds (per‑round arguments), and verdicts.
statistics.json: cumulative wins and totals.
Configuration (backend/config/settings.py)

MODEL_NAME = "llama3:8b"
TEMPERATURES = { "emotional": 0.8, "logical": 0.25, "judge": 0.0 }
Word limits: WORD_LIMIT_MIN = 250, WORD_LIMIT_MAX = 350, Judge 300–400
Tokens: MAX_TOKENS_ARGUMENT = 470, MAX_TOKENS_VERDICT = 700
Files: DEBATES_FILE, STATISTICS_FILE
Prompts (backend/config/prompts.py)

Emotional (defense) and Logical (prosecution) include explicit role and measurable traits.
Judge includes 5‑criterion rubric (relevance, coherence, evidence, persuasiveness, rebuttal) and strict JSON rules.
Data Model (backend/models/schemas.py)

CaseInput { title, description }
Argument { lawyer: "emotional"|"logical", round_number: 1–3, content, word_count }
Verdict { emotional_score, logical_score, winner, reasoning, criteria_scores }
DebateTranscript { debate_id, case, rounds[3][2], verdict, timestamp, status }
Testing & Benchmarking

Benchmark: python tests/benchmark_ollama.py
Measures tokens/sec, quality proxy, VRAM, temperature effect. Outputs to results/benchmark_results.json.
Unit tests: python -m unittest ai-judge/tests/test_agents.py -v
Validates word limits, temperature variance, judge verdict structure.
Flow test: python -m unittest ai-judge/tests/test_debate_flow.py -v
Runs two debates end‑to‑end, validates cross‑references and timing. Outputs to ai-judge/results/debate_flow_test_results.json.
Performance Notes

Expect ≈ 10–13 tokens/sec on mid‑range hardware for 8B model.
VRAM usage ≈ 3.5–4 GB observed.
If speed is a concern, consider smaller models or (later) enabling a provider toggle to a hosted API.
Troubleshooting

404 on /: Root now redirects to /static/index.html. Use http://127.0.0.1:8010/.
Static path errors: app mounts the absolute frontend dir; run server from ai-judge/.
“Failed to fetch debate progress”: debate failed or server unavailable.
The UI stops and shows the reason (status=failed). Start a new debate.
Word count validation errors: Agents pad/trim to 250–350 before saving. If you see errors, restart and try again.
“Model did not return strict JSON”: Judge uses JSON mode, repair, and a reformat pass; verdict still returns.
Roadmap

Optional provider toggle (local Ollama ↔ hosted API) guarded by settings/env.
Streaming tokens into the UI per round.
Persistent debates list with view/replay in the UI.
Project Structure (key files)

ai-judge/
├── backend/
│   ├── app.py
│   ├── config/
│   │   ├── prompts.py
│   │   └── settings.py
│   ├── agents/
│   │   ├── emotional_lawyer.py
│   │   ├── logical_lawyer.py
│   │   └── judge.py
│   ├── utils/
│   │   ├── ollama_client.py
│   │   └── debate_coordinator.py
│   ├── models/
│   │   └── schemas.py
│   └── data/
│       ├── debates.json
│       └── statistics.json
├── frontend/
│   ├── index.html
│   ├── css/styles.css
│   └── js/app.js
├── tests/
│   ├── benchmark_ollama.py
│   └── test_debate_flow.py
└── results/
    └── benchmark_results.json
