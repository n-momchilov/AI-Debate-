AI Judge – Week 2 End‑to‑End Test Checklist

Prerequisites
- Python 3.10+
- Ollama installed and running; model `llama3:8b` pulled
- Dependencies installed: `pip install -r ai-judge/requirements.txt`

Start Backend
- From `ai-judge/` run: `uvicorn backend.app:app --reload`
- Confirm logs show request middleware lines when you make calls

Open Frontend
- Visit: `http://localhost:8000/static/index.html`
- Page shows header, Case Input panel, Debate Arena, Statistics sidebar, Loading indicator

Run a Debate (Procedure)
1) Load example: choose “Neighbor Noise After 10 PM” → click “Load”
2) Click “Start Debate”
3) Observe loading text progress: Generating Round 1 → Round 2 → Round 3 → Evaluating verdict
4) Round outputs should appear per column:
   - Emotional 🔥: `#emotional-r1`, `#emotional-r2`, `#emotional-r3`
   - Logical 🧊: `#logical-r1`, `#logical-r2`, `#logical-r3`
   - Judge ⚖️: scores, winner, reasoning
5) Duration target: ~2–3 minutes depending on hardware

Quick API Smoke (optional)
- Create case:
  - `curl -X POST http://localhost:8000/api/case/create -H "Content-Type: application/json" -d "{\"title\":\"Noise\",\"description\":\"Neighbor plays loud music after 10 PM...\"}"`
- Start debate:
  - `curl -X POST http://localhost:8000/api/debate/start/<CASE_ID>`
- Poll debate:
  - `curl http://localhost:8000/api/debate/<DEBATE_ID>`
- Stats:
  - `curl http://localhost:8000/api/statistics`

Expected Results
- All three rounds render in Emotional and Logical columns (250–350 words each, typical)
- Judge shows emotional/logical scores, winner, and reasoning (300–400 words typical)
- Statistics update after completion (totals and win counts)
- UI remains responsive; no blocking or flicker; loading indicator updates

Pass/Fail Criteria
- Pass if all below are true:
  - No errors in browser console or server logs
  - All arguments display correctly in their rounds/columns
  - Verdict populated with scores and winner
  - Statistics reflect the new debate (totals increment)
  - End‑to‑end time ≤ 5 minutes
- Note: If verdict JSON fails to parse, system returns a safe fallback verdict; investigate prompt/template if frequent

Resource/Performance Hints
- GPU VRAM usage should remain < 7.5 GB with `llama3:8b`
- Throughput typically ~10–14 tokens/sec on mid‑range GPUs; <15 tok/sec may feel slower

Troubleshooting
- Frontend loads but API fails: verify server at `http://localhost:8000`, CORS enabled, and static mount path `/static`
- 500 errors in API: check `backend/data/*.json` integrity; the server auto‑repairs minimal structure
- Long generation times: ensure no other heavy GPU jobs; consider lowering load or switching to smaller models if needed
- No Ollama connection: confirm `ollama list` and that the service is running; pull `llama3:8b` if missing

Week 2 Validation Checkboxes
- Functional Tests:
  - [ ] Ollama benchmark passed (speed ≥ 15 tok/sec, quality ≥ 3/5)
  - [ ] All 3 agents generate arguments successfully
  - [ ] Full 3‑round debate completes without errors
  - [ ] UI displays debate correctly
  - [ ] Backend API endpoints all functional
  - [ ] System works offline (after model download)
- Quality Tests:
  - [ ] Arguments within word limits (250–350)
  - [ ] Temperature effects visible (emotional > logical variance)
  - [ ] Judge verdict includes reasoning and scores
  - [ ] No crashes/exceptions in 5 consecutive debates
- Integration Tests:
  - [ ] Frontend → Backend → Ollama → Frontend loop works
  - [ ] Error messages display properly (empty input, API failure)
  - [ ] Statistics tracking increments and persists

If Any Test Fails
- Capture server logs and browser console output
- Re‑run with a shorter case and watch progress labels
- For repeated judge JSON parse failures: tighten the judge prompt formatting or post‑process response with stricter extraction

