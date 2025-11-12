/* AI Judge Frontend Logic (Task 7.4)
 * - Creates cases, starts debates, polls status, renders arguments/verdict
 * - Updates statistics; handles errors and loading state
 */

(() => {
  'use strict';

  const $ = (sel) => document.querySelector(sel);
  const byId = (id) => document.getElementById(id);

  // UI elements
  const el = {
    title: byId('case-title'),
    description: byId('case-description'),
    exampleSelect: byId('example-case-select'),
    startBtn: byId('start-debate-btn'),
    loadBtn: byId('load-example-btn'),
    error: byId('error-message'),
    loading: byId('loading'),
    loadingText: byId('loading-text'),
    progressBar: byId('progress-bar'),
    descCounter: byId('case-counter'),
    // Emotional bubbles
    e1: byId('emotional-r1'),
    e2: byId('emotional-r2'),
    e3: byId('emotional-r3'),
    // Logical bubbles
    l1: byId('logical-r1'),
    l2: byId('logical-r2'),
    l3: byId('logical-r3'),
    // Judge
    scoreE: byId('score-emotional'),
    scoreL: byId('score-logical'),
    winner: byId('winner'),
    reasoning: byId('judge-reasoning'),
    // Stats
    statTotal: byId('stat-total'),
    statE: byId('stat-emotional'),
    statL: byId('stat-logical'),
    statEpct: byId('stat-emotional-pct'),
    statLpct: byId('stat-logical-pct'),
  };

  // Sample cases (aligned with Decision Log #10)
  const EXAMPLES = [
    {
      title: 'Neighbor Noise After 10 PM',
      description:
        "My neighbor plays loud music every night until 2 AM despite multiple complaints. They claim 'their apartment, their rules.' I want them to stop after 10 PM.",
    },
    {
      title: 'Freelancer Payment Dispute',
      description:
        'I delivered a website to a client who refuses to pay the remaining 40% claiming minor bugs. I provided fixes within 48 hours as per contract.',
    },
    {
      title: 'Defective Product Refund',
      description:
        'I bought a vacuum cleaner that stopped working after a week. The store refuses a refund, offering only store credit despite my receipt.',
    },
    {
      title: 'Parking Ticket Appeal',
      description:
        'I received a parking ticket while stopping for 3 minutes to pick up my elderly parent with mobility issues. I displayed hazard lights.',
    },
    {
      title: 'Shared Facility Cleaning',
      description:
        'The shared laundry room is often left dirty by a specific neighbor. The strata rules require users to clean up after use, but they ignore warnings.',
    },
    {
      title: 'Medical Emergency Parking (Defense‑Favored)',
      description:
        'A parent briefly stopped (≈2 minutes) in a disabled bay outside a pharmacy with hazard lights on to urgently pick up a child\'s life‑saving asthma medication during an acute episode. Pharmacy staff corroborate the timeline; security footage confirms no disabled permit holder attempted to use the space during that interval. A clinician note documents the child\'s condition and timing. Local ordinance recognizes a necessity defense for imminent threats to health and safety when no reasonable lawful alternative exists. The ticket cites \"no permit displayed\" but omits these mitigating facts.',
    },
  ];

  let pollTimer = null;
  let currentDebateId = null;
  let pollErrorStreak = 0;

  function setError(msg) {
    if (!el.error) return;
    if (msg) {
      el.error.textContent = msg;
      el.error.hidden = false;
    } else {
      el.error.textContent = '';
      el.error.hidden = true;
    }
  }

  function autoResizeTextarea() {
    if (!el.description) return;
    el.description.style.height = 'auto';
    el.description.style.height = Math.min(1000, el.description.scrollHeight) + 'px';
  }

  function updateDescCounter() {
    if (!el.description || !el.descCounter) return;
    const text = el.description.value || '';
    const chars = text.length;
    const words = (text.trim().match(/\S+/g) || []).length;
    el.descCounter.textContent = `Words: ${words} | Characters: ${chars}`;
  }

  function setLoading(active, text = 'Working...', progress = null) {
    if (!el.loading) return;
    if (active) {
      el.loading.removeAttribute('hidden');
      el.loading.setAttribute('aria-busy', 'true');
    } else {
      el.loading.setAttribute('hidden', '');
      el.loading.setAttribute('aria-busy', 'false');
    }
    if (el.loadingText) el.loadingText.textContent = text;
    if (el.progressBar && typeof progress === 'number') {
      el.progressBar.style.width = Math.max(0, Math.min(100, progress)) + '%';
    }
  }

  function resetArena() {
    [el.e1, el.e2, el.e3, el.l1, el.l2, el.l3].forEach((node) => {
      if (node) node.textContent = '';
    });
    if (el.scoreE) el.scoreE.textContent = '—';
    if (el.scoreL) el.scoreL.textContent = '—';
    if (el.winner) el.winner.textContent = '—';
    if (el.reasoning) el.reasoning.textContent = '';
    setError('');
  }

  function displayArgument(lawyer, round, text) {
    const id = `${lawyer}-${'r' + round}`.replace('emotional-r', 'emotional-r').replace('logical-r', 'logical-r');
    // Map to actual DOM ids
    const map = {
      'emotional-r1': el.e1,
      'emotional-r2': el.e2,
      'emotional-r3': el.e3,
      'logical-r1': el.l1,
      'logical-r2': el.l2,
      'logical-r3': el.l3,
    };
    const target = map[id];
    if (target) target.textContent = text || '';
  }

  function displayVerdict(verdict) {
    if (!verdict) return;
    const e = parseInt(verdict.emotional_score ?? 0, 10);
    const l = parseInt(verdict.logical_score ?? 0, 10);
    if (el.scoreE) el.scoreE.textContent = isFinite(e) ? String(e) : '—';
    if (el.scoreL) el.scoreL.textContent = isFinite(l) ? String(l) : '—';
    if (el.winner) el.winner.textContent = verdict.winner ?? '—';
    if (el.reasoning) el.reasoning.textContent = (verdict.reasoning || '').trim();
  }

  async function updateStatistics() {
    try {
      const res = await fetch('/api/statistics');
      if (!res.ok) throw new Error('Failed to fetch statistics');
      const data = await res.json();
      const total = Number(data.total_debates || 0);
      const ew = Number(data.emotional_wins || 0);
      const lw = Number(data.logical_wins || 0);
      if (el.statTotal) el.statTotal.textContent = String(total);
      if (el.statE) el.statE.textContent = String(ew);
      if (el.statL) el.statL.textContent = String(lw);
      const ep = total > 0 ? Math.round((ew / total) * 100) : 0;
      const lp = total > 0 ? Math.round((lw / total) * 100) : 0;
      if (el.statEpct) el.statEpct.textContent = ep + '%';
      if (el.statLpct) el.statLpct.textContent = lp + '%';
    } catch (err) {
      // Silently ignore in UI, but log for devs
      console.error(err);
    }
  }

  async function createCase(title, description) {
    const res = await fetch('/api/case/create', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title, description }),
    });
    if (!res.ok) throw new Error('Failed to create case');
    return res.json();
  }

  async function startDebate() {
    try {
      setError('');
      resetArena();

      const title = (el.title?.value || '').trim();
      const description = (el.description?.value || '').trim();
      if (!title || !description) {
        setError('Please provide both a case title and description.');
        return;
      }

      setLoading(true, 'Creating case...', 5);
      const { case_id } = await createCase(title, description);
      if (!case_id) throw new Error('No case_id returned');

      setLoading(true, 'Starting debate...', 10);
      const res = await fetch(`/api/debate/start/${encodeURIComponent(case_id)}`, { method: 'POST' });
      if (!res.ok) throw new Error('Failed to start debate');
      const { debate_id } = await res.json();
      if (!debate_id) throw new Error('No debate_id returned');

      currentDebateId = debate_id;
      setLoading(true, 'Generating Round 1...', 20);
      if (pollTimer) clearInterval(pollTimer);
      await pollDebateStatus(debate_id); // first immediate poll
      pollTimer = setInterval(() => pollDebateStatus(debate_id), 2000);
    } catch (err) {
      console.error(err);
      setError(err?.message || 'An unexpected error occurred.');
      setLoading(false, 'Ready.', 0);
    }
  }

  async function pollDebateStatus(debateId) {
    try {
      const res = await fetch(`/api/debate/${encodeURIComponent(debateId)}`);
      if (!res.ok) throw new Error('Failed to fetch debate status');
      const data = await res.json();
      pollErrorStreak = 0;

      // Rounds may be in-progress (empty arrays)
      const rounds = Array.isArray(data.rounds) ? data.rounds : [[], [], []];
      const status = data.status || 'in_progress';

      // Update arguments if present; each round has up to 2 items with lawyer/content
      rounds.forEach((rnd, idx) => {
        const roundNo = idx + 1;
        if (!Array.isArray(rnd)) return;
        rnd.forEach((arg) => {
          const lawyer = (arg.lawyer || '').toLowerCase();
          const content = arg.content || '';
          if (lawyer === 'emotional' || lawyer === 'logical') {
            displayArgument(lawyer, roundNo, content);
          }
        });
      });

      // Loading/progress hints
      const completedRounds = rounds.filter((r) => Array.isArray(r) && r.length === 2).length;
      if (status === 'in_progress') {
        const labels = ['Generating Round 1...', 'Generating Round 2...', 'Generating Round 3...', 'Evaluating verdict...'];
        const progressSteps = [20, 50, 75, 90];
        const idx = Math.min(completedRounds, labels.length - 1);
        setLoading(true, labels[idx], progressSteps[idx]);
      }

      // Verdict
      if (data.verdict) {
        displayVerdict(data.verdict);
      }

      if (status === 'complete') {
        setLoading(false, 'Complete', 100);
        if (pollTimer) {
          clearInterval(pollTimer);
          pollTimer = null;
        }
        // Refresh statistics after completion
        updateStatistics();
      } else if (status === 'failed') {
        // Stop polling and surface a clear, single failure message
        setLoading(false, 'Debate failed', 0);
        if (pollTimer) {
          clearInterval(pollTimer);
          pollTimer = null;
        }
        const reason = (data.verdict && data.verdict.reasoning) ? String(data.verdict.reasoning) : 'Unknown error.';
        setError('Debate failed. ' + reason);
      }
    } catch (err) {
      // Keep polling silently to avoid noisy UI; show a gentle hint after a few misses
      console.warn('Poll error:', err);
      pollErrorStreak += 1;
      if (pollErrorStreak >= 3) {
        setLoading(true, 'Waiting for server...', 0);
        // Do not spam user with errors
      }
    }
  }

  function loadExampleCase(caseIndex) {
    const idx = Number(caseIndex);
    if (!Number.isInteger(idx) || idx < 0 || idx >= EXAMPLES.length) return;
    const ex = EXAMPLES[idx];
    if (el.title) el.title.value = ex.title;
    if (el.description) el.description.value = ex.description;
    setError('');
  }

  // Event listeners
  if (el.startBtn) el.startBtn.addEventListener('click', startDebate);
  if (el.exampleSelect) el.exampleSelect.addEventListener('change', (e) => {
    const v = e.target?.value ?? '';
    // Do not auto-load to avoid clobbering typed input; only update selection state
  });
  if (el.loadBtn) el.loadBtn.addEventListener('click', () => {
    loadExampleCase(el.exampleSelect?.value ?? '');
    updateDescCounter();
    autoResizeTextarea();
  });

  // Initial state
  setLoading(false, 'Ready.', 0);
  updateStatistics();
  updateDescCounter();
  autoResizeTextarea();
  if (el.description) {
    el.description.addEventListener('input', () => {
      updateDescCounter();
      autoResizeTextarea();
    });
  }

  // Expose for inline HTML handler compatibility
  window.loadExampleCase = loadExampleCase;
  window.startDebate = startDebate;
})();
