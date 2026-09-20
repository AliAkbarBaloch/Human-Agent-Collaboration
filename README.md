# HALO — Human-Agent Loop Orchestrator

**Human-Agent Collaboration Platform for Supervised Multi-Agent Task Execution**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](#requirements)
[![Node 18+](https://img.shields.io/badge/node-18%2B-brightgreen)](#frontend-development)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Built on AutoGen](https://img.shields.io/badge/built%20on-AutoGen-orange)](https://github.com/microsoft/autogen)

> Plan together. Execute together. Trust together.

Based on the architecture of [Magentic-UI](https://github.com/microsoft/magentic-ui) (Mozannar et al., Microsoft Research 2025),
extended with three novel research contributions — together, the **HALO Adaptive Oversight Framework**. Built and
evaluated as a research seminar project at the University of Passau, Summer Semester 2026; the full write-up,
including a 7-framework survey and a live 22-scenario evaluation, is in [`Final_Report.pdf`](Final_Report.pdf).

## Results at a glance

| | |
|---|---|
| Live evaluation | 22/22 black-box scenarios passed against the running application (web browsing, code execution, file access, MCP, trust feedback) |
| Injection detection | Precision 1.00, Recall 0.958, F1 0.979 over 48 trials on a labeled ground-truth corpus |
| Trust convergence | Proven bound $\sigma_n = O(1/\sqrt{n})$; reaches $\sigma < 0.05$ in about 61–69 interactions in practice |
| Framework survey | 7 multi-agent frameworks rated against a 4-criterion catalogue; none implement risk-proportional approval or content-level injection scanning |
| Bugs found | 2 real defects surfaced by live testing (FileSurfer prompt ordering, MCP tool-name escaping), both fixed |

## Screenshots

<table>
<tr>
<td width="25%"><img src="Results/QA_WebSurfer_ResearchLowRisk.png" width="100%"><br><sub>Gap 1 — low-risk task auto-approved</sub></td>
<td width="25%"><img src="Results/QA_WebSurfer_DestructiveApprovalRequired.png" width="100%"><br><sub>Gap 1 — destructive task forces approval</sub></td>
<td width="25%"><img src="Results/QA_WebSurfer_PatternInjectionAlert.png" width="100%"><br><sub>Gap 2 — lexical injection alert</sub></td>
<td width="25%"><img src="Results/QA_WebSurfer_SemanticInjectionAlert.png" width="100%"><br><sub>Gap 2 — semantic-only injection caught</sub></td>
</tr>
<tr>
<td width="25%"><img src="Results/QA_Trust_PanelDetail.png" width="100%"><br><sub>Gap 3 — Bayesian trust panel</sub></td>
<td width="25%"><img src="Results/QA_Coder_ExceptionDebugLoopResult.png" width="100%"><br><sub>Coder agent self-correcting after an exception</sub></td>
<td width="25%"><img src="Results/QA_MCP_SessionResult.png" width="100%"><br><sub>MCP tool call executed end to end</sub></td>
<td width="25%"><img src="Results/Settings_UI.png" width="100%"><br><sub>Settings panel</sub></td>
</tr>
</table>

All 19 evaluation screenshots are in [`Results/`](Results/). The test matrix behind them is in
[`Final_Report.pdf`](Final_Report.pdf).

## Table of Contents

- [What is HALO?](#what-is-halo)
- [What I Built vs. Upstream Magentic-UI](#what-i-built-vs-upstream-magentic-ui)
- [Requirements](#requirements)
- [Installation](#installation)
- [Run](#run)
- [Configuration](#configuration)
- [Frontend Development](#frontend-development)
- [Architecture](#architecture)
- [Research Gaps](#research-gaps)
- [Testing & Evaluation](#testing--evaluation)
- [Report & Further Reading](#report--further-reading)
- [License](#license)
- [Author](#author)

The three research contributions this fork adds — together, the **HALO Adaptive Oversight Framework**:

- **Gap 1 — Adaptive Action Guard**: Classifies each task as research / transactional / destructive (rule-based, optionally fused with an LLM layer) and adjusts the approval policy accordingly — less friction on safe tasks, full scrutiny on risky ones.
- **Gap 2 — Prompt Injection Visibility Layer (Injection Gateway)**: Real-time detection of prompt injection across every content source an agent reads — web pages, files, code-execution output, and MCP tool results — through one shared gateway, plus **action-hijack screening** that catches an agent's proposed action being redirected by something it previously read. User-controlled allow/block decisions are surfaced in the UI.
- **Gap 3 — Bayesian Trust Calibration**: A per-user, per-task-type trust score (Beta-distribution model) built from approve/reject/block history, closing the loop by default — trust actually tightens or loosens future approval strictness, persists across sessions, and can never loosen a destructive task below full approval.

---

## What is HALO?

HALO is a human-in-the-loop web agent system. It coordinates multiple AI sub-agents (browser, coder, file surfer) under a human-supervised orchestrator. Every significant action requires or offers human approval. The user can:

- **Co-Plan**: Review, edit, or regenerate the execution plan before any step runs.
- **Co-Task**: Send mid-task corrections that cause the orchestrator to replan.
- **Approve Actions**: Approve or reject individual agent actions in real time.
- **Switch Tasks**: Manage multiple concurrent tasks from the sidebar.
- **Recall Plans**: Load similar past plans from memory to reuse as starting points.
- **Follow Up**: Ask follow-up questions after seeing the final answer, continuing the same session.

---

## What I Built vs. Upstream Magentic-UI

The orchestrator, the web/coder/file-surfer agents, and the WebSocket/UI plumbing are Microsoft's Magentic-UI —
credited in [LICENSE](LICENSE) and left as-is. Everything under **Gap 1–3** below is original work for this
project, layered on top without modifying that core:

| Layer | New files | What it does |
|---|---|---|
| Gap 1 — Adaptive Action Guard | `task_classifier.py`, `llm_risk_estimator.py` | Classifies every task fresh (research / transactional / destructive) and rewrites the live approval policy before agents act |
| Gap 2 — Injection Gateway | `injection_gateway.py`, `injection_scanner.py`, `semantic_injection_detector.py` | One shared scan-and-gate function every content source (pages, files, code output, MCP results, task text) routes through, plus action-hijack screening |
| Gap 3 — Bayesian Trust Calibration | `feedback_loop.py`, `TrustProfile` DB table | Per-user, per-task-type trust score that closes the loop on live enforcement and persists across sessions |
| Frontend | `HaloFeaturesPanel.tsx` and related components | Live display of risk classification, injection alerts, and trust state |
| Evaluation | `test_injection_pages/`, `tests/test_gap*_*.py` | Labeled injection-detection test pages and the pytest suites behind the numbers above |

`Final_Report.pdf` has the full design rationale and the survey and evaluation this table summarizes.

---

## Requirements

- Python 3.10+
- Docker Desktop (required for the live browser view and the code execution sandbox — see [Docker images](#docker-images) below)
- Node.js 18+ (for frontend build)

---

## Installation

```bash
# From the HALO/ directory
pip install -e ".[dev]"
```

---

## Run

```bash
# Standard mode (requires Docker for browser agent and code execution)
halo --port 8081                        # `halo-app` is an identical alias

# Without Docker (browser/coder agents disabled, orchestrator only)
halo --port 8081 --run-without-docker

# With a custom model config (Ollama, Azure, InnKube, etc.)
halo --port 8081 --config ollama_config.yaml

# FARA-7B web surfer variant (see fara_config.yaml)
halo --port 8081 --config fara_config.yaml --fara

# CLI mode (no UI, single task)
halo-cli
```

Then open **http://localhost:8081** in your browser.

### Docker images

On first run (unless `--run-without-docker` is passed), HALO checks for two Docker images and **pulls them
automatically** if missing — no manual `docker build` needed for a standard run:

- a browser image (Playwright + noVNC, for the live browser view)
- a Python image (sandboxed code execution for the coder agent)

By default these are pulled from `ghcr.io/microsoft/magentic-ui-browser` and `ghcr.io/microsoft/magentic-ui-python-env`
(the upstream Magentic-UI images — HALO doesn't yet publish its own). To use a locally built image instead (e.g. after
editing `docker/halo-browser-docker/` or `docker/halo-python-env/`), set `HALO_BROWSER_IMAGE` / `HALO_PYTHON_IMAGE`
before launching:

```bash
export HALO_BROWSER_IMAGE=halo-browser-docker:local
export HALO_PYTHON_IMAGE=halo-python-env:local
halo --port 8081
```

---

## Configuration

Copy and adapt `ollama_config.yaml` to point at your LLM provider:

```yaml
orchestrator_client:
  provider: autogen_ext.models.openai.OpenAIChatCompletionClient
  config:
    model: "your-model"
    base_url: "https://your-endpoint/v1"
    api_key: "your-key"
    model_info:
      vision: false
      function_calling: true
      json_output: true
      family: unknown

adaptive_approval: true            # Gap 1: enable adaptive action guard (default: false)
hybrid_risk_estimation: true       # Gap 1: fuse rule-based classifier with an LLM layer (default: true)
hybrid_injection_detection: true   # Gap 2: fuse pattern scanner with a semantic LLM layer, everywhere (default: true)
enable_trust_feedback: true        # Gap 3: close the trust loop — enforce + persist, not just display (default: true)
```

These four flags are YAML-only — there is no UI Settings toggle for any of them. Set any to `false` to ablate that
layer for evaluation (e.g. `hybrid_injection_detection: false` for pattern-only scanning). Restart `halo` after
changing them. `enable_trust_feedback` already defaults to `true`, so `ollama_config.yaml` doesn't declare it
explicitly — only add it if you want to turn it *off* for an ablation run.

### The one UI setting that also has to be right

None of the four flags above matter if the **Action Approval Policy** in the UI is set to `Never require approval`.
`is_scan_active()` treats `approval_policy == "never"` as a global kill switch for Gap 1 and Gap 2 alike — with it
set, no task is classified, no content is scanned, and no injection alert can ever fire, regardless of the YAML
flags. To exercise the adaptive-oversight gaps, open the gear icon → **General** → **Action Approval Policy** and
pick anything other than "Never require approval":

| Dropdown label | Underlying value |
|---|---|
| Never require approval | `never` — disables Gap 1 + Gap 2 entirely |
| AI based judgement | `auto-conservative` — the sensible default for testing |
| Always require approval | `always` |

There is a fourth backend value, `auto-permissive`, that does **not** appear in this dropdown — it's not something
you select, it's something HALO puts itself into: Gap 1 sets it automatically when a task classifies as `research`,
and Gap 3 can promote a tier into it after enough approval history. Don't go looking for it in Settings.

---

## Frontend Development

```bash
cd frontend
yarn install   # or: npm install
yarn dev       # hot-reload dev server at localhost:8000
yarn build     # production build → copied to src/halo/backend/web/ui/
```

---

## Architecture

HALO's defining architectural decision is that the Injection Gateway is not a preprocessing step bolted onto one
edge of the pipeline — it's a mandatory checkpoint every party in the system crosses, in both directions, every
time, before reaching the trusted agent core. The user's own task text is scanned before the agent reasons over
it; the agent's proposed action is screened for hijacking before it's allowed to execute; and whatever comes back
from the outside world (a page, a file, code output, an MCP result) is scanned again before it re-enters the
agent's context. It's the same pattern as a network firewall placed in front of a trusted service — the agent
never talks to the user or the outside world directly.

```mermaid
flowchart TB
    User(["👤 User"])
    FW{{"🛡️ Injection Gateway — AI Firewall<br/>inspects every crossing shown below"}}
    Agent["🤖 Agent / Orchestrator<br/>(trusted core)"]
    Risk[["⚖️ Risk + Trust Engine<br/>auto-permissive / auto-conservative / always"]]
    World[("🌐 External World<br/>Web · Sandbox · MCP")]

    User -- "① task text" --> FW
    FW -- "⑧ verified answer / block" --> User

    FW -- "②⑥ clean task / verified content" --> Agent
    Agent -- "③⑦ proposed action / draft answer" --> FW

    FW -- "④ scanned action" --> World
    World -- "⑤ page / file / tool result" --> FW

    FW -. "Ⓐ risk score" .-> Risk
    Risk -- "Ⓑ approval policy" --> Agent
    Agent -. "Ⓒ approve / reject feedback" .-> Risk
```

Reading the round trip in order: **①** the user's task text is scanned before the agent ever sees it. **②** the
clean task reaches the agent, which reasons and **③** proposes an action, screened for hijacking before **④** it's
allowed to reach the external world. **⑤** whatever comes back is scanned again before **⑥** it re-enters the
agent's context. The agent's **⑦** draft answer is itself checked before **⑧** it's released to the user. Running
alongside that numbered trip, not inside it: **Ⓐ** the gateway's risk classification is handed to the Risk +
Trust Engine the moment the task clears step ②; **Ⓑ** the engine's resulting policy becomes the standard the
agent's next action is checked against; **Ⓒ** the user's approve/reject decision on that action feeds back into
the engine, closing the trust-calibration loop for the next task of the same type.

### Code map

The diagram above is the conceptual shape; this is where each piece actually lives:

```
Browser ──WebSocket──▶ backend/web/routes/ws.py
                              │
                       WebSocketManager (backend/web/managers/connection.py)
                              │
                       TeamManager.run_stream()
                              │
               ┌──────────────┼──────────────┬─────────────┐
               ▼              ▼              ▼             ▼
          HALOWebSurfer    HALOCoder    HALOFileSurfer   McpAgent
               │               │             │              │
               └───────────────┴─────────────┴──────────────┘
                              │
                        ApprovalGuard ◀── input_func ◀── WebSocket
                              │
                  injection_gateway.py (Gap 2 — the "AI Firewall" above)
              scan_and_gate() + action-hijack screening
                   — every agent above routes through it
                              │
                       task_classifier.py (Gap 1 — Risk half of the Engine)
                              │
                     feedback_loop.py (Gap 3 — Trust half of the Engine)
                     Bayesian trust, closed loop by default
```

---

## Research Gaps

Every task, with no exceptions, starts with a fresh Gap 1 risk check based only on that task's own words; Gap 3's
trust score then fine-tunes how strict approvals are *within* whatever category Gap 1 just assigned — it cannot
skip Gap 1, reclassify a task, or loosen a destructive task no matter how trusted the user is. Gap 2 runs
independently of both, at every point where an agent reads content it didn't write itself. See
[`Final_Report.pdf`](Final_Report.pdf) for the full design writeup and evaluation.

### Gap 1 — Adaptive Action Guard

**Files:** `src/halo/task_classifier.py` (rule layer), `src/halo/llm_risk_estimator.py` (optional LLM layer)  
Classifies each incoming task as `research`, `transactional`, or `destructive` — a word-boundary keyword classifier
by default, conservatively fused with an LLM risk estimate when `hybrid_risk_estimation: true`. The orchestrator
re-classifies on *every* new task message (not just once) and rewrites `ApprovalGuard.config.approval_policy`
before execution begins. Result: research tasks get `auto-permissive`; transactional tasks get `auto-conservative`;
anything destructive is hard-floored to `always` regardless of what the LLM layer or accumulated trust says.

### Gap 2 — Prompt Injection Visibility Layer (Injection Gateway)

**Files:** `src/halo/injection_gateway.py`, `src/halo/injection_scanner.py`, `src/halo/semantic_injection_detector.py`  
One shared function, `scan_and_gate()`, that every content-ingestion point in HALO routes through: web pages,
files, code-execution output, MCP tool results, and the user's own task text. A pattern scanner (25 signatures)
is conservatively fused with an optional semantic/LLM layer (`hybrid_injection_detection: true`). If detected, the
run pauses and the user sees a banner — **Block Page** / **Continue Anyway** — and the LLM never receives the
injected content unless the user explicitly allows it.

A second, independent check — **action-hijack screening** (`screen_action_for_hijack()`) — asks a different
question: not "does this content contain hidden instructions" but "does the agent's *proposed action* look like it
was redirected by something it previously read." It forces an approval prompt even when the normal risk policy
would have let the action through silently, and covers every agent (web surfer, coder, file surfer, MCP).

### Gap 3 — Bayesian Trust Calibration

**File:** `src/halo/feedback_loop.py`  
A per-user, per-task-type Beta-distribution trust score, built from every approve/reject/block decision.
`enable_trust_feedback: true` (the default) closes the loop: the trust-derived policy is written back into the
live `ApprovalGuard` — so a task type the user keeps approving needs fewer prompts over time, and one they keep
rejecting needs more — and persists across sessions via a per-user `TrustProfile` DB row, so it isn't reset every
run. Rejections cost more trust than approvals earn back (asymmetric weighting), and `destructive` tasks are
hard-floored to `always` no matter how high trust climbs. Set `enable_trust_feedback: false` to make trust
display-only, e.g. for ablation studies.

---

## Testing & Evaluation

```bash
poe test                                  # pytest suite, excludes tests needing npx
python tests/eval_bayesian_convergence.py # standalone: proves + simulates Gap 3 convergence
```

The live QA driver and the labeled precision/recall/F1 ground-truth corpus that produced the numbers in
[Results at a glance](#results-at-a-glance) are internal evaluation scripts, not included in this repo; the
methodology, full 22-scenario test matrix, and all 19 screenshots they produced are in
[`Final_Report.pdf`](Final_Report.pdf) and [`Results/`](Results/). For manual injection testing, serve
`test_injection_pages/` locally (a fresh session, `approval_policy` set to anything but `never`):

```bash
python -m http.server 8888 --directory test_injection_pages
```

---

## Report & Further Reading

- [`Final_Report.pdf`](Final_Report.pdf) — the seminar report: framework survey, HALO's design, and the full live evaluation (9 pages + references).
- [`Results/`](Results/) — all 19 screenshots from the live evaluation.

---

## License

See [LICENSE](LICENSE). Based on Magentic-UI — Copyright (c) Microsoft Corporation.
HALO extensions — Copyright (c) 2026 Ali Akbar.

---

## Author

**Ali Akbar** — [@AliAkbarBaloch](https://github.com/AliAkbarBaloch)
