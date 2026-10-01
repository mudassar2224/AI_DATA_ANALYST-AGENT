# AI Data Scientist Agent

An agent that understands a dataset before you ask it anything — upload a
CSV or Excel file and get an automatic profile, data-quality audit, and
chart, with no question required.

The deterministic layer (ingestion, profiling, quality audit,
relationship scan) always runs, no API key needed. The reasoning layer
(interpret -> verify and ask -> verify LangGraph loops, RAG-grounded)
turns on once you configure Groq or OpenRouter. A bounded next-best-analysis
planner also proposes and executes small pandas/numpy checks with validation
before execution. See [`docs/architecture.md`](docs/architecture.md) for the
full design and remaining production roadmap.

## Project layout

```
dsagent/
   config.py                settings, multiple Groq keys, OpenRouter fallback
  ingestion/loader.py       CSV/Excel loading, encoding + sheet detection
  profiling/profile.py      column type inference + per-column stats
  quality/checks.py         missingness, duplicates, outliers
  analysis/relationships.py correlation/association scan, FDR-corrected
  evidence/store.py         EvidenceStore — bundles the above, resolves refs
  viz/charts.py             adaptive charts: distributions + relationship pairs
  knowledge/base.py         RAG knowledge base (tag-retrieved, not a vector store)
  agents/schemas.py         Finding — qualitative claim + evidence_refs, no numbers
  agents/grounding.py       shared fail-closed ref-resolution helper
   agents/interpreter.py     evidence summary + provider-fallback LLM clients
  agents/verifier.py        pure-Python verification for Findings
   agents/qa.py              grounded question-answering agent
   agents/planner.py         bounded next-best-analysis planner and executor
   modeling/triage.py        deterministic task/target detection and recommendations
   modeling/baseline.py      leakage-aware baseline classification models and metrics
   agents/graph.py           both LangGraph loops: findings, and chat Q&A
streamlit_app.py             UI entrypoint — wires all of the above together
tests/                       pytest suite, one file per dsagent submodule
data/sample/                 demo dataset (regenerate with scripts/generate_sample_data.py)
docs/architecture.md         full plan, what's built vs not, how grounding works
```

## Setup

Requires [uv](https://docs.astral.sh/uv/). Everything else uv installs for you.

```bash
git clone <your-repo-url>
cd ai-data-scientist
uv sync
```

This creates `.venv/`, resolves and locks dependencies into `uv.lock`, and
installs `dsagent` itself in editable mode.

In VS Code: open the folder, install the recommended extensions when
prompted (Python, Pylance, Ruff), and select `.venv/bin/python` as the
interpreter if it isn't picked up automatically (bottom-right corner, or
`Cmd/Ctrl+Shift+P` → "Python: Select Interpreter").

## Run it locally

```bash
uv run streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`. Click **Try a sample dataset** to see
it working immediately, or upload your own CSV/XLSX.

Debugging in VS Code: use the **Streamlit: run app** configuration in the
Run and Debug panel (already set up in `.vscode/launch.json`) to run with
breakpoints.

## Run the tests

```bash
uv run pytest
```

Runs with coverage by default (configured in `pyproject.toml`), 112 tests
— including `test_app_state.py`, which uses Streamlit's own `AppTest`
harness to click buttons and rerun the real app, for state-persistence
bugs a plain function test can't catch.
The fixture in `tests/conftest.py` plants known issues — one missing
value, one duplicate row, one disguised-missing token, one numeric
outlier — so tests assert against ground truth, not just "it didn't
crash." `tests/test_relationships.py` does the same with planted signal
*and* planted noise, checking the FDR correction actually rejects the
noise pair. The whole reasoning layer (`test_interpreter.py`,
`test_qa.py`, `test_verifier.py`, `test_agents_graph.py`,
`test_qa_graph.py`) is tested against a stub LLM — zero network calls,
zero API key needed to run the suite. What a stub can't tell you is
whether Groq itself responds well to the prompt; that needs your own key
against the live app, once (see below).

Lint and format check (same commands CI runs):

```bash
uv run ruff check .
uv run ruff format --check .
```

## Trying the AI features

The deterministic analysis works without an API key. AI findings, chat, and
next-best analysis require a configured provider. Automated tests use stub
models and make zero network calls; live provider behavior depends on the
selected model, account limits, and network access.

1. Get a free key at [console.groq.com](https://console.groq.com).
2. Copy `.env.example` to `.env` and set one or more Groq keys. You can also
   set `OPENROUTER_API_KEY` as a secondary fallback route. Groq keys are tried
   in order (`GROQ_API_KEY`, then `_1`, `_2`, and `_3`); retryable failures
   rotate to the next configured key/provider. Never commit `.env` — it is
   already in `.gitignore`.
3. `uv run streamlit run streamlit_app.py`, then try the sample dataset.
   An **AI-interpreted findings** section should appear below
   Relationships, each finding with an "Evidence behind this finding"
   expander showing the exact refs it cited and what they resolved to.
   Below that, an **Ask about this dataset** chat box — try something
   like "is the target column imbalanced?" or a question it genuinely
   can't answer from the evidence, to see the `cannot_answer` path.
4. If something looks off — a finding or answer rejected that shouldn't
   have been, or the model consistently citing the wrong ref format —
    inspect the `SYSTEM_PROMPT` constants and `summarize_evidence()` first.
    Keep the verifier fail-closed: it is the safety boundary that prevents
    unsupported claims from reaching the UI.

### Running next-best analysis

When at least one provider key is configured, expand **Next-best analysis**
below the dataset overview and enter an analysis goal. The planner returns one
to three small steps. Each step is parsed and checked before execution; imports,
loops, functions/classes, private attributes, assignment expressions, and
oversized code are rejected. Failed steps are reported without crashing the
Streamlit app.

### Running baseline modeling

The **Model readiness and baseline** panel works without an API key. It detects
likely task type and target, excludes explicit identifier columns, recommends
metrics, and can train a majority-class baseline plus balanced logistic
regression for binary targets. For imbalanced targets, review PR-AUC, recall,
F1, and balanced accuracy rather than accuracy alone. These are transparent
baselines, not production models or proof of causation.

## Test locally, then deploy

Do this in order — don't push to a public deploy without running the app
and the test suite locally first:

1. `uv sync` — confirm it resolves cleanly
2. `uv run pytest` — all tests green
3. `uv run streamlit run streamlit_app.py` — click through it yourself,
   try both a bad file (e.g. a `.txt`) and the sample dataset
4. Commit and push to a GitHub repo **you own** (Streamlit Community
   Cloud needs admin rights on the repo to deploy it)
5. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, and create a new app:
   - Repository: your repo
   - Branch: `main`
   - Main file path: `streamlit_app.py` (must be at the repo root)
6. Streamlit Cloud detects `uv.lock` at the repo root and uses uv to
   install — no `requirements.txt` needed. If it asks for a Python
   version, match `.python-version` (3.12).
7. Add `GROQ_API_KEY` (and optional `_1`, `_2`, `_3`) under the app's
   **Settings → Secrets** to use Groq, or add `OPENROUTER_API_KEY` for the
   secondary provider route. `GROQ_MODEL` defaults to
   `openai/gpt-oss-120b`; `OPENROUTER_MODEL` defaults to
   `openrouter/free`, OpenRouter's free-model router. No OpenAI API key is
   needed. Free model availability and rate limits are controlled by OpenRouter.
8. Deploy. You'll get a `*.streamlit.app` URL.

Community Cloud's free tier caps out around ~2.7 GB RAM / 2 CPU per app
(Streamlit's own published figures, subject to change). By default this
app analyzes every row and every column pair — no sampling, no capping —
which is fine on a local machine and was tested end-to-end at 40,000
rows x 184 columns (about 8 seconds total, see `docs/architecture.md`).
If you deploy somewhere that constrained and hit a real memory ceiling
on a much larger file, `sample_rows_if_large` in `dsagent/config.py`
(and `find_relationships`'s `max_columns` parameter) can be set via
environment variables to sample/cap again — they still work, they're
just off by default now.

## Regenerating the sample dataset

```bash
uv run python scripts/generate_sample_data.py
```

Deterministic (fixed random seed) — same planted issues every time,
different data if you change the generator.
