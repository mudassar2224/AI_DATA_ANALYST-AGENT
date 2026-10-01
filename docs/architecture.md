# Architecture

Two layers. The bottom layer never calls an LLM — it's pandas/scipy/
statsmodels wrapped as plain functions, so a number in the output can't
be a hallucination, only a bug. The top layer is where Groq, LangGraph,
and RAG live, and it's built so its LLM calls can't put an unverified
number in front of you either — see "How grounding actually works."

## Layer 1 — deterministic specialists (built)

```
Ingestion -> {Profiler, Quality audit, Relationships} -> Evidence store -> Charts
```

- `dsagent/ingestion/loader.py` — encoding + sheet detection, row sampling
- `dsagent/profiling/profile.py` — column type inference, per-column stats
- `dsagent/quality/checks.py` — missingness (incl. disguised tokens like
  "N/A"/"Unknown"), exact duplicates, IQR outlier flags
- `dsagent/analysis/relationships.py` — Spearman / Cramer's V / eta-squared
  across column pairs, Benjamini-Hochberg FDR-corrected so scanning many
  pairs doesn't manufacture false "insights" (`tests/test_relationships.py`
  checks planted signal is found and planted noise is not)
- `dsagent/evidence/store.py` — `EvidenceStore` bundles all of the above
  and resolves dotted-path refs like `missing.by_column_pct.Income` or
  `relationships.results[tenure_months,churn]`
- `dsagent/viz/charts.py` — the adaptive part: `column_chart()` picks a
  histogram or a top-N-plus-"Other" bar chart by the column's own type
  and cardinality, not a fixed checklist; `relationship_chart()` does the
  same for a column pair (scatter / box / stacked bar, keyed off
  `relationship_type`)
- `dsagent/modeling/triage.py` — detects likely task type, target, identifiers,
  class balance, and recommended evaluation metrics without an LLM
- `dsagent/modeling/baseline.py` — majority-class and balanced logistic-regression
  baselines for binary targets, with preprocessing and leakage-aware metrics

## Layer 2 — reasoning agents (built: interpret+verify, ask+verify, bounded planner)

```
                     Evidence store
                     /            \
          Interpreter (Groq+RAG)   Ask (Groq+RAG, one question)
                     \            /
                    Verifier (pure Python, fail-closed)
                     /            \
          Findings (auto, upload)  Answer (on demand, chat)
```

Both loops retry on rejection (feedback about what went wrong, up to
`max_attempts`) and both catch any exception from the LLM call itself
and surface it as `state.error` instead of crashing.

- `dsagent/knowledge/base.py` — the RAG knowledge base: curated
  data-science reference entries (imbalance bands, effect-size
  thresholds, missingness mechanisms...), retrieved by tag rather than a
  vector store. Deliberately lightweight — the knowledge base is small
  and both agents already know what kind of thing they're about to
  write, so exact tag matching is simpler and more testable than
  semantic search at this scale. A real vector store is a drop-in
  replacement for `retrieve()` if the knowledge base grows.
- `dsagent/agents/schemas.py` — `Finding`: qualitative headline +
  `evidence_refs`. The LLM never writes the number the user sees.
- `dsagent/agents/interpreter.py` — builds the evidence summary + RAG
  context, calls the LLM via LangChain structured output, for the
  auto-run findings.
- `dsagent/agents/qa.py` — the question-answering agent. **This replaces
  the Planner + Chat split from the original plan.** Planner's job was
  to turn a question into a call against a tool registry — but every
  specialist here already runs eagerly on upload, so there's no
  selection decision left to make by the time a question arrives; the
  evidence a question needs is always already in the store. What was
  going to be two agents is one: read the question, answer from what's
  already computed, say "cannot_answer" plainly when the evidence
  doesn't cover it.
- `dsagent/agents/grounding.py` — `resolve_refs()`, the fail-closed
  resolution loop shared by both verifiers below, so the two don't
  duplicate it.
- `dsagent/agents/verifier.py` / the verify step in `qa.py` — pure
  Python, no LLM call. A Finding is accepted only if every ref it cites
  resolves; a QAAnswer is accepted if every ref resolves OR it correctly
  said `cannot_answer`. Both are fully unit-tested without touching Groq.
- `dsagent/agents/graph.py` — both LangGraph loops:
  `build_graph`/`run_reasoning_layer` (findings) and
  `build_qa_graph`/`run_qa` (chat). Both take `llm` as a parameter, which
  is what let every node in this package get tested against a stub —
  see `tests/test_agents_graph.py` and `tests/test_qa_graph.py` for the
  happy path, the self-correcting retry, exhausting the retry budget
  without looping forever, and the exception-handling path (confirmed
  against a real `ChatGroq` call from this dev environment, which has no
  network route to `api.groq.com` and so exercises the exact "the call
  failed" branch a real rate limit or bad key would hit).

### How grounding actually works

Both agents are told the exact dotted-path refs available (printed into
the prompt by `summarize_evidence()`) and told never to invent one. The
verifier re-resolves every ref the model actually used against the same
`EvidenceStore` the prompt was built from. If a ref doesn't resolve, the
whole finding or answer is dropped, not trimmed. What reaches the page
is the LLM's interpretation next to the real, looked-up value — the
number on screen was never generated, only looked up.

### Not built

| Piece | Role | Status |
|---|---|---|
| **Next-best-analysis** | Bounded loop that proposes and executes one to three validated pandas/numpy checks | Built: first bounded version; not a full autonomous ML workflow |
| **MCP** | Expose the specialist registry as an MCP server | Optional/stretch |
| **LangSmith** | Tracing, eval datasets, prompt-version comparison | Not wired in (needs its own API key) |

Also not built: text/datetime-specific charts (`column_chart()` returns
`None` for those types), regression and multiclass baseline training,
automated model comparison beyond the initial binary baselines, and
multi-turn memory in the chat beyond what's in `st.session_state` for the
current session.

## No sampling, no column cap — by design, not by default happenstance

Earlier versions of this project sampled large datasets down and capped
wide datasets to the first N columns for relationship scanning, mainly
as a nod to Streamlit Community Cloud's resource limits. That's gone:
`Settings.sample_rows_if_large` defaults to `None` (analyze every row),
and `find_relationships`'s `max_columns` defaults to `None` (test every
pair). A dataset shaped like a real 184-column, 40,000-row file — genuinely
tested, not estimated — profiles, audits, and scans all 16,653 possible
pairs in about 8 seconds total, no sampling, no capping.

That number holds because the numeric-numeric case (usually the large
majority of pairs in a wide dataset) is one vectorized `scipy.stats.
spearmanr` call across the whole numeric matrix, not a Python loop over
individual pairs — about 200x faster in testing (178 columns, 15,753
pairs: 10.3s looped vs 0.05s vectorized). `nan_policy="omit"` keeps this
correct despite being batched: a missing value in one column only
affects pairs involving that column, verified directly against
per-pair computation in `tests/test_relationships.py`, not just assumed.
Categorical-categorical and categorical-numeric pairs stay as a loop
(cramers_v and eta-squared don't vectorize as cleanly, and in practice
there are usually far fewer categorical columns than numeric ones, so
this isn't the bottleneck — confirmed at 40,000 rows in the same test).

Both settings remain available, not deleted, for anyone deploying
somewhere genuinely resource-constrained: set `SAMPLE_ROWS_IF_LARGE` in
`.env`, or pass `max_columns` to `find_relationships` directly.

A related, smaller fix in the same spirit: `summarize_evidence()` used
to only mention the subset of columns it detailed in full, which meant
a question like "list all column names" on a wide dataset got answered
with "I can't, the evidence only shows a subset" — technically correct
per what the model was shown, but not what anyone asking that question
wants to hear. The complete column-name list is now always included
(cheap — a few hundred tokens even at 184 columns) independent of how
many get the expensive full stats treatment; see
`test_summary_lists_every_column_name_even_when_detail_is_capped` in
`tests/test_interpreter.py`.

## Model choice and rate limits on Groq

Default is `openai/gpt-oss-120b` (production tier — won't be pulled
without notice); `qwen/qwen3-32b` is a one-line config swap (preview
tier — fine to build against, don't depend on it alone for something
kept live). Free tier is roughly 30 requests/min and 1,000/day — the
findings loop is cached on evidence content (`streamlit_app.py`), and
the chat box costs one call per question by nature; both degrade
gracefully to an error message rather than crashing if the limit is hit.

Two failure modes look similar but need different fixes, and conflating
them was a real bug caught against an actual 184-column dataset:

- **429 (rate limit exceeded)** — too many requests too fast. Genuinely
  transient; retrying after a short wait usually works.
  `ChatGroq`'s underlying client already does this automatically
  (`max_retries=4`, set in `build_default_llm()`), no custom retry loop
  needed on top.
- **413 (request too large)** — this one *request's* prompt exceeds the
  model's per-request token budget (a fixed 40-column dataset's worth of
  column stats is fine; 184 columns' worth, plus 178 numeric columns
  each getting an outlier-flag line, was not — 13.8k tokens against an
  8k TPM limit). Retrying the same oversized request just gets the same
  413 again — no amount of backoff fixes a payload that's too big.
  `summarize_evidence()` in `dsagent/agents/interpreter.py` now caps
  itself on three axes (columns detailed, outliers detailed, and a hard
  character ceiling as a final backstop) specifically so this can't
  recur — see its docstring and the regression tests in
  `tests/test_interpreter.py`.

A related fix: `_run_reasoning_layer_cached()` in `streamlit_app.py`
used to cache a *failed* run right alongside successful ones (Streamlit
caches whatever a `@st.cache_data` function returns, and the function
was returning an error dict instead of raising) — meaning one 413 would
get remembered as the permanent answer for that dataset until the app
restarted, even after the payload-size fix landed. It now raises on
failure instead, which `st.cache_data` correctly treats as "don't
cache," so every attempt gets a fresh try.

| Technique | Role | Where |
|---|---|---|
| LangGraph | Orchestration — both the findings loop and the chat loop are graphs | `dsagent/agents/graph.py` |
| LangChain | `ChatGroq` client + structured-output parsing | `interpreter.py`, `qa.py` |
| RAG | Tag-retrieved data-science reference knowledge | `dsagent/knowledge/base.py` |
| Groq/OpenRouter | Provider chain for LLM calls, with multiple Groq keys and OpenRouter fallback | via `build_default_llm()` |
| LangSmith | — | not wired in |
| MCP | — | not wired in |

## A note on Streamlit's rerun model

Every widget interaction — including submitting the chat box — reruns
the *entire* script from top to bottom. `st.file_uploader`'s value
persists across a rerun it wasn't itself part of, but `st.button` does
not: it's `True` only on the exact run it was clicked, `False` on every
run after. A dataset loaded via "Try a sample dataset" was vanishing the
moment the chat box (or anything else) triggered a rerun, because
nothing was remembering that choice past that one run — `st.stop()`
would fire again as if no dataset had ever been loaded. Fixed by
stashing the active file in `st.session_state` the moment either the
uploader or the sample button produces one, and falling back to that on
any run where neither widget has a fresh value. Regression test:
`tests/test_app_state.py`, using Streamlit's own `AppTest` harness to
actually click the button and rerun the real app, not just the
persistence logic in isolation.

One trade-off from this fix, worth knowing: clicking the "x" on an
uploaded file no longer immediately clears the analysis, since that also
looks like "the uploader has no fresh value this run" from the app's
point of view. Upload a different file, or refresh the page, to switch
away from a loaded dataset.
