"""AI Data Scientist — Streamlit entrypoint.

Deterministic layer (always runs): ingestion, profiling, quality audit,
relationship scan, adaptive charts. Reasoning layer (runs only if
GROQ_API_KEY is set): the interpret -> verify loop for automatic
findings, plus a chat box backed by the same ask -> verify loop for
follow-up questions. See docs/architecture.md for the full picture,
including what's genuinely not built (next-best-analysis, MCP,
LangSmith) and why planner/chat became one agent instead of two.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from dsagent.agents.graph import run_qa, run_reasoning_layer
from dsagent.agents.interpreter import build_default_llm
from dsagent.agents.planner import evaluate_plan, plan_next_analysis
from dsagent.agents.qa import VerifiedAnswer
from dsagent.agents.schemas import VerificationResult
from dsagent.analysis.relationships import RelationshipReport, find_relationships
from dsagent.config import settings
from dsagent.evidence.store import EvidenceStore
from dsagent.ingestion.loader import LoadResult, UnsupportedFileError, load_file
from dsagent.modeling.baseline import run_baseline_classification
from dsagent.modeling.triage import infer_target_column, triage_dataset
from dsagent.profiling.profile import DatasetProfile, profile_dataset
from dsagent.quality.checks import (
    DuplicateReport,
    MissingReport,
    OutlierFlag,
    check_duplicates,
    check_missingness,
    check_outliers,
)
from dsagent.viz.charts import column_chart, missing_values_chart, relationship_chart

SAMPLE_DATA_PATH = Path(__file__).parent / "data" / "sample" / "customers_sample.csv"
SEVERITY_ICON = {"info": "ℹ️", "watch": "🟡", "concern": "🔴"}
MAX_RELATIONSHIP_CHARTS = 3


st.set_page_config(page_title=settings.app_name, page_icon="\U0001f9ea", layout="wide")
st.title(settings.app_name)
st.caption(
    "Upload a dataset and get an automatic profile, quality audit, and chart "
    "— no question required."
)


@st.cache_data(show_spinner=False)
def _run_deterministic_pipeline(
    file_bytes: bytes, filename: str
) -> tuple[
    LoadResult,
    DatasetProfile,
    MissingReport,
    DuplicateReport,
    list[OutlierFlag],
    RelationshipReport,
]:
    """Everything below this line is plain pandas/scipy — no LLM call, so
    it's safe (and cheap) to cache on the file's own bytes.
    """
    load_result = load_file(file_bytes, filename, max_rows=settings.sample_rows_if_large)
    df = load_result.dataframe
    profile = profile_dataset(df)
    column_types = {c.name: c.inferred_type for c in profile.columns}
    return (
        load_result,
        profile,
        check_missingness(df),
        check_duplicates(df),
        check_outliers(df),
        find_relationships(df, column_types),
    )


@st.cache_data(show_spinner=False)
def _run_reasoning_layer_cached(
    evidence_json: str,
    model: str,
    api_key: str | list[str] | tuple[str, ...] | None,
    *,
    openrouter_api_key: str | None = None,
    openrouter_model: str | None = None,
) -> dict:
    """Cached on the evidence content + model, not the API keys' validity --
    so switching keys with the same value doesn't cost an extra call, but
    a genuinely different dataset or model does.

    Raises on failure rather than returning an error dict: st.cache_data
    only caches a successful return, so a raised exception is retried
    fresh on the next run instead of being remembered as a permanent
    failure. A 413 (request too large) or a transient 429 both look the
    same to this function -- either way, this run didn't get findings.
    """
    store = EvidenceStore.model_validate_json(evidence_json)
    llm = build_default_llm(
        model,
        api_key,
        openrouter_api_key=openrouter_api_key,
        openrouter_model=openrouter_model,
    )
    result = run_reasoning_layer(store, llm)
    if result.error:
        raise RuntimeError(result.error)
    return result.model_dump(mode="json")


def _render_verified_answer(verified: VerifiedAnswer) -> None:
    answer = verified.answer
    if answer.confidence == "cannot_answer":
        st.write(f"🤷 {answer.answer}")
        return
    if not verified.is_grounded:
        st.write("⚠️ Got an answer, but it cited evidence that didn't check out — not shown.")
        return
    st.write(answer.answer)
    with st.expander("Evidence behind this answer"):
        for ref, value in verified.resolved_values.items():
            st.code(f"{ref} = {value}", language=None)


col_upload, col_sample = st.columns([3, 1])
with col_upload:
    uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
with col_sample:
    st.write("")
    st.write("")
    use_sample = st.button("Try a sample dataset", width="stretch")

file_bytes: bytes | None = None
filename: str | None = None

if uploaded is not None:
    file_bytes, filename = uploaded.getvalue(), uploaded.name
    st.session_state["_active_upload"] = (file_bytes, filename)
elif use_sample:
    file_bytes, filename = SAMPLE_DATA_PATH.read_bytes(), SAMPLE_DATA_PATH.name
    st.session_state["_active_upload"] = (file_bytes, filename)
elif "_active_upload" in st.session_state:
    # Neither widget produced a fresh value on this run. st.file_uploader
    # keeps its own value across an unrelated rerun (e.g. one triggered by
    # the chat box below), but st.button only returns True on the exact
    # run it was clicked, then reverts to False — so without this fallback,
    # a dataset loaded via "Try a sample dataset" would vanish (file_bytes
    # goes back to None, hitting st.stop() below) the moment any other
    # widget on the page, including the chat input, triggers a rerun.
    file_bytes, filename = st.session_state["_active_upload"]

if file_bytes is None:
    st.info("Upload a file, or try the sample dataset, to see the analysis.")
    st.stop()

max_bytes = settings.max_upload_mb * 1024 * 1024
if len(file_bytes) > max_bytes:
    st.error(f"That file is larger than the {settings.max_upload_mb} MB limit for this demo.")
    st.stop()

try:
    load_result, dataset_profile, missing, duplicates, outliers, relationships = (
        _run_deterministic_pipeline(file_bytes, filename)
    )
except UnsupportedFileError as exc:
    st.error(str(exc))
    st.stop()

df = load_result.dataframe
target_column = infer_target_column(df)
dataset_triage = triage_dataset(df, target_column)
target_relationships = (
    find_relationships(
        df,
        {c.name: c.inferred_type for c in dataset_profile.columns},
        target_column=target_column,
    )
    if target_column
    else None
)

# reset chat history when a new file comes in, so old answers don't linger
# against evidence that no longer matches what's on screen
if st.session_state.get("_active_file") != filename:
    st.session_state["_active_file"] = filename
    st.session_state["chat_history"] = []
    st.session_state.pop("baseline_results", None)

if load_result.was_sampled:
    st.warning(
        f"This file has more than {settings.sample_rows_if_large:,} rows — "
        "showing analysis on a random sample of that size."
    )
if len(load_result.available_sheets) > 1:
    st.caption(
        f"Using sheet '{load_result.sheet_name}' "
        f"(other sheets in this file: {', '.join(load_result.available_sheets)})"
    )

st.subheader("Dataset overview")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows", f"{dataset_profile.row_count:,}")
m2.metric("Columns", dataset_profile.column_count)
m3.metric("Duplicate rows", dataset_profile.duplicate_row_count)
m4.metric(
    "Numeric / categorical / datetime",
    f"{dataset_profile.numeric_columns} / {dataset_profile.categorical_columns} / "
    f"{dataset_profile.datetime_columns}",
)

if settings.has_any_llm_key:
    with st.expander("Next-best analysis", expanded=False):
        analysis_goal = st.text_input(
            "Analysis goal",
            value="Find the strongest signal to investigate next in this dataset.",
        )
        if st.button("Run next-best analysis", use_container_width=True):
            llm = build_default_llm(
                settings.groq_model,
                settings.groq_api_keys,
                openrouter_api_key=settings.openrouter_api_key,
                openrouter_model=settings.openrouter_model,
            )
            store = EvidenceStore(
                profile=dataset_profile,
                missing=missing,
                duplicates=duplicates,
                outliers=outliers,
                relationships=relationships,
            )
            try:
                plan = plan_next_analysis(analysis_goal, store, llm)
                execution = evaluate_plan(df, plan)
            except Exception as exc:  # noqa: BLE001 - keep provider failures inside the UI
                st.error(
                    "The next-best-analysis planner could not complete this request. "
                    f"{type(exc).__name__}: {exc}"
                )
            else:
                st.write(execution.summary)
                completed_steps = {step.title: step for step in plan.steps}
                for title, result in zip(
                    execution.completed_step_titles, execution.step_results, strict=True
                ):
                    step = completed_steps[title]
                    st.markdown(f"### {title}")
                    st.caption(step.rationale)
                    st.json(result)

st.subheader("Preview")
st.dataframe(df.head(50), width="stretch")

st.subheader("Data quality")
q1, q2 = st.columns(2)
q1.metric(
    "Missing cells",
    f"{missing.total_missing_cells:,}",
    f"{missing.overall_missing_pct}% of all cells",
)
q2.metric(
    "Exact duplicate rows",
    f"{duplicates.exact_duplicate_rows:,}",
    f"{duplicates.exact_duplicate_pct}%",
)

if missing.columns_with_disguised_missing:
    st.warning(
        "These columns contain values like 'N/A', 'Unknown', or '?' that pandas "
        "doesn't treat as missing by default: "
        + ", ".join(f"`{c}`" for c in missing.columns_with_disguised_missing)
    )

chart = missing_values_chart(df)
if chart is not None:
    st.plotly_chart(chart, width="stretch")

if outliers:
    st.subheader("Possible outliers (IQR rule)")
    st.dataframe(pd.DataFrame([o.model_dump() for o in outliers]), width="stretch")

st.subheader("Column distributions")
chartable = [
    c for c in dataset_profile.columns if c.inferred_type in ("numeric", "categorical", "boolean")
]
if chartable:
    picked_name = st.selectbox("Column", [c.name for c in chartable])
    picked_type = next(c.inferred_type for c in chartable if c.name == picked_name)
    dist_fig = column_chart(df[picked_name], picked_name, picked_type)
    if dist_fig is not None:
        st.plotly_chart(dist_fig, width="stretch")
else:
    st.caption(
        "No numeric or categorical columns to chart (text/datetime charts aren't built yet)."
    )

if target_column and target_relationships is not None:
    st.subheader("Target analysis")
    target_counts = df[target_column].value_counts(dropna=False).sort_index()
    st.caption(
        f"Detected likely target `{target_column}`. This section compares every other "
        "usable column against it; associations are not causal predictions."
    )
    st.dataframe(
        pd.DataFrame(
            {
                "target_value": target_counts.index.astype(str),
                "count": target_counts.values,
                "share_pct": (target_counts.values / len(df) * 100).round(2),
            }
        ),
        width="stretch",
    )
    target_signals = sorted(
        target_relationships.results,
        key=lambda r: abs(r.metric_value),
        reverse=True,
    )[:10]
    if target_signals:
        st.caption(
            "Strongest target associations. Small effects are shown for discovery, "
            "but are not treated as practically significant."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "feature": r.column_b if r.column_a == target_column else r.column_a,
                        "metric": r.metric_name,
                        "value": r.metric_value,
                        "strength": r.effect_label,
                        "adjusted_p_value": r.adjusted_p_value,
                        "fdr_significant": r.is_significant,
                    }
                    for r in target_signals
                ]
            ),
            width="stretch",
        )

with st.expander("Model readiness and baseline", expanded=False):
    st.write(f"**Detected task:** `{dataset_triage.task_type}`")
    if dataset_triage.target_column:
        st.write(f"**Target:** `{dataset_triage.target_column}`")
    st.caption(dataset_triage.rationale)
    if dataset_triage.identifier_columns:
        st.write(
            "Excluded identifier-like columns: "
            + ", ".join(f"`{column}`" for column in dataset_triage.identifier_columns)
        )
    st.write("**Recommended metrics:** " + ", ".join(dataset_triage.recommended_metrics))

    if dataset_triage.task_type == "binary_classification" and dataset_triage.target_column:
        if dataset_triage.target_positive_rate is not None:
            st.caption(
                f"Minority-class rate: {dataset_triage.target_positive_rate:.1%}. "
                "Accuracy should not be the only metric."
            )
        if st.button("Run baseline classification", use_container_width=True):
            with st.spinner("Training deterministic baseline models..."):
                try:
                    baseline_results = run_baseline_classification(df, dataset_triage.target_column)
                except Exception as exc:  # noqa: BLE001 - shown without breaking the page
                    st.error(f"Baseline modeling could not run: {type(exc).__name__}: {exc}")
                else:
                    st.session_state["baseline_results"] = [
                        result.model_dump() for result in baseline_results
                    ]
        for result in st.session_state.get("baseline_results", []):
            st.markdown(f"#### {result['model_name'].title()}")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"metric": metric, "value": value}
                        for metric, value in result["metrics"].items()
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
            for warning in result["warnings"]:
                st.caption(f"⚠️ {warning}")
    else:
        st.info(
            "No binary classification baseline is available for this dataset yet. "
            "Choose a target or use the recommended unsupervised/regression workflow."
        )

st.subheader("Relationships")
significant = [r for r in relationships.results if r.is_significant]
if relationships.was_capped:
    st.caption(
        f"{dataset_profile.column_count} columns is a lot to scan pairwise — "
        f"limited to the first {len(relationships.columns_considered)} for this view."
    )
if significant:
    st.caption(
        f"{len(significant)} of {relationships.pairs_tested} tested pairs survived "
        f"false-discovery-rate correction (α={relationships.alpha}):"
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "column A": r.column_a,
                    "column B": r.column_b,
                    "metric": r.metric_name,
                    "value": r.metric_value,
                    "strength": r.effect_label,
                }
                for r in significant
            ]
        ),
        width="stretch",
    )
    for r in significant[:MAX_RELATIONSHIP_CHARTS]:
        rel_fig = relationship_chart(df, r.column_a, r.column_b, r.relationship_type)
        if rel_fig is not None:
            st.plotly_chart(rel_fig, width="stretch")
else:
    st.caption(
        f"None of the {relationships.pairs_tested} tested column pairs survived "
        "false-discovery-rate correction — no relationships reported as significant."
    )

st.subheader("Column profile")
st.dataframe(pd.DataFrame([c.model_dump() for c in dataset_profile.columns]), width="stretch")

st.subheader("AI-interpreted findings")
evidence_store = EvidenceStore(
    profile=dataset_profile,
    missing=missing,
    duplicates=duplicates,
    outliers=outliers,
    relationships=relationships,
)

if not settings.has_any_llm_key:
    st.info(
        "Add one or more Groq keys such as `GROQ_API_KEY`, `GROQ_API_KEY_1`, or "
        "`GROQ_API_KEY_2`, and optionally an `OPENROUTER_API_KEY`, in a `.env` file "
        "(see `.env.example`) to unlock this section and the chat box below. "
        "Everything else on this page works without it."
    )
else:
    with st.spinner("Interpreting the evidence..."):
        try:
            raw_state = _run_reasoning_layer_cached(
                evidence_store.model_dump_json(),
                settings.groq_model,
                settings.groq_api_keys,
                openrouter_api_key=settings.openrouter_api_key,
                openrouter_model=settings.openrouter_model,
            )
        except Exception as exc:  # noqa: BLE001 - shown to the user, not fatal to the page
            raw_state = {
                "error": f"{type(exc).__name__}: {exc}",
                "verification": None,
                "attempt": 0,
            }

    if raw_state.get("error"):
        st.warning(
            f"Couldn't generate AI findings this run ({raw_state['error']}). "
            "The rest of this page is unaffected — it's all computed above, not generated."
        )
    else:
        verification = VerificationResult.model_validate(raw_state["verification"])
        if not verification.verified:
            st.caption("The model didn't ground any findings in the evidence this run.")
        for vf in verification.verified:
            finding = vf.finding
            icon = SEVERITY_ICON.get(finding.severity, "•")
            with st.container(border=True):
                st.markdown(f"{icon} **{finding.headline}**")
                st.write(finding.meaning)
                st.caption(f"Next: {finding.next_step}")
                with st.expander("Evidence behind this finding"):
                    for ref, value in vf.resolved_values.items():
                        st.code(f"{ref} = {value}", language=None)
        if verification.rejected_finding_ids:
            st.caption(
                f"{len(verification.rejected_finding_ids)} finding(s) were generated but "
                "rejected by the verifier for citing evidence that didn't check out, "
                "and aren't shown."
            )

    st.subheader("Ask about this dataset")
    for role, content in st.session_state.get("chat_history", []):
        with st.chat_message(role):
            if role == "assistant":
                _render_verified_answer(content)
            else:
                st.write(content)

    question = st.chat_input("e.g. Is the target column imbalanced?")
    if question:
        st.session_state.setdefault("chat_history", []).append(("user", question))
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Checking the evidence..."):
                try:
                    llm = build_default_llm(
                        settings.groq_model,
                        settings.groq_api_keys,
                        openrouter_api_key=settings.openrouter_api_key,
                        openrouter_model=settings.openrouter_model,
                    )
                    qa_state = run_qa(question, evidence_store, llm)
                except Exception as exc:  # noqa: BLE001 - shown to the user
                    st.warning(f"Couldn't answer that ({type(exc).__name__}: {exc}).")
                    qa_state = None
            if qa_state is not None:
                if qa_state.error:
                    st.warning(f"Couldn't answer that ({qa_state.error}).")
                elif qa_state.verified is not None:
                    _render_verified_answer(qa_state.verified)
                    st.session_state["chat_history"].append(("assistant", qa_state.verified))


