"""A lightweight next-best-analysis planner and executor.

This is the first real execution loop for the project: the planner picks a
small set of analysis steps to run next, and the executor runs the generated
Python on the current dataframe. The goal is not full autonomous ML yet — it is
an explicit, testable step toward a real AI data-scientist workflow without
breaking the current evidence-grounded design.
"""

from __future__ import annotations

import ast
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from dsagent.agents.interpreter import summarize_evidence
from dsagent.evidence.store import EvidenceStore

SYSTEM_PROMPT_PLAN = """You are the next-best-analysis planner for a data-science assistant.

Your job is to propose one to three small, high-value analysis steps that
should be run next for the dataset in front of you.

Rules:
- Keep the steps concrete and explainable.
- Each step must include a short title, a rationale, and Python code that runs
    against a pandas DataFrame called df.
- Use only common pandas/numpy operations.
- The code should be simple and safe, and should assign a dictionary to a variable named result.
- Do not use imports, file/network access, dunder attributes, assignment expressions (:=),
  classes, functions, loops, or exception handling.
- Assign derived values to ordinary variables first, then use them in result; for example,
  `df['new_column'] = ...` is allowed, but `(df['new_column'] := ...)` is not valid Python.
- Prefer steps like missingness checks, distribution checks, segment comparisons,
  relationship checks, or baseline modeling signals.
- Return only the plan, not a narrative explanation outside the object.

Example output structure:
{
  "objective": "Find the strongest signal to investigate first",
  "steps": [
    {
      "title": "Check missingness",
      "rationale": "Missing values can distort the signal and are a common first check.",
      "python_code": "result = {'rows': len(df), 'missing_cells': int(df.isna().sum().sum()) }"
    }
  ]
}
"""


class AnalysisStep(BaseModel):
    title: str = Field(description="Short name for the analysis step.")
    rationale: str = Field(description="Why this step matters for the dataset.")
    python_code: str = Field(description="Python code that runs against df and assigns result.")


class AnalysisPlan(BaseModel):
    objective: str = Field(description="What the analysis is trying to learn next.")
    steps: list[AnalysisStep] = Field(min_length=1, max_length=3)


class AnalysisExecutionResult(BaseModel):
    objective: str
    step_results: list[dict[str, Any]]
    completed_step_titles: list[str] = Field(default_factory=list)
    summary: str


def _dataset_context(store: EvidenceStore | None) -> str:
    if store is None:
        return "Dataset context unavailable; use high-value, data-agnostic checks."
    return summarize_evidence(store)


def plan_next_analysis(
    objective: str,
    store: EvidenceStore | None,
    llm: BaseChatModel,
    retry_feedback: str = "",
) -> AnalysisPlan:
    """Ask the model to propose a set of next analysis actions for the current dataset."""
    parts = [
        f"Objective: {objective}",
        f"\nDataset context:\n{_dataset_context(store)}",
    ]
    if retry_feedback:
        parts.insert(0, f"Previous feedback:\n{retry_feedback}\n")

    messages = [
        SystemMessage(content=SYSTEM_PROMPT_PLAN),
        HumanMessage(content="\n".join(parts)),
    ]
    structured_llm = llm.with_structured_output(AnalysisPlan)
    result = structured_llm.invoke(messages)

    if isinstance(result, AnalysisPlan):
        if objective and result.objective != objective:
            result.objective = objective
        return result
    if isinstance(result, dict):
        parsed = AnalysisPlan.model_validate(result)
        if objective and parsed.objective != objective:
            parsed.objective = objective
        return parsed
    raise TypeError(f"Unexpected structured output type from planner: {type(result)!r}")


def _validate_step_code(python_code: str) -> str | None:
    """Reject malformed or clearly unsafe LLM-generated Python before execution.

    Assignment expressions (:=) are explicitly disallowed because they are
    invalid when used against dataframe subscript assignment patterns like
    `df['x'] := ...`, and they can also be hard to reason about in a sandboxed
    analysis environment.
    """
    if len(python_code) > 12_000:
        return "code is too long (maximum 12,000 characters)"

    try:
        tree = ast.parse(python_code, mode="exec")
    except SyntaxError as exc:
        return f"invalid Python: {exc.msg}"

    for node in ast.walk(tree):
        if isinstance(node, ast.NamedExpr):
            return "invalid Python: assignment expressions (:=) are not allowed"
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return "unsafe Python: imports are not allowed"
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return "unsafe Python: functions and classes are not allowed"
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith)):
            return "unsafe Python: loops, context managers, and exception handlers are not allowed"
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            return "unsafe Python: dunder/private attributes are not allowed"
    return None


def evaluate_plan(df: pd.DataFrame, plan: AnalysisPlan) -> AnalysisExecutionResult:
    """Execute a generated plan against the dataframe and return a summary."""
    safe_globals: dict[str, Any] = {
        "__builtins__": {
            "abs": abs,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "float": float,
            "int": int,
            "len": len,
            "list": list,
            "max": max,
            "min": min,
            "range": range,
            "round": round,
            "set": set,
            "str": str,
            "sum": sum,
            "tuple": tuple,
        },
        "pd": pd,
        "np": np,
        "df": df,
    }

    step_results: list[dict[str, Any]] = []
    completed_step_titles: list[str] = []
    failed_errors: list[str] = []
    for step in plan.steps:
        validation_error = _validate_step_code(step.python_code)
        if validation_error:
            failed_errors.append(f"{step.title}: {validation_error}")
            continue

        local: dict[str, Any] = {"df": df, "pd": pd, "np": np}
        try:
            exec(step.python_code, safe_globals, local)
        except Exception as exc:  # noqa: BLE001 - surface as a soft failure, not a crash.
            failed_errors.append(f"{step.title}: execution error: {type(exc).__name__}: {exc}")
            continue

        result = local.get("result")
        if result is None:
            result = {key: value for key, value in local.items() if key not in {"df", "pd", "np"}}
        if not isinstance(result, dict):
            result = {"value": result}
        step_results.append(result)
        completed_step_titles.append(step.title)

    if not step_results:
        summary = (
            f"Planner generated invalid Python for '{plan.objective}' and no steps ran. "
            + "; ".join(failed_errors)
            if failed_errors
            else f"No analysis steps completed for '{plan.objective}'."
        )
        return AnalysisExecutionResult(
            objective=plan.objective,
            step_results=[],
            completed_step_titles=[],
            summary=summary,
        )

    summary_lines = [
        f"Completed the planned analysis for '{plan.objective}'.",
        f"Ran {len(step_results)} step(s) on a dataframe with {len(df)} rows "
        f"and {len(df.columns)} columns.",
    ]
    for index, title in enumerate(completed_step_titles, start=1):
        summary_lines.append(f"Step {index}: {title}")
    if failed_errors:
        summary_lines.append(
            "Some steps were skipped because the generated Python was invalid: "
            + "; ".join(failed_errors)
        )

    return AnalysisExecutionResult(
        objective=plan.objective,
        step_results=step_results,
        completed_step_titles=completed_step_titles,
        summary=" ".join(summary_lines),
    )
