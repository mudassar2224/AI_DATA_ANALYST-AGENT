from __future__ import annotations

import pandas as pd

from dsagent.agents.planner import AnalysisPlan, AnalysisStep, evaluate_plan, plan_next_analysis


class _PlanStubLLM:
    def __init__(self, plan: AnalysisPlan):
        self._plan = plan

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        return self._plan


def test_plan_next_analysis_uses_structured_output():
    plan = AnalysisPlan(
        objective="Find the strongest signal in the dataset",
        steps=[
            AnalysisStep(
                title="Check missingness",
                rationale="Missing values can distort the main signal.",
                python_code="result = {'missing': int(df.isna().sum().sum())}",
            )
        ],
    )
    llm = _PlanStubLLM(plan)

    generated = plan_next_analysis("Find the strongest signal", None, llm)

    assert generated.objective == "Find the strongest signal"
    assert generated.steps[0].title == "Check missingness"


def test_execute_plan_eval_runs_python_and_returns_summary():
    plan = AnalysisPlan(
        objective="Profile the target variable",
        steps=[
            AnalysisStep(
                title="Compute row count and missing values",
                rationale="Useful baseline before modeling.",
                python_code="result = {'rows': len(df), 'missing': int(df.isna().sum().sum())}",
            )
        ],
    )

    df = pd.DataFrame({"x": [1, 2, None, 4], "y": [10, 20, 30, 40]})
    result = evaluate_plan(df, plan)

    assert result.step_results[0]["rows"] == 4
    assert result.step_results[0]["missing"] == 1
    assert result.completed_step_titles == ["Compute row count and missing values"]
    assert "row count" in result.summary.lower()


def test_execute_plan_handles_invalid_python_gracefully():
    plan = AnalysisPlan(
        objective="Find the strongest signal",
        steps=[
            AnalysisStep(
                title="Build a derived column",
                rationale="A bad generated expression should not crash the app.",
                python_code=(
                    "(df['tenure_bin'] := pd.qcut(df['tenure_months'], q=4, duplicates='drop'))\n"
                    "result = {'status': 'bad-syntax'}"
                ),
            )
        ],
    )

    df = pd.DataFrame({"tenure_months": [1, 2, 3, 4, 5, 6]})
    result = evaluate_plan(df, plan)

    assert result.step_results == []
    assert "invalid python" in result.summary.lower()


def test_execute_plan_rejects_imports_and_private_attributes():
    plan = AnalysisPlan(
        objective="Check safety boundaries",
        steps=[
            AnalysisStep(
                title="Attempt an import",
                rationale="This must never execute.",
                python_code="import os\nresult = {'cwd': os.getcwd()}",
            ),
            AnalysisStep(
                title="Attempt private access",
                rationale="This must never execute.",
                python_code="result = {'class': df.__class__.__name__}",
            ),
        ],
    )

    result = evaluate_plan(pd.DataFrame({"x": [1, 2]}), plan)

    assert result.step_results == []
    assert "unsafe python" in result.summary.lower()
