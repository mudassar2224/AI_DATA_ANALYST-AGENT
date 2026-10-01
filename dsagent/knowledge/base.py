"""The interpreter agent's RAG knowledge base.

Deliberately not a vector store: the reference material is small (a few
dozen short entries) and the interpreter already knows what KIND of
finding it's about to write (missingness, an outlier flag, a relationship
result...), so tag-based retrieval is exact, needs no embedding model or
API call, and is fully unit-testable. Swapping in a real vector store
(e.g. Chroma) later is a drop-in replacement for retrieve() if the
knowledge base grows large enough that semantic search earns its cost.

Thresholds here match the ones dsagent.analysis.relationships already
uses for effect-size labels, and the imbalance bands are the common
mild/moderate/severe convention from the imbalanced-learning literature.
These are rules of thumb, not universal cutoffs — the interpreter agent
is told as much in its prompt.
"""

from __future__ import annotations

from pydantic import BaseModel


class KnowledgeEntry(BaseModel):
    id: str
    tags: list[str]
    text: str


_ENTRIES: list[KnowledgeEntry] = [
    KnowledgeEntry(
        id="missingness-severity",
        tags=["missing", "quality"],
        text=(
            "As a rough guide: under 5% missing in a column is usually low-impact, "
            "5-20% is moderate and worth understanding before modeling, and above "
            "20% is high enough that the column's usability is in question unless "
            "there's a clear reason for the gap."
        ),
    ),
    KnowledgeEntry(
        id="missingness-mechanism",
        tags=["missing", "quality"],
        text=(
            "Missing data is usually described as MCAR (missing completely at "
            "random, no pattern), MAR (missing depends on other observed columns), "
            "or MNAR (missing depends on the unobserved value itself, e.g. high "
            "earners skipping an income field). Only note a mechanism when the "
            "evidence actually supports it — e.g. missingness concentrated in one "
            "segment suggests MAR, not MCAR."
        ),
    ),
    KnowledgeEntry(
        id="disguised-missing",
        tags=["missing", "quality"],
        text=(
            "Tokens like 'N/A', 'Unknown', '?', or '-' in a text column are "
            "usually a disguised missing value, not a real category, and should "
            "be flagged even though pandas doesn't treat them as null by default."
        ),
    ),
    KnowledgeEntry(
        id="duplicate-rows",
        tags=["duplicates", "quality"],
        text=(
            "A handful of exact duplicate rows in a large dataset is often a "
            "logging or export artifact rather than a serious problem, but "
            "duplicates concentrated around specific IDs or timestamps can "
            "indicate a join or ingestion bug worth investigating."
        ),
    ),
    KnowledgeEntry(
        id="outlier-iqr",
        tags=["outliers"],
        text=(
            "The IQR rule (values beyond 1.5x the interquartile range past Q1/Q3) "
            "flags statistically unusual values, not necessarily errors. Outliers "
            "concentrated in one segment or category are more likely to be real "
            "and meaningful than outliers scattered at random."
        ),
    ),
    KnowledgeEntry(
        id="class-imbalance-bands",
        tags=["imbalance", "target"],
        text=(
            "A common rule of thumb for binary class imbalance: minority class "
            "share of 20-40% is mild, 5-20% is moderate, and under 5% is severe. "
            "Severe imbalance means plain accuracy is a poor metric — precision, "
            "recall, F1, and PR-AUC give a truer picture."
        ),
    ),
    KnowledgeEntry(
        id="correlation-strength-spearman",
        tags=["relationship", "numeric_numeric"],
        text=(
            "For a correlation coefficient, |r| under 0.1 is negligible, 0.1-0.3 "
            "is weak, 0.3-0.5 is moderate, and above 0.5 is strong. Correlation "
            "describes association, not causation — say 'associated with', "
            "never 'causes'."
        ),
    ),
    KnowledgeEntry(
        id="cramers-v-strength",
        tags=["relationship", "categorical_categorical"],
        text=(
            "For Cramer's V (categorical-categorical association), under 0.1 is "
            "negligible, 0.1-0.2 is weak, 0.2-0.4 is moderate, and above 0.4 is "
            "strong."
        ),
    ),
    KnowledgeEntry(
        id="eta-squared-strength",
        tags=["relationship", "categorical_numeric"],
        text=(
            "For eta-squared (how much of a numeric column's variance is "
            "explained by a categorical grouping), under 0.01 is negligible, "
            "0.01-0.06 is weak, 0.06-0.14 is moderate, and above 0.14 is strong."
        ),
    ),
    KnowledgeEntry(
        id="multiple-testing",
        tags=["relationship"],
        text=(
            "When many column pairs are tested at once, some will look "
            "significant by chance alone. Only relationships that survive "
            "false-discovery-rate correction should be reported as real."
        ),
    ),
    KnowledgeEntry(
        id="high-cardinality",
        tags=["cardinality", "categorical"],
        text=(
            "A categorical column with a large number of unique values relative "
            "to its row count (rough guide: more than ~50 unique values, or more "
            "than 5% of rows) is high-cardinality. It's often better summarized "
            "by its top categories and a long-tail count than shown in full, and "
            "may actually be an identifier rather than a meaningful category."
        ),
    ),
]

_BY_TAG: dict[str, list[KnowledgeEntry]] = {}
for _entry in _ENTRIES:
    for _tag in _entry.tags:
        _BY_TAG.setdefault(_tag, []).append(_entry)


def retrieve(tags: list[str]) -> list[KnowledgeEntry]:
    """Return knowledge entries matching any of the given tags, deduplicated,
    in the knowledge base's own order.
    """
    seen: set[str] = set()
    results: list[KnowledgeEntry] = []
    for tag in tags:
        for entry in _BY_TAG.get(tag, []):
            if entry.id not in seen:
                seen.add(entry.id)
                results.append(entry)
    return results


def all_entries() -> list[KnowledgeEntry]:
    return list(_ENTRIES)
