"""Tests for dsagent.config.

Specifically covers a real bug found while building this: pydantic-settings
treats an empty `.env` value ("GROQ_MODEL=") as an explicit empty string,
not as "unset, use the default" — which would otherwise send an empty
model name to ChatGroq and fail confusingly.
"""

from __future__ import annotations

from dsagent.config import Settings


def test_default_model_when_unset():
    s = Settings()
    assert s.groq_model == "openai/gpt-oss-120b"


def test_empty_string_falls_back_to_default():
    s = Settings(groq_model="")
    assert s.groq_model == "openai/gpt-oss-120b"


def test_real_override_is_respected():
    s = Settings(groq_model="qwen/qwen3-32b")
    assert s.groq_model == "qwen/qwen3-32b"


def test_missing_api_key_is_falsy():
    s = Settings(groq_api_key=None)
    assert not s.groq_api_key


def test_openrouter_key_is_supported_as_secondary_provider():
    s = Settings(openrouter_api_key="or-key")
    assert s.openrouter_api_key == "or-key"
    assert s.has_any_llm_key is True
    assert s.llm_provider_chain[-1] == ("openrouter", "or-key")


def test_multiple_fallback_keys_are_collected_in_order():
    s = Settings(
        groq_api_key="key-1",
        groq_api_key_1="key-2",
        groq_api_key_2="",
        groq_api_key_3="key-3",
    )
    assert s.groq_api_keys == ["key-1", "key-2", "key-3"]


def test_blank_fallback_keys_are_ignored():
    s = Settings(groq_api_key="", groq_api_key_1="   ", groq_api_key_2="key-9")
    assert s.groq_api_keys == ["key-9"]


def test_no_sampling_by_default():
    # The whole point of this default: upload a dataset and get analysis
    # on the real thing, not a silent sample, unless explicitly configured.
    s = Settings()
    assert s.sample_rows_if_large is None


def test_sample_rows_empty_string_stays_none_not_a_crash():
    # int | None fields don't get the same "empty means unset" treatment
    # from pydantic-settings that a bare str field does -- an empty .env
    # value here would otherwise raise a ValidationError, not just pick
    # a wrong default.
    s = Settings(sample_rows_if_large="")
    assert s.sample_rows_if_large is None


def test_sample_rows_real_override_is_respected():
    s = Settings(sample_rows_if_large=50_000)
    assert s.sample_rows_if_large == 50_000
