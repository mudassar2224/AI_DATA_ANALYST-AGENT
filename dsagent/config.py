"""Application configuration, loaded from environment variables or a .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

_DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
_DEFAULT_OPENROUTER_MODEL = "openrouter/free"
_LEGACY_OPENROUTER_MODEL = "openai/gpt-oss-20b:free"


class Settings(BaseSettings):
    """Central app settings. Override any of these via env vars or a .env file."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "AI Data Scientist"
    max_upload_mb: int = 200

    # None means "analyze the full dataset, always" -- the default. Set to
    # a row count (e.g. 200000) to sample down instead, if you're
    # deploying somewhere with tighter memory/CPU limits than a local
    # machine (see docs/architecture.md's Streamlit Cloud resource note).
    sample_rows_if_large: int | None = None

    # Groq is the primary provider and can have multiple keys configured for
    # automatic rotation when a free-tier key hits rate limits.
    groq_api_key: str | None = None
    groq_api_key_1: str | None = None
    groq_api_key_2: str | None = None
    groq_api_key_3: str | None = None
    groq_model: str = _DEFAULT_GROQ_MODEL

    # OpenRouter is the secondary fallback provider; it is intentionally not
    # a direct OpenAI route, and is used as a provider path through OpenRouter.
    openrouter_api_key: str | None = None
    openrouter_model: str = _DEFAULT_OPENROUTER_MODEL

    @property
    def groq_api_keys(self) -> list[str]:
        keys: list[str] = []
        for candidate in (
            self.groq_api_key,
            self.groq_api_key_1,
            self.groq_api_key_2,
            self.groq_api_key_3,
        ):
            cleaned = (candidate or "").strip()
            if cleaned and cleaned not in keys:
                keys.append(cleaned)
        return keys

    @property
    def llm_provider_chain(self) -> list[tuple[str, str]]:
        chain: list[tuple[str, str]] = [("groq", key) for key in self.groq_api_keys]
        if self.openrouter_api_key:
            chain.append(("openrouter", self.openrouter_api_key))
        return chain

    @property
    def has_any_llm_key(self) -> bool:
        return bool(self.groq_api_keys or self.openrouter_api_key)

    @field_validator(
        "groq_api_key",
        "groq_api_key_1",
        "groq_api_key_2",
        "groq_api_key_3",
        "openrouter_api_key",
        mode="before",
    )
    @classmethod
    def _normalize_api_key(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = str(value).strip()
        return stripped or None

    @field_validator("groq_model", mode="before")
    @classmethod
    def _empty_env_value_means_default(cls, value: str | None) -> str:
        # An `.env` line like "GROQ_MODEL=" sets this to "", not "unset" —
        # pydantic-settings doesn't fall back to the field default for an
        # explicitly-empty value, so we do it here instead of leaving a
        # blank model name to reach ChatGroq and fail confusingly.
        if not value:
            return _DEFAULT_GROQ_MODEL
        return value

    @field_validator("openrouter_model", mode="before")
    @classmethod
    def _empty_openrouter_model_means_default(cls, value: str | None) -> str:
        if not value:
            return _DEFAULT_OPENROUTER_MODEL
        cleaned = value.strip()
        if cleaned == _LEGACY_OPENROUTER_MODEL:
            return _DEFAULT_OPENROUTER_MODEL
        return cleaned

    @field_validator("sample_rows_if_large", mode="before")
    @classmethod
    def _empty_env_value_means_no_sampling(cls, value: object) -> object:
        # Same issue as groq_model above, but worse here: this field is
        # int | None, and pydantic-settings tries to parse an empty
        # ".env" string as an int rather than treating it as "unset" —
        # that raises a ValidationError and crashes the whole app, not
        # just silently picking a wrong value. Treat blank the same as
        # not being set at all, which for this field is also correct:
        # no sampling.
        if value == "":
            return None
        return value


settings = Settings()
