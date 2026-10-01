"""Regression test for a real bug: a dataset loaded via the "Try a sample
dataset" button disappeared the moment any other widget on the page
triggered a rerun (the chat box, in particular) — st.button only returns
True on the exact run it was clicked, then reverts to False, and nothing
was persisting the loaded file across the reruns that came after.

Uses Streamlit's own AppTest harness to run the real app and click the
real button, rather than testing the persistence logic in isolation —
this is a bug about *reruns*, so the test has to actually rerun the app
to mean anything. No GROQ_API_KEY is set, so this only exercises the
deterministic path; that's the point — the bug and the fix are both
about whether the dataset survives, not about anything LLM-related.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).parent.parent / "streamlit_app.py")


def test_no_dataset_shows_upload_prompt():
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert at.exception == []
    assert any("Upload a file" in b.value for b in at.info)
    assert len(at.subheader) == 0


def test_sample_dataset_loads():
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    at.button[0].click().run()
    assert at.exception == []
    headers = [h.value for h in at.subheader]
    assert "Dataset overview" in headers


def test_sample_dataset_survives_an_unrelated_rerun():
    # This is the actual regression: rerun the app again with no new
    # interaction, simulating what st.chat_input (or any other widget)
    # does when submitted — the loaded dataset must still be there.
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    at.button[0].click().run()
    assert "Dataset overview" in [h.value for h in at.subheader]

    at.run()  # the unrelated rerun

    assert at.exception == []
    assert "Dataset overview" in [h.value for h in at.subheader]
    assert not any("Upload a file" in b.value for b in at.info)


def test_sample_dataset_survives_several_reruns():
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    at.button[0].click().run()

    for _ in range(3):
        at.run()

    assert at.exception == []
    assert "Dataset overview" in [h.value for h in at.subheader]
