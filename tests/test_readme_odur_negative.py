from pathlib import Path


README = Path(__file__).resolve().parents[1] / "README.md"


def test_readme_leads_with_odur_method_and_terminal_negative_result():
    text = README.read_text()
    assert "CrashBench: ODUR Negative Result" in text
    assert "Option-Conditioned Distributional Utility Router" in text
    assert "SCOPED_TEST_SCOPE_FAILURE_RELEASE" in text
    assert "31 sources" in text and "only 1 source" in text
    assert "required at least 3 controls" in text
    assert "100% of decisions" in text
    assert "No method superiority test was opened" in text


def test_readme_is_bilingual_and_preserves_historical_glass_scope():
    text = README.read_text()
    assert "核心方法" in text and "为什么失败" in text
    assert "Knowing When to Intervene" in text
    assert "Always Retreat" in text
    assert "PUBLICATION_FIRST_RESOLUTION.md" in text
    assert "does not rewrite or inflate that frozen glass evidence" in text


def test_readme_forbids_inflated_claims_and_links_release():
    text = README.read_text()
    assert "does **not** claim" in text
    assert "successful, superior, or deployable ODUR method" in text
    assert "general multi-mechanism VLA-safety benchmark" in text
    assert "release_audit.json" in text
