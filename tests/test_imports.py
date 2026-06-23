"""Smoke test: verify the public API is fully importable.

This is the minimum gate that proves CI wiring is correct.  If this test
fails, something is wrong with the package install, not the logic.
"""
import text_classifier


def test_public_api_importable() -> None:
    for name in text_classifier.__all__:
        assert hasattr(text_classifier, name), (
            f"text_classifier.{name} is listed in __all__ but not importable"
        )


def test_all_is_nonempty() -> None:
    assert len(text_classifier.__all__) > 0
