"""Tests for classifier model loading (built-in variants vs checkpoint paths).

Detoxify/torch are never imported for real; a fake ``detoxify`` module is injected
into ``sys.modules`` so ``load_model`` exercises its routing logic offline.
"""

import sys
import types

import pytest

from app.services import classifier


class _FakeDetoxify:
    """Records the constructor args so tests can assert how it was invoked."""

    last_args: tuple = ()
    last_kwargs: dict = {}

    def __init__(self, *args, **kwargs):
        type(self).last_args = args
        type(self).last_kwargs = kwargs

    def predict(self, text):
        return {"toxicity": 0.5}


@pytest.fixture
def fake_detoxify(monkeypatch):
    """Inject a fake detoxify module and reset classifier globals after the test."""
    module = types.ModuleType("detoxify")
    module.Detoxify = _FakeDetoxify
    monkeypatch.setitem(sys.modules, "detoxify", module)
    # Captured as None now, so monkeypatch restores them to None on teardown.
    monkeypatch.setattr(classifier, "_model", None)
    monkeypatch.setattr(classifier, "_model_variant", None)
    _FakeDetoxify.last_args = ()
    _FakeDetoxify.last_kwargs = {}
    return _FakeDetoxify


def test_load_builtin_variant(fake_detoxify):
    classifier.load_model("multilingual")
    assert fake_detoxify.last_args == ("multilingual",)
    assert classifier.model_name() == "multilingual"
    assert classifier.is_loaded()


def test_load_checkpoint_path(fake_detoxify):
    classifier.load_model("/models/finetuned.ckpt")
    # Loaded via the checkpoint kwarg, not as a built-in variant.
    assert fake_detoxify.last_kwargs.get("checkpoint") == "/models/finetuned.ckpt"
    assert classifier.model_name() == "custom:finetuned.ckpt"


def test_load_unknown_variant_raises(fake_detoxify):
    with pytest.raises(ValueError):
        classifier.load_model("not-a-real-variant")
