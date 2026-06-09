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


class _WordModel:
    """Fake model whose tokenizer counts words and whose predict flags 'toxic'.

    Lets us exercise the chunk-and-pool path with a tiny window and no torch.
    """

    def __init__(self):
        self.calls: list[str] = []

    def tokenizer(self, text, add_special_tokens=True):
        return {"input_ids": text.split()}

    def predict(self, text):
        self.calls.append(text)
        return {"toxicity": 0.99 if "toxic" in text else 0.01}


@pytest.fixture
def word_model(monkeypatch):
    """Install the word-counting fake model with a tiny (5-token) window."""
    model = _WordModel()
    monkeypatch.setattr(classifier, "_model", model)
    monkeypatch.setattr(classifier, "MODEL_WINDOW", 5)
    return model


def test_short_text_scored_in_a_single_pass(word_model):
    """Text within the window scores directly — one predict call, no chunking."""
    scores = classifier._predict_pooled_sync("you are clean here")
    assert scores["toxicity"] == 0.01
    assert len(word_model.calls) == 1


def test_long_text_pools_toxic_tail_via_max(word_model):
    """A toxic sentence past the window is recovered as the per-attribute max.

    Without pooling this trailing sentence would be truncated and the score would
    read ~clean; pooling over chunks must surface the 0.99.
    """
    # 18 words over a 5-token window; the toxic content is in the final sentence.
    text = "all is calm and fine here. nothing wrong at all really. you are toxic now friend."
    scores = classifier._predict_pooled_sync(text)
    assert scores["toxicity"] == 0.99
    assert len(word_model.calls) > 1  # proves it chunked rather than truncated
