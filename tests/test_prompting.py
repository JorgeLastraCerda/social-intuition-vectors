import pytest
import torch

from src.utils.prompting import (
    decision_token_ids,
    encode_decision_prompt,
    encode_passage,
)


class FakeProcessor:
    def apply_chat_template(self, messages, **kwargs):
        assert messages == [{"role": "user", "content": "Decide."}]
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return "<bos><user>Decide.</user><model>"


class FakeModel:
    processor = FakeProcessor()

    def to_tokens(self, text, prepend_bos):
        if text == "passage":
            assert prepend_bos is True
            return torch.tensor([[9, 10]])
        mapping = {
            "<bos><user>Decide.</user><model>": [1, 2, 3],
            "<bos><user>Decide.</user><model>Yes": [1, 2, 3, 7],
            "<bos><user>Decide.</user><model>No": [1, 2, 3, 8],
        }
        assert prepend_bos is False
        return torch.tensor([mapping[text]])


def test_passage_encoding_remains_raw_with_bos():
    torch.testing.assert_close(encode_passage(FakeModel(), "passage"), torch.tensor([[9, 10]]))


def test_native_chat_uses_exact_prefix_for_candidate_tokens():
    model = FakeModel()
    rendered, tokens = encode_decision_prompt(model, "Decide.", "native-chat")
    assert rendered.endswith("<model>")
    torch.testing.assert_close(tokens, torch.tensor([[1, 2, 3]]))
    assert decision_token_ids(model, rendered, "native-chat") == (7, 8)


def test_native_chat_fails_without_processor():
    model = FakeModel()
    model.processor = None
    with pytest.raises(ValueError, match="native-chat requires"):
        encode_decision_prompt(model, "Decide.", "native-chat")
