from unittest import mock

from src.utils.data import _render_template, load_text_splits


def test_render_template_missing_fields_are_blank():
    template = "Instruction: {instruction}\nInput: {input}\nResponse: {output}"
    example = {"instruction": "Add", "output": "2"}
    rendered = _render_template(template, example)
    assert "Add" in rendered
    assert "Response: 2" in rendered
    assert "Input: " in rendered  # missing field replaced with empty string


def test_load_text_splits_honors_config_and_trust(monkeypatch):
    fake_dataset = [{"text": "hello"}]

    def fake_load_dataset(hf_id, config=None, split="train", trust_remote_code=False):
        assert hf_id == "Salesforce/wikitext"
        assert config == "wikitext-2-v1"
        assert split == "train"
        assert trust_remote_code is True
        return fake_dataset

    with mock.patch("src.utils.data.load_dataset", side_effect=fake_load_dataset):
        samples = load_text_splits(
            [
                {
                    "hf_id": "Salesforce/wikitext",
                    "config": "wikitext-2-v1",
                    "split": "train",
                    "name": "wikitext2",
                    "trust_remote_code": True,
                }
            ]
        )
    assert samples[0].text == "hello"
