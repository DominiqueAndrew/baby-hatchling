from src.utils.data import load_text_splits


def test_load_text_splits_from_local_jsonl(tmp_path):
    shard = tmp_path / "shard.jsonl"
    shard.write_text('{"text": "hello world"}\n', encoding="utf8")
    specs = [{"path": str(shard), "name": "local_shard"}]
    samples = load_text_splits(specs)
    assert len(samples) == 1
    assert samples[0].text == "hello world"
    assert samples[0].source == "local_shard"
