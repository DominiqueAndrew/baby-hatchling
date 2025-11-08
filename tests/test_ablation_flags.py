import torch

from src.model import BabyHatchlingModel, ModelConfig


def make_cfg(**kwargs):
    defaults = dict(
        name="test",
        d_model=16,
        n_layers=4,
        n_heads=2,
        dk=4,
        dv=4,
        d_ff=32,
        rank_gate=4,
        group_kv=1,
        gw_tokens=0,
        max_seq=32,
        vocab_size=32,
        episodic_bytes=1024,
    )
    defaults.update(kwargs)
    return ModelConfig(**defaults)


def test_ablation_disables_predictive_and_memory():
    cfg = make_cfg(use_predictive_head=False, use_episodic_memory=False, use_curiosity_bonus=False)
    model = BabyHatchlingModel(cfg, episodic_path="/tmp/episodic.db")
    assert not model.pred_head.enabled
    assert model.memory is None
    tokens = torch.randint(0, cfg.vocab_size, (1, 4))
    pred_out, _ = model(tokens)
    assert pred_out.pc_loss.item() == 0.0
    assert float(model.episodic_gate_penalty()) == 0.0
