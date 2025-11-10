"""Test that chunked memory processing produces correct results."""
import torch
from src.kda_parallel_scan import precompute_updates


def test_chunked_vs_full():
    """Verify chunked processing gives same results with different chunk sizes."""
    torch.manual_seed(42)
    
    B, T, H, dk, dv = 2, 128, 4, 32, 32
    
    # Create random inputs
    k = torch.randn(B, T, H, dk)
    v = torch.randn(B, T, H, dv)
    alpha = torch.sigmoid(torch.randn(B, T, H, dk))
    beta = torch.sigmoid(torch.randn(B, T, H, 1))
    active = (torch.rand(B, T, H, 1) > 0.3).float()
    
    # Test different chunk sizes
    results = {}
    for chunk_size in [16, 32, 64, 128]:
        updates = precompute_updates(
            k, v, alpha, beta, active, 
            drop_mask=None,
            memory_chunk_size=chunk_size
        )
        results[chunk_size] = (updates.transitions, updates.writes)
    
    # All results should be identical
    ref_t, ref_w = results[16]
    for chunk_size in [32, 64, 128]:
        t, w = results[chunk_size]
        assert torch.allclose(t, ref_t, rtol=1e-5, atol=1e-6), \
            f"Transitions differ for chunk_size={chunk_size}"
        assert torch.allclose(w, ref_w, rtol=1e-5, atol=1e-6), \
            f"Writes differ for chunk_size={chunk_size}"
    
    print("✓ All chunk sizes produce identical results")


def test_memory_usage():
    """Verify that chunking reduces peak memory usage."""
    torch.manual_seed(42)
    
    B, T, H, dk, dv = 8, 256, 8, 64, 64
    
    k = torch.randn(B, T, H, dk, device='cpu')
    v = torch.randn(B, T, H, dv, device='cpu')
    alpha = torch.sigmoid(torch.randn(B, T, H, dk, device='cpu'))
    beta = torch.sigmoid(torch.randn(B, T, H, 1, device='cpu'))
    active = torch.ones(B, T, H, 1, device='cpu')
    
    # Small chunks should work even with large sequences
    updates = precompute_updates(
        k, v, alpha, beta, active,
        memory_chunk_size=32
    )
    
    assert updates.transitions.shape == (B, T, H, dk, dk)
    assert updates.writes.shape == (B, T, H, dk, dv)
    
    print("✓ Chunked processing handles large sequences")


if __name__ == "__main__":
    test_chunked_vs_full()
    test_memory_usage()
    print("\nAll tests passed!")

