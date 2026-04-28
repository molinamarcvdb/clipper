### Implementation of various clipping methods from stadard to ghost clipping to triton implmentation

import torch
import triton
import triton.language as tl
from typing import Callable

import time
import functools
import torch

from triton.testing import do_bench

def profile(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # Sync before timing — CUDA ops are async otherwise
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        peak_mb = (
            torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
        )
        print(f"[{fn.__name__}] {elapsed * 1000:.2f} ms, peak GPU mem {peak_mb:.1f} MB")
        return result

    return wrapper


# All methods share this signature:
#   inputs:  X  (B, D_in)   activations into the Linear layer
#            dY (B, D_out)  output gradients of the Linear layer
#            C  float       clip norm
#            sigma float    noise multiplier (set to 0 to disable noise for testing)
#            generator      torch.Generator for reproducible noise (optional)
#   returns: g  (D_out, D_in)  clipped, noised, summed gradient ready for optimizer


@profile
def naive_per_sample(
    X: torch.Tensor,
    dY: torch.Tensor,
    C: float,
    sigma: float,
) -> torch.Tensor:
    """Materializes the pre-sample gradient. SLow, OOMs, large shapes"""

    g = torch.einsum("bo,bi->boi", dY, X)
    norms = (g**2).sum(dim=(1, 2)).sqrt()
    c = torch.clamp(C / norms.clamp(min=1e-6), max=1.0).view(-1, 1, 1)
    g_clipped = (g * c).sum(dim=0)
    noise = torch.randn(g_clipped.shape, device=g_clipped.device, dtype=g_clipped.dtype)

    return g_clipped + sigma * C * noise


def ghost_clipping(X, dY, C, sigma):
    """Ghost trick: norm factors as ||dY_i|| * ||X_i||. No per-sample gradient ever materialized."""
    a = (X * X).sum(dim=1)  # (B,) ||x_i||^2
    b = (dY * dY).sum(dim=1)  # (B,) ||dy_i||^2
    norms = (a * b).sqrt()  # (B,) ||g_i||_F
    c = torch.clamp(C / norms.clamp(min=1e-12), max=1.0)  # (B,)
    dY_scaled = dY * c[:, None]  # (B, D_out)
    g = dY_scaled.T @ X  # (D_out, D_in)
    noise = torch.randn(g.shape, device=g.device, dtype=g.dtype)
    return g + sigma * C * noise

@triton.jit
def triton_sq_norm(X_ptr: torch.Tensor, norms_ptr: torch.Tensor, B: int, D: int, BLOCK_B: tl.constexpr, BLOCK_D: tl.constexpr, stride_b, stride_d):
    
    pid = tl.program_id(axis=0)
    sample_offsets = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    sample_masks = sample_offsets < B
    
    acc = tl.zeros((BLOCK_B,), tl.float32)

    for d_start in range(0, D, BLOCK_D):
        feature_offsets = d_start + tl.arange(0, BLOCK_D)
        feature_masks = feature_offsets < D

        ptrs = X_ptr + sample_offsets[:, None] * stride_b + feature_offsets[None, :] * stride_d
        
        mask_2d = sample_masks[:, None] & feature_masks[None, :]
        
        tile = tl.load(ptrs, mask=mask_2d, other=0.0)
        acc += tl.sum(tile*tile, axis=1)

    tl.store(norms_ptr + sample_offsets, acc, mask=sample_masks)

    # frist we need to x * x sum


@triton.jit
def triton_ghost_norm(
        X_ptr: torch.Tensor, 
        dY_ptr: torch.Tensor, 
        norms_ptr: torch.Tensor, 
        B: int, 
        D1: int,
        D2: int,
        BLOCK_B: tl.constexpr, 
        BLOCK_D1: tl.constexpr, 
        BLOCK_D2: tl.constexpr, 
        stride_b1,
        stride_b2,
        stride_d1,
        stride_d2
    ):
    
    pid = tl.program_id(axis=0)
    sample_offsets = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    sample_masks = sample_offsets < B
    
    acc_x = tl.zeros((BLOCK_B,), tl.float32)
    acc_y = tl.zeros((BLOCK_B,), tl.float32)

    for d_start in range(0, D1, BLOCK_D1):
        feature_offsets = d_start + tl.arange(0, BLOCK_D1)
        feature_masks = feature_offsets < D1

        ptrs = X_ptr + sample_offsets[:, None] * stride_b1 + feature_offsets[None, :] * stride_d1
        
        mask_2d = sample_masks[:, None] & feature_masks[None, :]
        
        tile = tl.load(ptrs, mask=mask_2d, other=0.0)
        acc_x += tl.sum(tile*tile, axis=1)

    for d_start in range(0, D2, BLOCK_D2):
        feature_offsets = d_start + tl.arange(0, BLOCK_D2)
        feature_masks = feature_offsets < D2

        ptrs = dY_ptr + sample_offsets[:, None] * stride_b2 + feature_offsets[None, :] * stride_d2
        
        mask_2d = sample_masks[:, None] & feature_masks[None, :]
        
        tile = tl.load(ptrs, mask=mask_2d, other=0.0)
        acc_y += tl.sum(tile*tile, axis=1)

    tl.store(norms_ptr + sample_offsets, acc_x*acc_y, mask=sample_masks)

@triton.jit
def triton_matmul(cdY_ptr, X_ptr, g_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Accumulator for this output tile, kept in registers
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in tiles of BLOCK_K
    for k_start in range(0, K, BLOCK_K):

        k_mask = (k_start + offs_k) < K
        a_mask = (offs_m[:, None] < M) & (k_mask[None, :])
        b_mask = (k_mask[:, None]) & (offs_n[None, :] < N)

        a_tile = tl.load(cdY_ptr + offs_m[:, None] * stride_am + (k_start + offs_k)[None, :] * stride_ak, mask=a_mask, other=0.0)
        b_tile = tl.load(X_ptr + (k_start + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn, mask=b_mask, other=0.0)

        acc += tl.dot(a_tile, b_tile)

    # Write back
    c_mask = (offs_m[:, None]<M) & (offs_n[None, :]<N)
    tl.store(g_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc, mask=c_mask)

def triton_clipping(X: torch.Tensor, dY: torch.Tensor, C: float, sigma: float):
    """Initally we will create two kernels one that computes grad norms and then another applying the actual 
        c_i to gradsient collapsin to batched gradient plus noise"""
    
    B, Din = X.shape
    _, Dout = dY.shape
    device = X.device

    norms_ptr = torch.empty((B,), device=device)
    
    BLOCK_B = 8
    BLOCK_D = 64
    BLOCK_M, BLOCK_N = 64, 64
    
    grid = (triton.cdiv(B, BLOCK_B),) 
    #triton_sq_norm[grid](X, norms_ptr, B, Din, BLOCK_B, BLOCK_D, stride_b, stride_d)
    triton_ghost_norm[grid](X, dY, norms_ptr, B, Din, Dout, BLOCK_B, BLOCK_D, BLOCK_D, X.stride(0), dY.stride(0), X.stride(1), dY.stride(1))
    norms = norms_ptr.sqrt()

    c = torch.clamp(C / norms.clamp(min=1e-12), max=1.0)
    
    cdY = dY * c[:, None]

    g = torch.zeros((Dout, Din), device=device)
    grid = (triton.cdiv(Dout, BLOCK_M), triton.cdiv(Din, BLOCK_N))
    triton_matmul[grid](cdY_ptr=cdY, X_ptr=X, g_ptr=g, M=Dout, N=Din, K=B, stride_am=cdY.stride(1), stride_ak=cdY.stride(0), 
                       stride_bk=X.stride(0), stride_bn=X.stride(1), stride_cm=g.stride(0), stride_cn=g.stride(1),
                       BLOCK_M = BLOCK_M, BLOCK_N = BLOCK_N, BLOCK_K = 32)

    g = g + sigma * C * torch.randn_like(g)
    return g

if __name__ == "__main__":

    X = torch.randn(1024, 4096, device="cuda")
    dY = torch.randn(1024, 4096, device="cuda")
    C = 1.0
    sigma = 1.0
    seed = 67

    torch.manual_seed(0)
    #out_naive = naive_per_sample(X, dY, C, sigma=0.0)
    torch.manual_seed(0)
    out_ghost = ghost_clipping(X, dY, C, sigma=0.0)
    out_triton_ghost = triton_clipping(X, dY, C, sigma=0.0)
    
    assert torch.allclose(out_triton_ghost, out_ghost, atol=1e-4), f"max diff: {(out_triton_ghost - out_ghost).abs().max().item()}"

    ms = do_bench(lambda: triton_clipping(X, dY, C=1.0, sigma=0.0))
    print(f"triton: {ms:.3f} ms")

    ms = do_bench(lambda: ghost_clipping(X, dY, C, sigma=0.0))
    print(f"pytorch: {ms:.3f} ms")
