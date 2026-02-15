import os
import argparse
import math
import time
import torch
import torch.nn.functional as F

from flashattn import flash_attention_forward_v5


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_inputs(B, H, N, D, dtype, device, seed=42):
    set_seed(seed)
    q = torch.randn(B, H, N, D, dtype=dtype, device=device).contiguous()
    k = torch.randn(B, H, N, D, dtype=dtype, device=device).contiguous()
    v = torch.randn(B, H, N, D, dtype=dtype, device=device).contiguous()
    return q, k, v


def attn_torch_sdp(q, k, v):
    # expects [..., T, D] with q,k,v shape (B,H,N,D) is fine
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)


def attn_matmul(q, k, v):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def tensor_info(x, name: str):
    ptr = x.data_ptr()
    print(f"\n[{name}]")
    print(f"  shape:   {tuple(x.shape)}")
    print(f"  dtype:   {x.dtype}")
    print(f"  device:  {x.device}")
    print(f"  contig:  {x.is_contiguous()}")
    print(f"  strides: {x.stride()}")
    print(f"  ptr:     0x{ptr:x}")
    # 常见 vectorized load/store 对齐：16B（uint4）/ 8B / 4B
    print(f"  align16: {ptr % 16 == 0}  (ptr%16={ptr%16})")
    print(f"  align8:  {ptr % 8  == 0}  (ptr%8 ={ptr%8})")
    print(f"  align4:  {ptr % 4  == 0}  (ptr%4 ={ptr%4})")


def finite_stats(x, name: str):
    x_f = x.float()
    isnan = torch.isnan(x_f)
    isinf = torch.isinf(x_f)
    n_nan = int(isnan.sum().item())
    n_inf = int(isinf.sum().item())
    print(f"\n[{name} finite check]")
    print(f"  nan: {n_nan} / {x.numel()}")
    print(f"  inf: {n_inf} / {x.numel()}")
    if n_nan or n_inf:
        bad = isnan | isinf
        first = bad.flatten().nonzero(as_tuple=False)[0].item()
        idx = list(torch.unravel_index(torch.tensor(first), x.shape))
        val = x_f.flatten()[first].item()
        print(f"  first bad at flat={first}, idx={idx}, val={val}")
    return (n_nan == 0 and n_inf == 0)


def error_report(ref, out, name=""):
    ref_f = ref.float().cpu()
    out_f = out.float().cpu()
    abs_err = (ref_f - out_f).abs()
    rel_err = abs_err / (ref_f.abs() + 1e-8)

    max_abs = abs_err.max().item()
    mean_abs = abs_err.mean().item()
    max_rel = rel_err.max().item()
    mean_rel = rel_err.mean().item()
    rmse = torch.sqrt(torch.mean((ref_f - out_f) ** 2)).item()

    # norm diff (%)
    ref_norm = torch.norm(ref_f).item()
    diff_norm = torch.norm(ref_f - out_f).item()
    rel_diff_pct = (diff_norm / (ref_norm + 1e-12)) * 100.0

    print(f"\n[Error vs SDP] ({name})")
    print(f"  max_abs: {max_abs:.6e}")
    print(f"  mean_abs:{mean_abs:.6e}")
    print(f"  max_rel: {max_rel:.6e}")
    print(f"  mean_rel:{mean_rel:.6e}")
    print(f"  rmse:    {rmse:.6e}")
    print(f"  rel_diff:{rel_diff_pct:.6f}%")
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "max_rel": max_rel,
        "mean_rel": mean_rel,
        "rmse": rmse,
        "rel_diff_pct": rel_diff_pct,
    }


@torch.no_grad()
def run_once(B, H, N, D, dtype=torch.float16, seed=42, device="cuda",
             do_sdp=True, do_mm=False, dump_path="debug_dump.pt"):
    print("\n" + "=" * 80)
    print(f"DEBUG  config: B={B}, H={H}, N={N}, D={D}, dtype={dtype}, seed={seed}")
    print("=" * 80)

    q, k, v = make_inputs(B, H, N, D, dtype=dtype, device=device, seed=seed)

    # 基本检查（很多 vectorized kernel 假设最后一维 contiguous）
    tensor_info(q, "q")
    tensor_info(k, "k")
    tensor_info(v, "v")

    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous(), "Inputs must be contiguous"
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim must be contiguous"
    assert q.dtype == dtype and k.dtype == dtype and v.dtype == dtype, "dtype mismatch"
    assert q.is_cuda and k.is_cuda and v.is_cuda, "must be CUDA tensors"

    # 参考输出
    ref = None
    if do_sdp:
        ref = attn_torch_sdp(q, k, v)
        torch.cuda.synchronize()
        finite_stats(ref, "sdp_out")

    if do_mm:
        mm = attn_matmul(q, k, v)
        torch.cuda.synchronize()
        finite_stats(mm, "matmul_out")

    # v5 输出
    t0 = time.time()
    out = flash_attention_forward_v5(q, k, v)
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"\n time: {(t1 - t0) * 1000:.3f} ms (single run)")

    ok = finite_stats(out, "v5_out")

    # 如果有参考，算误差（即使 out 有 NaN，也会在 float 转换时体现）
    metrics = None
    if ref is not None:
        metrics = error_report(ref, out, name="v5")

    # 如果发现 NaN/Inf：保存可复现 dump
    if not ok:
        print(f"\n!!! v5 produced NaN/Inf. Saving dump to: {dump_path}")
        payload = {
            "config": {"B": B, "H": H, "N": N, "D": D, "dtype": str(dtype), "seed": seed},
            "q": q.detach().cpu(),
            "k": k.detach().cpu(),
            "v": v.detach().cpu(),
            "kernel_out": out.detach().cpu(),
        }
        if ref is not None:
            payload["sdp_out"] = ref.detach().cpu()
        torch.save(payload, dump_path)
        print("Dump saved. You can reload with torch.load and reproduce exactly.")

    return ok, metrics


def scan_min_fail(dtype=torch.float16, seed=42, device="cuda"):
    """
    扫一批小尺寸，找第一个会炸的（最小复现）。
    建议配合 CUDA_LAUNCH_BLOCKING=1 / compute-sanitizer 使用。
    """
    # 只扫一些常见组合，避免太慢
    B_list = [1, 2]
    H_list = [1, 2, 4, 8]
    N_list = [16, 32, 64, 128, 256]
    D_list = [32, 64]   # 你当前是 64

    for B in B_list:
        for H in H_list:
            for N in N_list:
                for D in D_list:
                    try:
                        ok, _ = run_once(B, H, N, D, dtype=dtype, seed=seed, device=device,
                                         do_sdp=True, do_mm=False,
                                         dump_path=f"debug_dump_B{B}_H{H}_N{N}_D{D}.pt")
                        if not ok:
                            print(f"\n>>> FOUND FAILING CASE: B={B},H={H},N={N},D={D}")
                            return
                    except Exception as e:
                        print(f"\n>>> EXCEPTION at B={B},H={H},N={N},D={D}: {e}")
                        return
    print("\nScan finished: no NaN/Inf found in scanned cases.")


def main():
    parser = argparse.ArgumentParser("debug ")
    parser.add_argument("--B", type=int, default=8)
    parser.add_argument("--H", type=int, default=16)
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_sdp", action="store_true", help="do not run torch SDP ref")
    parser.add_argument("--mm", action="store_true", help="also run matmul ref (slow)")
    parser.add_argument("--dump", type=str, default="debug_dump.pt")
    parser.add_argument("--scan", action="store_true", help="scan small shapes to find minimal failing case")
    parser.add_argument("--sync", action="store_true", help="set CUDA_LAUNCH_BLOCKING=1")
    args = parser.parse_args()

    if args.sync:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        print("Set CUDA_LAUNCH_BLOCKING=1")

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    if args.scan:
        scan_min_fail(dtype=dtype, seed=args.seed)
        return

    run_once(
        B=args.B, H=args.H, N=args.N, D=args.D,
        dtype=dtype, seed=args.seed,
        do_sdp=not args.no_sdp,
        do_mm=args.mm,
        dump_path=args.dump,
    )


if __name__ == "__main__":
    main()
