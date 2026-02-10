import argparse
import time
import math
import torch
import torch.nn.functional as F

from flashattn import flash_attention_forward_v1, flash_attention_forward_v2, flash_attention_forward_v3

def make_inputs(B=4, H=8, N=128, D=8, dtype=torch.float16, device="cuda"):
    q = torch.randn(B, H, N, D, dtype=dtype, device=device).contiguous()
    k = torch.randn(B, H, N, D, dtype=dtype, device=device).contiguous()
    v = torch.randn(B, H, N, D, dtype=dtype, device=device).contiguous()
    return q, k, v

def attn_matmul(q, k, v, dropout_p=0.0):
    # scores: (B, H, N, N)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    attn_weights = F.softmax(scores, dim=-1)
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)
    output = torch.matmul(attn_weights, v)
    return output

def attn_torch_sdp(q, k, v):
    # F.scaled_dot_product_attention expects [..., T, D]
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

def compute_errors(ref_output, test_output, name="My Kernel"):
    ref = ref_output.cpu().float()
    test = test_output.cpu().float()

    abs_err = torch.abs(ref - test)
    rel_err = abs_err / (torch.abs(ref) + 1e-8)
    
    max_abs_err = abs_err.max().item()
    max_rel_err = rel_err.max().item()
    
    mean_abs_err = abs_err.mean().item()
    mean_rel_err = rel_err.mean().item()
    
    mse = torch.mean((ref - test) ** 2)
    rmse = torch.sqrt(mse).item()
    
    sdp_norm = torch.norm(ref)
    diff_norm = torch.norm(ref - test)
    if sdp_norm > 0:
        rel_diff = (diff_norm / sdp_norm).item() * 100 
    else:
        rel_diff = 0.0
    
    print(f"\n📊 {name} 误差分析:")
    print(f"  Max Absolute Error:   {max_abs_err:.6e}")
    print(f"  Max Relative Error:   {max_rel_err:.6e}")
    print(f"  Mean Absolute Error:  {mean_abs_err:.6e}")
    print(f"  Mean Relative Error:  {mean_rel_err:.6e}")
    print(f"  RMSE:                {rmse:.6e}")
    print(f"  Relative Diff (%):   {rel_diff:.4f}%")
    
    # 检查NaN和Inf
    has_nan = torch.isnan(test).any().item()
    has_inf = torch.isinf(test).any().item()
    if has_nan or has_inf:
        print(f"  ⚠️ 警告: 输出包含 {'NaN' if has_nan else ''}{' and ' if has_nan and has_inf else ''}{'Inf' if has_inf else ''}")
    
    return {
        'max_abs_err': max_abs_err,
        'max_rel_err': max_rel_err,
        'mean_abs_err': mean_abs_err,
        'mean_rel_err': mean_rel_err,
        'rmse': rmse,
        'rel_diff_pct': rel_diff,
        'has_nan': has_nan,
        'has_inf': has_inf
    }

@torch.no_grad()
def bench(fn, q, k, v, iters=50, warmup=10, name="Function"):
    # warmup
    for _ in range(warmup):
        fn(q, k, v)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        output = fn(q, k, v)
    torch.cuda.synchronize()
    t1 = time.time()
    
    avg_time = (t1 - t0) * 1000 / iters  # ms
    return output, avg_time

def run_bench(B=4, H=8, N=128, D=64, dtype=torch.float16, iters=100):
    """运行完整的基准测试和误差分析"""
    print(f"\n{'='*60}")
    print(f"测试配置: B={B}, H={H}, N={N}, D={D}, dtype={dtype}")
    print(f"{'='*60}")
    
    # 生成输入数据（固定随机种子以便可重复）
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    q = torch.randn(B, H, N, D, device="cuda", dtype=dtype).contiguous()
    k = torch.randn(B, H, N, D, device="cuda", dtype=dtype).contiguous()
    v = torch.randn(B, H, N, D, device="cuda", dtype=dtype).contiguous()
    
    # 保存输入用于调试
    inputs = (q.clone(), k.clone(), v.clone())
    
    print("\n⏱️  性能测试:")
    
    # 测试朴素矩阵乘法实现
    output_mm, ms_mm = bench(
        lambda a,b,c: attn_matmul(a,b,c), 
        q, k, v, 
        iters=min(iters, 20),  # 朴素实现较慢，减少迭代次数
        name="Matmul Attention"
    )
    
    # 测试PyTorch SDP实现
    output_sdp, ms_sdp = bench(
        lambda a,b,c: attn_torch_sdp(a,b,c),
        q, k, v,
        iters=iters,
        name="Torch SDP"
    )
    
    # 测试你的CUDA实现
    try:
        output_my, ms_my = bench(
            lambda a,b,c: flash_attention_forward_v1(a,b,c),
            q, k, v,
            iters=iters,
            name="My Flash Attention v1"
        )
    except Exception as e:
        print(f"❌ My Flash Attention v1 实现失败: {e}")   
        output_my = None
        ms_my = float('inf')
    try:
        output_my2, ms_my2 = bench(
            lambda a,b,c: flash_attention_forward_v2(a,b,c),
            q, k, v,
            iters=iters,
            name="My Flash Attention v2"
        )
    except Exception as e:
        print(f"❌ My Flash Attention v2 实现失败: {e}")   
        output_my2 = None
        ms_my2 = float('inf')
    try:
        output_my3, ms_my3 = bench(
            lambda a,b,c: flash_attention_forward_v3(a,b,c),
            q, k, v,
            iters=iters,
            name="My Flash Attention v3"
        )
    except Exception as e:
        print(f"❌ My Flash Attention v3 实现失败: {e}")   
        output_my3 = None
        ms_my3 = float('inf')
    
    print(f"\n📈 性能结果:")
    print(f"  Matmul Attention:      {ms_mm:.3f} ms")
    print(f"  Torch SDP Attention:   {ms_sdp:.3f} ms")
    print(f"  My Flash Attention v1: {ms_my:.3f} ms")
    print(f"  My Flash Attention v2: {ms_my2:.3f} ms")
    print(f"  My Flash Attention v3: {ms_my3:.3f} ms")
    
    if ms_my < float('inf'):
        speedup_vs_mm = ms_mm / ms_my
        speedup_vs_sdp = ms_sdp / ms_my
        print(f"\n🚀 加速比:")
        print(f"  vs Matmul:            {speedup_vs_mm:.2f}x")
        print(f"  vs Torch SDP:         {speedup_vs_sdp:.2f}x")
    if ms_my2 < float('inf'):
        speedup_vs_mm2 = ms_mm / ms_my2
        speedup_vs_sdp2 = ms_sdp / ms_my2
        print(f"\n🚀 加速比 (v2):")
        print(f"  vs Matmul:            {speedup_vs_mm2:.2f}x")
        print(f"  vs Torch SDP:         {speedup_vs_sdp2:.2f}x")
    if ms_my3 < float('inf'):
        speedup_vs_mm3 = ms_mm / ms_my3
        speedup_vs_sdp3 = ms_sdp / ms_my3
        print(f"\n🚀 加速比 (v3):")
        print(f"  vs Matmul:            {speedup_vs_mm3:.2f}x")
        print(f"  vs Torch SDP:         {speedup_vs_sdp3:.2f}x")
    
    print("\n🔍 误差分析:")
    
    # 以PyTorch SDP为参考标准
    if output_my is not None or output_my2 is not None or output_my3 is not None:
        # 检查与PyTorch SDP的误差
        if output_my is not None:
            errors_my = compute_errors(output_sdp, output_my, "My Kernel v1 vs PyTorch SDP")
        else:
            errors_my = None
        if output_my2 is not None:
            errors_my2 = compute_errors(output_sdp, output_my2, "My Kernel v2 vs PyTorch SDP")
        else:
            errors_my2 = None
        if output_my3 is not None:
            errors_my3 = compute_errors(output_sdp, output_my3, "My Kernel v3 vs PyTorch SDP")
        else:
            errors_my3 = None
        
        # 检查与朴素实现的误差（作为交叉验证）
        if output_my is not None:
            errors_mm = compute_errors(output_mm, output_my, "My Kernel v1 vs Matmul")
        else:
            errors_mm = None
        if output_my2 is not None:
            errors_mm2 = compute_errors(output_mm, output_my2, "My Kernel v2 vs Matmul")
        else:
            errors_mm2 = None
        if output_my3 is not None:
            errors_mm3 = compute_errors(output_mm, output_my3, "My Kernel v3 vs Matmul")
        else:
            errors_mm3 = None
        
        # 检查PyTorch SDP与朴素实现的误差（参考）
        errors_ref = compute_errors(output_mm, output_sdp, "PyTorch SDP vs Matmul")
        
        # 总结
        print(f"\n{'='*60}")
        print("📋 测试总结:")
        print(f"{'='*60}")
        
        # 性能总结
        if ms_my < float('inf'):
            print(f"性能 v1: {ms_my:.3f} ms (PyTorch SDP: {ms_sdp:.3f} ms)")
        if ms_my2 < float('inf'):
            print(f"性能 v2: {ms_my2:.3f} ms (PyTorch SDP: {ms_sdp:.3f} ms)")
        if ms_my3 < float('inf'):
            print(f"性能 v3: {ms_my3:.3f} ms (PyTorch SDP: {ms_sdp:.3f} ms)")
        
        # 误差总结
        if errors_my:
            if errors_my['rel_diff_pct'] < 0.1:
                print("✅ 误差 v1: 优秀 (<0.1%)")
            elif errors_my['rel_diff_pct'] < 1.0:
                print("⚠️  误差 v1: 可接受 (<1%)")
            elif errors_my['rel_diff_pct'] < 5.0:
                print("⚠️  误差 v1: 较大 (<5%)")
            else:
                print("❌ 误差 v1: 过大 (>=5%)")
        if errors_my2:
            if errors_my2['rel_diff_pct'] < 0.1:
                print("✅ 误差 v2: 优秀 (<0.1%)")
            elif errors_my2['rel_diff_pct'] < 1.0:
                print("⚠️  误差 v2: 可接受 (<1%)")
            elif errors_my2['rel_diff_pct'] < 5.0:
                print("⚠️  误差 v2: 较大 (<5%)")
            else:
                print("❌ 误差 v2: 过大 (>=5%)")
        if errors_my3:
            if errors_my3['rel_diff_pct'] < 0.1:
                print("✅ 误差 v3: 优秀 (<0.1%)")
            elif errors_my3['rel_diff_pct'] < 1.0:
                print("⚠️  误差 v3: 可接受 (<1%)")
            elif errors_my3['rel_diff_pct'] < 5.0:
                print("⚠️  误差 v3: 较大 (<5%)")
            else:
                print("❌ 误差 v3: 过大 (>=5%)")
        
        # 数值稳定性检查
        def _ns(err):
            return err and (err['has_nan'] or err['has_inf'])
        if _ns(errors_my) or _ns(errors_my2) or _ns(errors_my3):
            print("❌ 数值稳定性: 存在NaN/Inf")
        else:
            print("✅ 数值稳定性: 良好")
        
        return {
            'performance': {'my_v1': ms_my, 'my_v2': ms_my2, 'my_v3': ms_my3, 'sdp': ms_sdp, 'matmul': ms_mm},
            'errors': {'vs_sdp_v1': errors_my, 'vs_sdp_v2': errors_my2, 'vs_sdp_v3': errors_my3, 'vs_matmul_v1': errors_mm, 'vs_matmul_v2': errors_mm2, 'vs_matmul_v3': errors_mm3, 'ref': errors_ref}
        }
    else:
        return None

def run_comprehensive_test():
    """运行全面的测试套件"""
    test_cases = [
        # (B, H, N, D) - 不同规模和配置
        (2, 4, 64, 32),     # 小规模
        (4, 8, 128, 64),    # 中等规模
        # (2, 12, 256, 64),   # 长序列
        # (8, 16, 512, 64),   # 大规模
    ]
    
    results = {}
    
    for i, (B, H, N, D) in enumerate(test_cases, 1):
        print(f"\n{'#'*60}")
        print(f"测试用例 {i}/{len(test_cases)}: B={B}, H={H}, N={N}, D={D}")
        print(f"{'#'*60}")
        
        result = run_bench(B=B, H=H, N=N, D=D, dtype=torch.float16)
        if result:
            results[f"case_{i}"] = result
        
        # 给GPU一些时间冷却
        if i < len(test_cases):
            time.sleep(1)
    
    # 打印汇总报告
    if results:
        print(f"\n{'='*60}")
        print("📊 汇总报告")
        print(f"{'='*60}")
        
        avg_speedup = []
        max_errors = []
        
        for case, data in results.items():
            perf = data['performance']
            errors = data['errors']['vs_sdp']
            
            if perf['sdp'] > 0:
                speedup = perf['sdp'] / perf['my']
                avg_speedup.append(speedup)
                max_errors.append(errors['rel_diff_pct'])
            
            print(f"\n{case}:")
            print(f"  速度: {perf['my']:.3f} ms (vs SDP: {speedup:.2f}x)")
            print(f"  误差: {errors['rel_diff_pct']:.4f}%")
        
        if avg_speedup:
            print(f"\n📈 平均加速比: {sum(avg_speedup)/len(avg_speedup):.2f}x")
            print(f"📉 最大误差: {max(max_errors):.4f}%")
            print(f"📊 平均误差: {sum(max_errors)/len(max_errors):.4f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='测试Flash Attention实现')
    parser.add_argument('--B', type=int, default=4, help='Batch size')
    parser.add_argument('--H', type=int, default=8, help='Number of heads')
    parser.add_argument('--N', type=int, default=128, help='Sequence length')
    parser.add_argument('--D', type=int, default=64, help='Head dimension')
    parser.add_argument('--comprehensive', action='store_true', help='Run comprehensive test suite')
    
    args = parser.parse_args()
    
    if args.comprehensive:
        run_comprehensive_test()
    else:
        run_bench(B=args.B, H=args.H, N=args.N, D=args.D)
