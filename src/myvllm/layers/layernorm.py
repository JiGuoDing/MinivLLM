import torch
import time 

class LayerNorm(torch.nn.Module):
    def __init__(self, gamma: torch.Tensor, eps: float = 1e-5):
        super().__init__()
        # Use nn.Parameter to make gamma learnable and loadable from checkpoints
        self.weight = torch.nn.Parameter(gamma.detach().clone())
        self.eps = eps

    @property
    def gamma(self):
        """Backward compatibility: gamma alias for weight"""
        return self.weight

    @torch.compile
    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm (Root Mean Square Normalization，均方根归一化)
        # RMS(x) = sqrt(mean(x² + ε))
        # RMSNorm(x) = (x / RMS(x)) ⊙ γ
        # γ 是一个可学习的缩放参数。
        # ε 是一个小常数，防止除以零。
        variance = x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        sqrt_variance = variance.sqrt()
        x_norm = (x / sqrt_variance * self.weight)

        return x_norm

    # 带残差的前向传播，先将输入 x 与残差 residual 相加，然后再进行 RMSNorm 计算。
    def residual_rms_forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        x = x + residual
        return self.rms_forward(x), x

    # residual 表示残差
    # 进入第一个 Decoder Layer 时，x 是没有残差的，也就是说 residual 是 None，此时仅进行 RMSNorm 计算。
    # 每个子层都不应该完全覆盖上一层的信息，而是只学习一个增量。也就是把子层输出看成对原表示的修正项
    # 因此需要有这么一个加残差的过程，先相加，表示“保留旧信息，再叠加新信息”。
    # 如果不加，后面的层只能看到当前子层的新输出，前面累积的信息链就被切断了。
    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        if residual is not None:
            return self.residual_rms_forward(x, residual)
        else:
            return self.rms_forward(x)

if __name__ == "__main__":
    # Example usage
    # x = torch.randn(400,800).cuda()
    x = torch.randn(4000,8000).cuda()
    # x = torch.randn(8,4000,8000).cuda()
    # gamma 在 RMSNorm 里表示 “对每个 hidden channel 单独缩放的参数”
    # 必须与 x 的最后一个维度，也就是 hidden_size 对齐
    # gamma = torch.full((800,), 0.5, device="cuda", dtype=x.dtype)
    gamma = torch.full((8000,), 0.5, device="cuda", dtype=x.dtype)
    layer = LayerNorm(gamma=gamma).cuda()
    residual = torch.full_like(x,fill_value=1)

    for _ in range(10): # Warm-up iterations
        _ = layer(x)
    
    # Without residuals
    times = [] 
    for _ in range(100): # Timing iterations
        torch.cuda.synchronize()
        start_time = time.time()
        _ = layer(x)
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    print(f"[Without residuals] Average inference time over 100 runs: {avg_time * 1000:.4f} ms")

    # With residuals
    times.clear()
    for _ in range(100): # Timing iterations
        torch.cuda.synchronize()
        start_time = time.time()
        _ = layer(x,residual)
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    print(f"[With residuals] Average inference time over 100 runs: {avg_time * 1000:.4f} ms")
    
