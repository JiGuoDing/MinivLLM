import torch 
import torch.nn as nn
import torch.nn.functional as F
import time

class SiluAndMul(nn.Module):
    """
    A custom activation layer that applies the SiLU (Sigmoid Linear Unit) activation
    function followed by element-wise multiplication with the input tensor.
    """

    def __init__(self):
        super().__init__()

    # @torch.compile 把 PyTorch 的动态图代码在运行时编译成更高效的执行版本
    # 对于较小的张量，编译可能反而会增加开销，因此在实际使用中需要根据具体情况进行测试和调整。
    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 将输入张量 x 沿最后一个维度一分为二，得到 x 和 y 两个子张量
        x, y = x.chunk(2, -1)
        # 对 x 应用 SiLU 激活函数，再与 y 做逐元素相乘
        return F.silu(x) * y
        # SiLU 激活函数的定义为：SiLU(x) = x * sigmoid(x)，其中 sigmoid(x) = 1 / (1 + exp(-x))。

'''
这个 main 函数的作用是测试 SiluAndMul 层的性能。它首先创建一个 SiluAndMul 层并将其移动到 GPU 上。然后，它生成一个随机输入张量，并进行多次前向传播以进行热身。接下来，它测量了 100 次前向传播的时间，并计算了平均推理时间，最后将结果打印出来。

具体测了这些内容：

构建并放到 CUDA
实例化 SiluAndMul 层，并移动到 GPU。

生成固定形状输入
创建一个随机输入张量，形状是 (8, 4000, 8000)，也放到 GPU。

预热（warm-up）10 次
先跑 10 次前向，目的是让 CUDA 内核、可能的编译优化（比如 torch.compile 触发的图编译）先稳定下来，避免把初始化开销算进正式计时。

正式计时 100 次
每次前向前后都调用 torch.cuda.synchronize()，确保计时只覆盖真实计算时间（CUDA 默认异步，不同步会计不准）。
记录每次耗时，最后求平均值。

输出平均延迟
打印 100 次的平均推理时间（毫秒）。
'''
if __name__ == "__main__":
    # Example usage
    layer = SiluAndMul().cuda()
    # input_tensor = torch.randn(400, 800).cuda()  # Example input tensor with shape (400, 800)
    # input_tensor = torch.randn(4000, 8000).cuda()  # Example input tensor with shape (4000, 8000)
    input_tensor = torch.randn(8, 4000, 8000).cuda()  # Example input tensor with shape (8, 4000, 8000)
    
    for _ in range(10):  # Warm-up iterations
        _ = layer(input_tensor)

    times = []
    for _ in range(100):  # Timing iterations
        # torch.cuda.synchronize() 让 CPU 阻塞等待，确保所有之前的 CUDA 操作完成 (直到当前 CUDA 设备上的所有已提交任务都执行完)，避免计时不准确
        torch.cuda.synchronize()
        start_time = time.time()
        output_tensor = layer(input_tensor)
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    print(f"Average inference time over 100 runs: {avg_time * 1000:.4f} ms")

'''
为什么需要 torch.cuda.synchronize()？

CUDA 默认是异步执行
你在 Python 里调用 GPU 算子时，通常只是“把任务提交给 GPU 队列”，函数很快就返回了。
如果直接用 time.time() 包住 GPU 调用，不加同步，测到的大多是“提交开销”，不是“实际计算完成时间”。

保证计时准确
在计时前 synchronize()：确保前面的任务不影响本次测量。
在计时后 synchronize()：确保本次任务真正执行完再停止计时。
这样得到的是该段 GPU 计算的真实墙钟耗时。

调试时也常用
某些 CUDA 错误会延迟到同步点才抛出，所以它也有助于定位问题。

代价：
会打断 CPU/GPU 的异步并行，频繁调用会降低吞吐。
所以通常只在基准测试、精确 profiling、调试时使用。
'''
