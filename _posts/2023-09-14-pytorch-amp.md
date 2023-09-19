---
title: pytorch训练优化-自动混合精度训练（AMP）
date: 2023-08-16 +0800
categories: [训练技巧, 训练优化]
tags: [训练优化]
---

Pytorch 版本：1.6及以上的版本，支持CUDA
GPU版本：支持 Tensor core的 CUDA（Volta、Turing、Ampere），在较早版本的GPU（Kepler、Maxwell、Pascal）提升一般

PyTorch 通常在 32 位浮点数据 (FP32) 上进行训练，如果你创建一个Tensor， 默认类型都是 `torch.FloatTensor` (32-bit floating point)。

NVIDIA 的工程师开发了混合精度训练（AMP），让少量操作在 FP32 中的训练，而大部分网络在 FP16 中运行，因此可以节省时间和内存。

 `torch.cuda.amp` 提供了混合精度的便捷方法，其中某些操作使用 FP32 ，其他操作使用 FP16。神经网络训练过程中的运算主要可以分三类：
- 可以受益于 FP16 速度提升的数学函数。包括矩阵乘法（线性层）和卷积。
- 对于 16 位精度可能不够的函数，输入应采用 FP32。例如减法。
- 其他操作，可以在 FP16 中运行的函数，但在 FP16 中加速并不显着，因此它们的 FP32 -> FP16 转换不值得。

混合精度训练将每个操作与其适当的数据类型相匹配，这可以减少网络的运行时间和内存占用。

>**bfloat16 vs. float16**
bfloat16 是一种z专门用于深度学习的 16 位浮点格式，由 1 个符号位、8 个指数位和 7 个尾数位组成。而行业标准 IEEE 16 位浮点是1 个符号位、5 个指数位和 10 个尾数位。
实验表明使用 bfloat16 可以提高训练效率，因为深度学习模型通常对指数变化更加敏感，而16位使用内存更少。
bfloat16 的指数位和 float32 一样，在训练过程中不容易出现下溢，也就不容易出现 NaN 或者 Inf 之类的错误。
使用 bfloat16： `dtype=torch.bfloat16`


## 一、一般的训练流程

通常自动混合精度训练会同时使用 `torch.autocast` 和 `torch.cuda.amp.GradScaler`。

假设我们已经定义好了一个模型， 并写好了其他相关代码（懒得写出来了）。

**1. torch.autocast**
 `torch.autocast` 实例作为上下文管理器，允许脚本区域以混合精度运行。
在这些区域中，CUDA 操作将以 autocast 选择的 dtype 运行，以提高性能，同时保持准确性。

autocast应该只封装前向和 loss 计算， 在 backward() 前退出 autocast，反向计算时数据类型和前向的数据类型一致。

训练部分的代码：
```
for epoch in range(epochs): 
    for input, target in zip(data, targets):
        # 在 ``autocast`` 下进行前向
        with torch.autocast(device_type=device, dtype=torch.float16):
            output = net(input)
            # output is float16 because linear layers ``autocast`` to float16.
          
            loss = loss_fn(output, target)
            # loss is float32 because ``mse_loss`` layers ``autocast`` to float32.
           
        # 在 backward() 前退出``autocast``
        # 不建议在“autocast”下进行反向传递
        # Backward 在相应前向操作选择的相同“dtype”“autocast”中运行。
        loss.backward()
        opt.step()
        opt.zero_grad() # set_to_none=True here can modestly improve performance
```

可以 autocast 到 FP16 的 CUDA 操作：
`__matmul__, addbmm, addmm, addmv, addr, baddbmm, bmm, chain_matmul, multi_dot, conv1d, conv2d, conv3d, conv_transpose1d, conv_transpose2d, conv_transpose3d, GRUCell, linear, LSTMCell, matmul, mm, mv, prelu, RNNCell`

autocast 到 FP32 的 CUDA 操作：
`__pow__, __rdiv__, __rpow__, __rtruediv__, acos, asin, binary_cross_entropy_with_logits, cosh, cosine_embedding_loss, cdist, cosine_similarity, cross_entropy, cumprod, cumsum, dist, erfinv, exp, expm1, group_norm, hinge_embedding_loss, kl_div, l1_loss, layer_norm, log, log_softmax, log10, log1p, log2, margin_ranking_loss, mse_loss, multilabel_margin_loss, multi_margin_loss, nll_loss, norm, normalize, pdist, poisson_nll_loss, pow, prod, reciprocal, rsqrt, sinh, smooth_l1_loss, soft_margin_loss, softmax, softmin, softplus, sum, renorm, tan, triplet_margin_loss`

应该优先选择 `binary_cross_entropy_with_logits` 而不是 `binary_cross_entropy`，因为:

`torch.nn.function.binary_cross_entropy()`（以及包装它的`torch.nn.BCELoss`）的向后传递可以产生无法在 FP16 中表示的梯度。在启用 autocast 的区域中，前向输入可能是 FP16，这意味着反向梯度必须可以用 FP16 表示。因此，binary_cross_entropy 和 BCELoss 在启用 autocast 的区域中会引发错误。
可以使用 `torch.nn.function.binary_cross_entropy_with_logits()` 或 `torch.nn.BCEWithLogitsLoss` 来代替。 

**2. GradScaler**

梯度缩放（gradient scaling）有助于防止在使用混合精度进行训练时，出现梯度下溢，也就是在 FP16 下过小的梯度值会变成 0，因此相应参数的更新将丢失。同样的道理，如果网络中有过小的值，比如防止出现除零而加入的 eps 值如果过小（比如 1e-8），也会导致除零错误出现。

为了防止下溢，梯度缩放将网络的损失乘以比例因子，并对缩放后的损失调用向后传递。然后通过网络向后流动的梯度按相同的因子缩放。换句话说，梯度值具有较大的幅度，因此它们不会刷新为零。

每个参数的梯度（.grad 属性）应该在优化器更新参数之前取消缩放，因此缩放因子不会干扰学习率。

`torch.cuda.amp.GradScaler` 可以执行梯度缩放步骤。
```
scaler = torch.cuda.amp.GradScaler()
```

**1+2: Automatic Mixed Precision**

```
use_amp = True

net = make_model(in_size, out_size, num_layers)
opt = torch.optim.SGD(net.parameters(), lr=0.001)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

for epoch in range(epochs):
    for input, target in zip(data, targets):
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            output = net(input)
            loss = loss_fn(output, target)
        # scaler.scale(loss) 缩放梯度，然后进行反向计算
        scaler.scale(loss).backward()
        # scaler.step() 首先取消化器分配参数梯度的缩放，如果梯度中不包括 ``inf`` ``NaN``，就运行 optimizer.step() 
        # 否则会跳过 optimizer.step() 
        scaler.step(opt)
        scaler.update()
        opt.zero_grad() # set_to_none=True here can modestly improve performance
```
**检查 loss scale**
训练过程中检查 scale，避免掉到0.
```
scaler = torch.cuda.amp.GradScaler()
current_loss_scale = scaler.get_scale()
if step % log_iter == 0:
   print('scale:', current_loss_scale)
```

**保存和加载**
如果 checkpoint 是在没有 Amp 的情况下保存的，并且你想要使用 Amp 恢复训练，直接从checkpoint 加载模型和优化器状态，然后用新创建的 GradScaler。
如果checkpoint是通过使用 Amp 创建的，并且想要在不使用 Amp 的情况下恢复训练，可以直接从checkpoint 加载模型和优化器状态，忽略保存的 scaler 。
```
# 保存
checkpoint = {"model": net.state_dict(),
              "optimizer": opt.state_dict(),
              "scaler": scaler.state_dict()}
# Write checkpoint as desired, e.g.,
# torch.save(checkpoint, "filename")

# 加载
dev = torch.cuda.current_device()
checkpoint = torch.load("filename",
                        map_location = lambda storage, loc: storage.cuda(dev))
net.load_state_dict(checkpoint["model"])
opt.load_state_dict(checkpoint["optimizer"])
scaler.load_state_dict(checkpoint["scaler"])
```

## 二、 多个XX

**多个 model，loss， optimizer**
如果有多个损失，则必须分别对每个损失调用 scaler.scale。如果网络有多个优化器，可以分别对其中任何一个优化器调用scaler.unscale_，并且必须对每个优化器单独调用 scaler.step。
但是，scaler.update 只能调用一次.
```
scaler = torch.cuda.amp.GradScaler()

for epoch in epochs:
    for input, target in data:
        optimizer0.zero_grad()
        optimizer1.zero_grad()
        with autocast(device_type='cuda', dtype=torch.float16):
            output0 = model0(input)
            output1 = model1(input)
            loss0 = loss_fn(2 * output0 + 3 * output1, target)
            loss1 = loss_fn(3 * output0 - 5 * output1, target)

        scaler.scale(loss0).backward()
        scaler.scale(loss1).backward()

        # You can choose which optimizers receive explicit unscaling, if you
        # want to inspect or modify the gradients of the params they own.
        scaler.unscale_(optimizer0)

        scaler.step(optimizer0)
        scaler.step(optimizer1)

        scaler.update()
```

**多个 GPU**
autocast 状态会在每个线程上传播，不管是在单个进程的多线程，还是每个 GPU一个进程。（和原来的没什么区别）
```
model = MyModel()
dp_model = nn.DataParallel(model)

# Sets autocast in the main thread
with autocast(device_type='cuda', dtype=torch.float16):
    # dp_model's internal threads will autocast.
    output = dp_model(input)
    # loss_fn also autocast
    loss = loss_fn(output)
```
多个GPU一个进程，这里 `torch.nn.parallel.DistributedDataParallel `可能会产生一个侧线程来在每个设备上运行前向传递，就像 `torch.nn.DataParallel` 一样。修复方法是相同的：将自动转换作为模型前向方法的一部分应用，以确保它在侧线程中启用。
```
MyModel(nn.Module):
    ...
    @autocast()
    def forward(self, input):
       ...

# Alternatively
MyModel(nn.Module):
    ...
    def forward(self, input):
        with autocast():
            ...
```

## 三、常见问题

1. 加速有限，可能的原因有：
-   显卡不支持
- GPU饱和
- FP32 -> FP16 的转换消耗了过多时间，应该避免多个小的 CUDA操作
- 过多的CPU和GPU的通信
- `matmul` 操作的尺寸应该是 8 的倍数

2. loss 是 inf/NaN
- 如果网络中有较小的数字，转成 FP16 就会变成0，导致出现inf/NaN，先去掉 GradScaler 检查前向过程中是不是会有这种问题。
- 如果前向过程中出现 NaN，一般是前向过程中某些步骤蕴含求和求平均的操作导致了上溢，找到这些可能出现上溢的地方，手动固定为 FP32 就可以了。

3. loss scale 掉到了0
通常也是因为上溢，找到上溢的层固定到 FP32 就可以了。

4. 混合精度下 transformer 的位置编码碰撞问题
目前广泛采用的位置编码算法比如 Rope 和 Alibi， 需要为每个位置生成一个整型的 position_id，在 float16/bfloat16 下浮点数精度不足，导致整数范围超过 256 时， bfloat16 无法准确表示每个整数，因此相邻的若干个 token 会共享一个位置编码。
解决思路也是保证 position_id 的精度在 FP32上就可以了。 
（在图像里ViT上下文没这么长的就不用担心这个问题）


### 参考：
1. [Pytorch AMP 教程](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html#adding-torch-autocast)
2. https://pytorch.org/docs/stable/notes/amp_examples.html
3. https://pytorch.org/docs/stable/amp.html#autocast-op-reference
5. [Pytorch中混合精度训练的使用和debug](https://mp.weixin.qq.com/s/WBl9WimNf9lwM8AR7xgNiw)
6. [Llama也中招，混合精度下位置编码竟有大坑，百川智能给出修复方案](https://mp.weixin.qq.com/s/qA6rdFUPmPsd4elxGnNf2A)
7. [To Bfloat or not to Bfloat? That is the Question!](https://www.cerebras.net/machine-learning/to-bfloat-or-not-to-bfloat-that-is-the-question/)
