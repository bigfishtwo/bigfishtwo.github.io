---
title: Deep learning questions
date: 2024-03-13 +0800
categories: [深度学习]
tags: []
math: true
---

1. 什么是卷积？
: 卷积运算的过程，将卷积核与矩阵上对应位置的值相乘后的和，放在输出矩阵对应的位置，输出的矩阵也就是通过卷积得到的特征矩阵，不同的卷积核可以得到不同的特征输出。把不同的特征组合起来，就能得到输入图像的特征。

2. 卷积的计算
: 假设一个大小为 $H\times W$的特征图，与一个大小为$h \times w$的卷积核做卷积，padding大小为$p$，stride大小为$s$，得到的输出特征图的尺寸为 $H'= (H-h+2p)/s+1, W' = (W-w+2p)/s+1$。其中除法部分是向下取整的。
假设：输入图片大小为224×224，依次经过一层卷积（channel 3, kernel size 3×3，padding 1，stride 2）， 输出特征图大小为：3×112×112.

```python
import torch 
x = torch.randn((1,224,224))  # input image
conv = torch.nn.Conv2d(in_channels=1,out_channels=3,kernel_size=(3,3), stride=2,padding=1)
y = conv(x)
print(y.shape)
>> torch.Size([3, 112, 112])
```

3. 什么是空洞卷积？计算空洞卷积的特征大小
：空洞卷积是通过在卷积核的元素之间插入0，将原来的卷积核扩展成为一个更大的卷积核，这么做的目的是在不增加实际核函数参数数目的情况下，增大输出的感受野大小，同时避免下采样过程中的信息损失。空洞卷积计算公式 $H'= (H-h+(h-1)(d-1)+2p)/s+1, W' = (W-w+(w-1)(d-1)+2p)/s+1$，$d$ 为空洞率。

```python
import torch 
x = torch.randn((1,224,224))  # input image
conv = torch.nn.Conv2d(in_channels=1,out_channels=3,kernel_size=(3,3), stride=2,padding=1,dilation=2)
y = conv(x)
print(y.shape)
>> torch.Size([3, 111, 111])
```

4. 什么是池化？计算池化后的特征大小
: 池化通常用于卷积层后面，作用是提取范围更广的特征，同时减少计算量。池化和卷积计算特征图大小公式相同。
```python
import torch 
x = torch.randn((1,224,224))  # input image
conv = torch.nn.Conv2d(in_channels=1,out_channels=3,kernel_size=(3,3), stride=2,padding=1)
pool = torch.nn.AvgPool2d(kernel_size=(3,3), stride=1,padding=0)
y = conv(x)
y_p = pool(y)
print(y_p.shape)
>> torch.Size([3, 110, 110])
```

5. Self-Attention 怎么计算？QKV分别代表什么？ 参数量如何计算？
:输入为$x$，通过

6. 卷积的 Attention 方法有什么？

7. 分类任务的常见指标

分割相关

8. 分割任务的常见指标

9.  分割任务常用loss

10. 医学图像分割前处理，后处理

11. 