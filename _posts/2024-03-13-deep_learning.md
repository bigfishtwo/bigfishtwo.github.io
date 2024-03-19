---
title: Deep learning questions
date: 2024-03-13 +0800
categories: [深度学习]
tags: []
math: true
---

**什么是卷积？**
: 卷积运算的过程，将卷积核与矩阵上对应位置的值相乘后的和，放在输出矩阵对应的位置，输出的矩阵也就是通过卷积得到的特征矩阵，不同的卷积核可以得到不同的特征输出。把不同的特征组合起来，就能得到输入图像的特征。

**卷积的计算**
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

**什么是空洞卷积？计算空洞卷积的特征大小**
:空洞卷积是通过在卷积核的元素之间插入0，将原来的卷积核扩展成为一个更大的卷积核，这么做的目的是在不增加实际核函数参数数目的情况下，增大输出的感受野大小，同时避免下采样过程中的信息损失。空洞卷积计算公式 $H'= (H-h+(h-1)(d-1)+2p)/s+1, W' = (W-w+(w-1)(d-1)+2p)/s+1$，$d$ 为空洞率。

```python
import torch 
x = torch.randn((1,224,224))  # input image
conv = torch.nn.Conv2d(in_channels=1,out_channels=3,kernel_size=(3,3), stride=2,padding=1,dilation=2)
y = conv(x)
print(y.shape)
>> torch.Size([3, 111, 111])
```

**什么是池化？计算池化后的特征大小**
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

**Self-Attention 怎么计算？QKV分别代表什么？**
:输入为$X$，通过线性层得到 $Q=W_QX, K=W_KX,V=W_VX$。QKV 表示每个 token 拿着自己的 Query 去和别的 Key 寻找相似性，并用在自己的 Value 上。  Q 和 K 点乘可以得到一个注意力分数矩阵，表示每个 token 对 其他 token 的注意力。利用向量维度来缩小分数矩阵，保证梯度稳定性。分数矩阵通过 softmax 转为概率，并与 V 相乘，让模型学习到概率分数高的更重要。 Self-attention 的计算公式如下， 其中 $d_k$ 是向量维度：

$Attention=softmax({QK^T}/{\sqrt{d_k}})V$

**为什么要除以 $\sqrt{d_k}$ ？**
:假设 Q 和 K 的均值为 0，方差为 1。它们的矩阵乘积将有均值为 0，方差为 $d_k$ ，因此使用  $d_k$ 的平方根被用于缩放，使得 Q 和 K 的矩阵乘积的均值为 0，方差为1。

**Multi-Head Attention 如何计算**
:Multi-Head Attention 包含多个 Self-Attention 层，首先将输入$X$分别传递到$k$ 个不同的 Self-Attention 中，计算得到 $k$ 个输出，所有 $k$ 个头的结果会被 concatenate 合在一起，传入一个Linear层，得到最终的输出。

卷积的 Attention 方法有什么？

分类任务的常见指标
- Accuracy：
$\text { accuracy }=\frac{T P+T N}{T P+T N+F P+F N}$ 。accuracy指的是正确预测的样本数占总预测样本数的比值，它不考虑预测的样本是正例还是负例,考虑的是全部样本。

- Precision（查准率）：
$\text { precision }=\frac{T P}{T P+FP}$。Precision指的是正确预测的正样本数占所有预测为正样本的数量的比值，也就是说所有预测为正样本的样本中有多少是真正的正样本。从这我们可以看出，precision只**关注预测为正样本**的部分，

- Recall（召回率）：
$\text { recall }=\frac{T P}{T P+FN}$。它指的是正确预测的正样本数占**真实正样本总数**的比值，也就是能从这些样本中能够正确找出多少个正样本。

- F1-score：
$F1-\text { score }=\frac{2}{1 / \text { precision }+1 / \text { recall }}$. F-score 相当于 precision 和 recall 的调和平均，用意是要参考两个指标。从公式我们可以看出，recall和precision任何一个数值减小，F-score都会减小，反之，亦然。

- Specificity：
$\text { specificity }=\frac{T N}{T N+F P}$。 Specificity 指的是正确预测的负样本数占**真实负样本总数**的比值，也就是能从这些样本中能够正确找出多少个负样本。

- Sensitivity(TPR)：
$\text { sensitivity }=\frac{T P}{T P+F N}=\text { recall }$

- Auc(Area under curve)：
Auc指的是计算roc的面积。 AUC值是一个概率值，当你随机挑选一个正样本以及负样本，当前的分类算法根据计算得到的Score值将这个正样本排在负样本前面的概率就是AUC值，AUC值越大，当前分类算法越有可能将正样本排在负样本前面，从而能够更好地分类。
$$A U C=\frac{\sum_{i \in \text { positiveClass }} \operatorname{rank}_{i}-\frac{M(1+M)}{2}}{M \times N}$$

分割相关

分割常用的方法有哪些

介绍 UNet 

**为什么Unet在医疗图像分割种表现好**
:医学影像的特点，各种各样的模态导致很难用一个统一的大模型去处理所有的细分领域任务；较少的数据量和标注的困难，使得复杂的网络会很容易过拟合，而较为轻量的UNet结构反而能取得更好的效果。并且对于医学图像分割来说，相比于模型结构上的改动，数据处理和优化方式反而更加要紧一些（nnUNet）。


**分割任务的常见指标**


**分割任务常用loss**
- 交叉熵损失 Cross Entropy Loss Function：
用于图像语义分割任务的最常用损失函数是像素级别的交叉熵损失，这种损失会逐个检查每个像素，将对每个像素类别的预测结果（概率分布向量）与独热编码标签向量( one-hot 形式)进行比较。
每个像素对应的损失函数为 $L = -\sum_{c=1}^{M}y_c log(p_c)$
其中，$M$ 代表类别数，$y_c$ 是one-hot向量，元素只有 0 和 1 两种取值，至于$p_c$表示预测样本属于 $c$ 类别的概率。

- Focal Loss：
在样本数量不平衡的情况下，模型应关注那些难分样本，将高置信度的样本损失函数降低一些，Focal loss ：
$$FL = \begin{matrix} 
-\alpha (1-p)^{\gamma} log(p) & if \ y =1\\ -(1-\alpha) p^{\gamma} log(1-p) & if \ y =0  
\end{matrix}$$

- Dice Loss：
在医疗图像分割模型 VNet 中提出的，感兴趣的解剖结构仅占据扫描的非常小的区域，从而使学习过程陷入损失函数的局部最小值。所以要加大前景区域的权重。
Dice系数是一种集合相似度度量的函数，可以理解为是两个轮廓区域的相似程度，用A、B表示两个轮廓区域所包含的点集，公式为： $Dice(A,B)=2 \frac{|A⋂B|}{|A|+|B|}$。
其次Dice也可以表示为： $Dice(A, B) = \frac{2TP}{2TP+FN+FP}$， 其中TP，FP，FN分别是真阳性、假阳性、假阴性的个数。

$$ dice \ loss = 1- Dice$$

- IOU Loss：
可类比DICE LOSS，也是直接针对评价标准进行优化，公式如下： $IOU = 1 - \frac{A \bigcap B}{A \bigcup B}$


医学图像分割前处理，后处理

   