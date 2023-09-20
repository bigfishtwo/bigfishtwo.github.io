---
title: 论文笔记-Evidential Deep Learning to Quantify Classification Uncertainty
date: 2023-03-30 +0800
categories: [论文笔记]
tags: [可解释]
math: true
---

原文：https://papers.nips.cc/paper/2018/file/a981f2b708044d6fb4a71a1463242520-Paper.pdf

一句话总结：利用证据理论进行不确定性估计，提出了 Evidential Deep Learning。

## 1. 使用 Softmax 的不足

对于分类任务，用 softmax作为输出类别概率的操作是很常见的， 最小化负的似然对数对应的 loss 是 cross-entropy。 cross-entropy 的概率解释只是最大似然估计（MAE），作为一个频率学派的方法，不能推理出预测分布的方差。由于神经网络输出所采用的指数，Softmax 会夸大预测类别概率，其结果是不可靠的不确定性估计。

> 频率学派认为似然函数 $p(x|\theta)$ 中的参数 $\theta$ 是固定的，可以通过数据x的概率分布的得到，最大似然估计是找到是的似然函数最大的 $\theta$ 值。而贝叶斯学派认为参数 $\theta$ 是个分布，因此可以给出不确定性。

## 2. 不确定性和证据理论

Dempster-Shafer 证据理论 (DST) 是贝叶斯理论对主观概率的推广 。它将**信念质量（belief mass）** 分配给识别框架的子集，该子集表示唯一可能状态的集合，例如一个样本可能的类别标签。一个信念质量可以分配给框架的任何子集，包括整个框架本身，它代表了真理可以是任何可能的状态的信念，例如，所有类别是均匀分布的。

**主观逻辑 (subjective logic，SL)** 将 DST 在识别框架上的信念分配概念形式化为 Dirichlet 分布 。因此，它允许人们使用证据理论的原理，通过定义明确的理论框架来量化信念质量和不确定性。

说人话就是，假设 K 个相互独立的类别，对于每个类别都分配一个 belief mass $b_k$，并且有一个整体的$\mu$。这 K + 1 个质量值都是非负的并且总和为 1，即 $\mu + \sum_{k=1}^Kb_k= 1$，其中各项都是≥0.

计算 belief mass 需要用到证据（evidence） $e_k$，
$$b_k=\frac{e_k}{S}, \mu = K/S, S=\sum_{i+1}^K(e_i+1)$$

不确定性与总证据成反比。当没有证据时，每个类别的信念为 0，不确定性为 1。作者把证据称为从数据中收集到的，有利于将样本归入某个类别的支持量的量度。belief mass 的分配，即**主观看法（subjective opinion）**，对应于参数为Dirichlet分布的参数 $a_k=e_k+1$。

也就是说主观看法可以通过 Dirichlet 分布得到，$b_k=(\alpha_k-1)/S$。

标准神经网络分类器的输出是对每个样本的可能类别的概率分配。然而，对证据进行参数化的 Dirichlet 分布代表了每个这样的概率分配的密度；因此，它二阶概率和不确定性的模型。

对于一个看法，第k个类别的期望概率为相应 Dirichlet 分布的平均值，并计算为$\hat{p}_k=\frac{\alpha_k}{S}$。

在本文中，作者认为神经网络能够形成Dirichlet 分布分类任务的意见。假设$α_i = < α_{i1}，……，α_{iK} >$为样本i分类的Dirichlet分布的参数，$(α_{ij}−1)$为网络估计的样本 i 分配到第j类的总证据。此外，给定这些参数，分类的认知不确定性可以很容易地用上面的公式计算出来。

## 3. 方法实现

把神经网络的最后一层的 softmax 换成一个产生非负输出的激活函数，比如ReLU，然后把输出作为预测 Dirichlet 分布的证据。

loss：
（以后看）
