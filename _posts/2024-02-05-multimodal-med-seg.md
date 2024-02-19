---
title: 医学多模态图像分割
date: 2024-02-05 +0800
categories: [论文笔记]
tags: [医学图像,分割]
---

## 一、图像多模态

这里指的是有一些具有多个模态的医学图像，例如 MRI，或者不同成像方式得到的图像，比如 CT-MR，PET-MR 组成的多模态医学数据。

多模态融合方式包括在输入的融合，layer-level 的融合，以及在决策端 or 输出的融合。

### 1. Input-level fusion
 
 多数模型采用的方法：UNet，nnUNet, CNN+ViT,UNETR_v, Swin UNETR。

#### 1.1 Convolution-based：

几个模态拼接在一起输

#### 1.2  Transformer-based：

*Multi-modal medical Transformers: A meta-analysis for medical image segmentation in oncology，2023*

i. tokenization：每个模态单独tokenize，或者多模态拼接起来tokenize

ii.	linear projection：共用一个，或者每个模态单独一个 projection

![20240205183252](https://cdn.jsdelivr.net/gh/bigfishtwo/BlogPics@main/imgs/20240205183252.png)

iii. before transformer block：每个模态在对应位置加权求和，或者concat

![20240205182758](https://cdn.jsdelivr.net/gh/bigfishtwo/BlogPics@main/imgs/20240205182758.png)


### 2. Layer-level fusion

#### 2.1 卷积网络

i.	*Modality-Aware Mutual Learning for Multi-modal Medical Image Segmentation，2021*

两个模态单独分支和 concat 一起的分支融合
![20240205183414](https://cdn.jsdelivr.net/gh/bigfishtwo/BlogPics@main/imgs/20240205183414.png)

ii.	*Cross-Modal Prostate Cancer Segmentation via Self-Attention Distillation，2021*

两个模态分别输入在中间层融合

![20240205183601](https://cdn.jsdelivr.net/gh/bigfishtwo/BlogPics@main/imgs/20240205183601.png)

iii. *Cross-modality deep feature learning for brain tumor segmentation，2021*

基于 GAN 的方法，先训练模态间的生成器，然后模态融合做分割
![20240205183622](https://cdn.jsdelivr.net/gh/bigfishtwo/BlogPics@main/imgs/20240205183622.png)

iv.	*Feature-enhanced generation and multi-modality fusion based deep neural network for brain tumor segmentation with missing MR modalities，2021*

生成缺失模态，然后分割
![20240205183640](https://cdn.jsdelivr.net/gh/bigfishtwo/BlogPics@main/imgs/20240205183640.png)

#### 2.2 Transformer

i. 模态间的 cross-attention：

把所有 block 里都换成 cross-attention，或者结合 cross-attention 和 self-attention

![20240205183657](https://cdn.jsdelivr.net/gh/bigfishtwo/BlogPics@main/imgs/20240205183657.png)


### 3. Decision-level fusion

决策层上的融合时，每个模态图像都用作单个分割网络的单个输入。单一网络可以更好地利用相应模态的独特信息。然后将整合各个网络的输出以获得最终的分割结果。

决策层的融合策略包括平均和投票。平均策略通常对各个网络的置信度进行平均。通过为每个体素分配最高置信度来获得最终的分割。对于多数投票策略，体素的最终标签取决于各个网络的大多数标签。

决策层融合的缺点是需要更多内存，因为需要训练更多的参数。

### 4.	结论

*Multi-modal medical Transformers: A meta-analysis for medical image segmentation in oncology，2023*

i. ViT+CNN混合模型表现最好，但是取决于任务，随着模态数量增加，操作数增加

ii. nnUNet 在不同任务上表现稳定，well-traine nnUNet在中小型数据集的多模态分割任务上足够了

iii. 建议使用nnUNet pipline

*STU-Net: Scalable and Transferable Medical Image Segmentation Models Empowered by Large-Scale Supervised Pre-training， 2023*

i. nnUNet比transformer-based 网络在多种任务上更稳定
ii. nnUNet 的up-scale版本，在大数据集上预训练后在多模态数据集上微调，效果比nnUNet好

### 5. paper list 

- Multi-Modal CO-learning for Live Lesion Segmentation on PET-CT Image, TMI 2021
- Disentangle domain features for cross-modality cardiac image segmentation, MIA 2021
- United adversarial learning for liver tumor segmentation and detection of multi-modality non-contrast MRI, MIA 2021
- Multi-phase and Multi-level Selective Feature Fusion for Automated Pancreas Segmentation from CT lmages, MICCAI 2020
- Semi-Supervised Unpaired Multi-Modal Learning for Label-Efficient Medical lmage Segmentation, MICCAI 2021
- Semantic Consistent Unsupervised Domain Adaptation for Cross-Modality Medical lmage Segmentation, MICCAI 2021
- Modality-Aware Mutual Learning for Multi-modal Medical lmage Segmentation, MICCAI 2021
- Generalizable Cross-modality Medical Image Segmentation via Style Augmentation and Dual Normalization, CVPR 2022
- SDC-UDA, cvpr2023
