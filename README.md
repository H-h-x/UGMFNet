# UGMFNet: Uncertainty Guided Multimodal Fusion Network for Remote Sensing Data

**中文说明见下文 / Chinese version follows the English version.**

## Overview

UGMFNet is an uncertainty-guided multimodal fusion network for remote-sensing classification. It is designed to combine complementary information from a spectral modality (e.g., hyperspectral or multispectral imagery) and an auxiliary spatial/structural modality (e.g., LiDAR or SAR, depending on the dataset setup).

This initial repository snapshot provides **the core PyTorch model implementation only**.

### Main components

- **Xvolution:** combines a depthwise-convolution branch with a dynamic local-attention branch.
- **Spectral and spatial feature extraction:** extracts modality-specific feature representations.
- **Feature-level confidence:** estimates modality reliability using prototype-anchored intra-modal similarity and inter-modal similarity.
- **Prediction-level confidence:** derives confidence proxies from classification logits using an energy score.
- **Confidence-guided fusion and classification:** combines the confidence cues to weight modality-specific representations before classification.

## Repository contents

```text
UGMFNet_GitHub_Release/
├── model.py          # Core UGMFNet model implementation
├── requirements.txt  # Minimal dependency for the model module
└── README.md         # English and Chinese documentation
```

Training scripts, dataset files, logs, ablation scripts, and checkpoints are intentionally excluded from this partial release.

## Installation

The model module depends on PyTorch.

```bash
pip install -r requirements.txt
```

The manuscript reports an experimental environment of Python 3.9.21, PyTorch 2.6.0, CUDA 12.6, and cuDNN 9.8. GPU users should install a PyTorch build compatible with their operating system and CUDA setup.

## Quick start

The current implementation retains the original class name `octfusion_multi_adder_1`.

```python
import torch
from model import octfusion_multi_adder_1

# Illustrative dimensions only; adjust them to your dataset.
# These values are a shape-check example, not the official training configuration.
model = octfusion_multi_adder_1(
    in_channels_1=15,   # Number of channels in the spectral input
    in_channels_2=1,    # Number of channels in the auxiliary input
    out_channels=6,     # Number of target classes
    weight_channels=64,
    fusion_hidden=[128, 64],
    dropout=0.5,
)

spectral_input = torch.randn(2, 15, 23, 23)
auxiliary_input = torch.randn(2, 1, 23, 23)

model.eval()
with torch.no_grad():
    probabilities = model(spectral_input, auxiliary_input)

print(probabilities.shape)  # torch.Size([2, 6])
```

### Input and output

- `x1`: spectral input tensor with shape `[batch_size, spectral_channels, height, width]`.
- `x2`: auxiliary-modality tensor with shape `[batch_size, auxiliary_channels, height, width]`.
- The two inputs should have matching spatial dimensions.
- Output: class probabilities with shape `[batch_size, num_classes]`. The current implementation applies `Softmax` in the model's forward pass.

The example uses random tensors only to demonstrate the expected tensor shapes. It does **not** reproduce the reported benchmark results.

## Reported evaluation

The accompanying manuscript evaluates UGMFNet on three public multimodal remote-sensing datasets: Houston, Trento, and GRSS-DFC-2007. The manuscript reports the following overall accuracy (OA):

| Dataset | OA reported in the manuscript |
|---|---:|
| Houston | 93.12% |
| Trento | 98.00% |
| GRSS-DFC-2007 | 94.97% |

These figures are reported for the full experimental setup; they should not be interpreted as results reproduced by this model-only release.

## Planned release

The repository is currently a partial code release. **Pretrained model weights and detailed training settings/hyperparameters will be released in a future update.** Dataset preparation and training scripts are not included in this snapshot.

## License

A license has not yet been specified for this initial release. Please add an appropriate `LICENSE` file before granting reuse or redistribution permissions.

---

# 中文说明

## 项目简介

UGMFNet（不确定性引导的多模态融合网络）面向遥感图像分类任务，旨在融合不同传感器提供的互补信息。根据具体数据集，输入模态可以是高光谱/多光谱影像，以及 LiDAR 或 SAR 等辅助空间或结构模态。

当前仓库为**部分代码公开版本，仅包含核心 PyTorch 模型实现**。

### 核心模块

- **Xvolution 算子：** 将深度卷积分支与动态局部注意力分支相结合。
- **光谱与空间特征提取：** 分别提取不同模态的特征表示。
- **特征层置信度：** 基于原型引导的模态内相似度与模态间相似度，评估各模态的可靠性。
- **预测层置信度：** 根据分类 logits 计算能量分数，构建预测层面的置信度代理。
- **置信度引导的融合与分类：** 联合使用两类置信度线索，对模态特征进行加权融合并完成分类。

## 仓库结构

```text
UGMFNet_GitHub_Release/
├── model.py          # UGMFNet 核心模型代码
├── requirements.txt  # 模型模块的最小依赖
└── README.md         # 中英文说明文档
```

本次暂不包含训练脚本、数据集文件、日志、消融实验脚本及模型权重。

## 环境安装

模型模块依赖 PyTorch，可执行以下命令安装：

```bash
pip install -r requirements.txt
```

论文中报告的实验环境为 Python 3.9.21、PyTorch 2.6.0、CUDA 12.6 和 cuDNN 9.8。若使用 GPU，请根据操作系统及 CUDA 环境安装相匹配的 PyTorch 版本。

## 快速使用

当前代码保留了原始模型类名 `octfusion_multi_adder_1`。

```python
import torch
from model import octfusion_multi_adder_1

# 以下仅用于演示输入输出尺寸，请根据实际数据集调整。
# 这些示例值不是论文最终训练配置。
model = octfusion_multi_adder_1(
    in_channels_1=15,   # 光谱模态的输入通道数
    in_channels_2=1,    # 辅助模态的输入通道数
    out_channels=6,     # 目标类别数量
    weight_channels=64,
    fusion_hidden=[128, 64],
    dropout=0.5,
)

spectral_input = torch.randn(2, 15, 23, 23)
auxiliary_input = torch.randn(2, 1, 23, 23)

model.eval()
with torch.no_grad():
    probabilities = model(spectral_input, auxiliary_input)

print(probabilities.shape)  # torch.Size([2, 6])
```

### 输入与输出

- `x1`：光谱模态输入，形状为 `[batch_size, spectral_channels, height, width]`。
- `x2`：辅助模态输入，形状为 `[batch_size, auxiliary_channels, height, width]`。
- 两个输入的空间尺寸应保持一致。
- 输出：形状为 `[batch_size, num_classes]` 的类别概率。当前实现会在模型前向传播中执行 `Softmax`。

示例中的随机张量仅用于检查输入输出尺寸，**不能用于复现论文中的实验结果**。

## 论文报告的实验结果

论文在三个公开多模态遥感数据集 Houston、Trento 和 GRSS-DFC-2007 上进行了评估，报告的总体准确率（OA）如下：

| 数据集 | 论文报告的 OA |
|---|---:|
| Houston | 93.12% |
| Trento | 98.00% |
| GRSS-DFC-2007 | 94.97% |

以上数值来自论文的完整实验设置，不代表当前仅开放模型代码的版本已经复现这些结果。

## 后续发布计划

当前仓库属于部分代码公开版本。**后续将发布预训练模型权重以及详细的训练设置和超参数。** 本次暂不提供数据预处理和训练脚本。

## 许可证

当前版本尚未指定许可证。若希望他人使用、修改或再分发代码，请先添加合适的 `LICENSE` 文件。
# UGMFNet
