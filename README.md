# CIGN-CD

Compact-IGNSI Guided T1-only Latent Editing with Certified Residual Decoding for Remote Sensing Change Detection

中文名称：紧凑IGNSI结构先验引导的T1-only潜空间编辑与认证残差解码遥感变化检测方法。

本工程是研究型完整代码框架，不是论文官方代码。

## 方法简介

CIGN-CD不直接把前后时相图像 `T1` 与 `T2` 做简单差分，而是先构造一个“无变化假设下的后时相参考” `T2_ref`。模型从 `T1` 中抽取结构先验，从 `T2` 中抽取低频观测风格，再用T1-only latent editing生成 `T2_ref = O(S1, c2)`。之后比较真实 `T2` 与参考 `T2_ref`，得到认证残差 `D_cert = Z2 - Z2_ref_align`，作为变化检测的主要证据。

固定流程如下：

```text
T1, T2 -> SharedBackbone -> Z1, Z2
Z1 -> CompactIGNSIControlEncoder -> P1, W1 -> ControlAdapter -> C1
Z2 -> T2StyleEncoder -> c2
T1, C1, c2 -> T1OnlyLatentEditor -> T2_ref
T2_ref -> SharedBackbone -> Z2_ref
Z1, Z2_ref, W1 -> ReferenceBoundaryAligner -> Z2_ref_align, O_ref
Z1, Z2, Z2_ref_align -> D_obs, D_ref, D_cert
D_obs, D_ref, D_cert, W1 -> CertifiedResidualGatedDecoder -> logits -> sigmoid -> prob
```

核心差异定义：

```text
D_obs  = Z2 - Z1
D_ref  = Z2_ref_align - Z1
D_cert = Z2 - Z2_ref_align
```

## 工程结构

```text
cign_cd/
  configs/                 配置文件
  scripts/                 训练、测试、推理和数据生成脚本
  src/datasets/            数据集读取与增强
  src/models/              CIGN-CD全部模型模块
  src/losses/              CD、参考一致性、结构保持和对齐损失
  src/metrics/             Precision、Recall、F1、IoU、OA、Kappa
  src/engine/              训练与评估流程
  src/utils/               配置、日志、checkpoint、图像与可视化工具
  tests/                   shape、loss和metrics测试
```

## 数据集格式

默认支持LEVIR-CD、WHU-CD、CDD、SYSU-CD等二值变化检测数据整理为以下结构：

```text
dataset_root/
  train/A/*.png
  train/B/*.png
  train/label/*.png
  val/A/*.png
  val/B/*.png
  val/label/*.png
  test/A/*.png
  test/B/*.png
  test/label/*.png
```

`A` 为前时相 `T1`，`B` 为后时相 `T2`，`label` 为二值变化标签。mask中大于127的像素视为变化类别1，其余为0。图像文件名需要一一对应。

## 安装依赖

```bash
pip install -r requirements.txt
```

依赖只包含 `torch`、`torchvision`、`numpy`、`pillow`、`tqdm`、`pyyaml`、`pytest`，不使用OpenCV、albumentations或scikit-learn。

## 生成假数据

```bash
python scripts/make_dummy_dataset.py
```

默认会生成 `dummy_cd_dataset/`，包含train、val、test三个划分。每张图像尺寸为256x256，A和B之间有可见的矩形或圆形变化区域。

## 新手直接运行方式

如果你不想使用yaml配置文件，可以直接运行下面这种最常见的深度学习项目入口：

```bash
python train.py
python test.py
```

当前版本的 `T1OnlyLatentEditor` 已经是完整 latent diffusion 形式：训练时使用DDPM前向加噪和噪声预测损失，推理时使用DDIM或DDPM多步反向采样生成 `T2_ref`，不是单步去噪简化模块。

第一次运行前先生成假数据：

```bash
python scripts/make_dummy_dataset.py
```

`train.py` 默认会先预训练AutoEncoder，再训练CIGN-CD主模型。训练好的主模型默认保存到：

```text
outputs/cign_cd/checkpoints/best.pth
```

如果你想用自己的数据集，只需要这样改路径：

打开 `train.py` 和 `test.py`，修改文件顶部 `HYPERPARAMS` 里的 `data_root`。

Windows电脑如果DataLoader卡住，可以把线程数设为0：

打开 `train.py` 和 `test.py`，把文件顶部 `HYPERPARAMS` 里的 `num_workers` 改成 `0`。

常用扩散参数也写在 `HYPERPARAMS` 里，不需要在命令后面加一长串参数。其中 `diffusion_steps` 是DDPM训练总时间步，`inference_steps` 是测试时反向采样步数，`sample_method` 可选 `ddim` 或 `ddpm`，`edit_strength` 表示从T1 latent加噪开始编辑的强度。

下面的yaml方式仍然保留，适合后续做大量实验时统一管理参数。

## 预训练AutoEncoder

```bash
python scripts/pretrain_autoencoder.py --config configs/ae_pretrain.yaml
```

脚本读取训练集中的A和B图像作为普通重建样本，训练 `TinyAutoEncoder`，并保存：

```text
outputs/ae_pretrain/checkpoints/best_ae.pth
outputs/ae_pretrain/checkpoints/last_ae.pth
```

## 训练CIGN-CD

```bash
python scripts/train_cign_cd.py --config configs/cign_cd.yaml
```

支持从last checkpoint恢复：

```bash
python scripts/train_cign_cd.py --config configs/cign_cd.yaml --resume outputs/cign_cd/checkpoints/last.pth
```

如果配置中的 `device: cuda` 但当前没有GPU，脚本会自动切换到CPU。训练日志写入 `outputs/cign_cd/train_log.jsonl`，checkpoint保存到 `outputs/cign_cd/checkpoints/`。

## 测试

```bash
python scripts/test_cign_cd.py --config configs/cign_cd.yaml --checkpoint outputs/cign_cd/checkpoints/best.pth
```

输出 Precision、Recall、F1、IoU、OA、Kappa，并将预测图保存到：

```text
outputs/cign_cd/test_predictions/
```

## 文件夹推理

```bash
python scripts/infer_folder.py \
  --checkpoint outputs/cign_cd/checkpoints/best.pth \
  --t1_dir dummy_cd_dataset/test/A \
  --t2_dir dummy_cd_dataset/test/B \
  --out_dir outputs/cign_cd/infer_results
```

输出包括 `prob` 概率图、`pred` 二值变化图、`t2_ref` 无变化参考图、`a_exp` 解释权重图、`a_cert` 认证权重图和 `structure_weight` 结构权重图。

## 可视化

```bash
python scripts/visualize_batch.py --config configs/cign_cd.yaml --checkpoint outputs/cign_cd/checkpoints/best.pth
```

可视化内容包含：

```text
T1, T2, GT, T2_ref, Pred, A_exp, A_cert, Structure_Weight
```

## 主要模块

- `SharedBackbone`：共享轻量CNN编码器，T1、T2和T2_ref使用同一个实例。
- `CompactIGNSIControlEncoder`：从Z1中提取结构先验P1和结构权重W1，不引入basis、rank gate或正交损失。
- `T2StyleEncoder`：对Z2做FFT低通滤波，再通过GAP和MLP得到低频风格向量c2。
- `TinyAutoEncoder`：T1-only latent editing使用的轻量自编码器。
- `T1OnlyLatentEditor`：完整T1-only latent diffusion编辑器，只编码T1 latent，不编码T2，不以T2 latent作为扩散目标；训练时预测噪声，推理时DDIM/DDPM多步采样生成T2_ref。
- `ReferenceBoundaryAligner`：只warp参考特征Z2_ref，不warp真实Z2。
- `CertifiedResidualGatedDecoder`：用解释门控和认证门控融合D_obs、D_ref、D_cert并输出变化logits。

## 测试

```bash
pytest -q
```

测试覆盖模型forward shape、AutoEncoder shape、CIGNSI shape、decoder shape、反向传播、loss标量返回与metrics计算。

## 注意事项

训练前建议先运行假数据脚本和测试，确认本地PyTorch环境可用。实际接入遥感数据集时，只需要按默认目录组织A、B和label，并在配置文件中修改 `data.root`。本工程强调可运行的研究框架与清晰模块边界，便于继续扩展论文实验、消融和可视化分析。
