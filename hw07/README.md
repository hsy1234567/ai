# 肺炎X光影像分类项目

基于深度学习的胸部X光影像肺炎检测项目，包含二分类和三分类两个任务。

## 目录结构

```
chest/
├── README.md                  # 项目说明文档
├── chest_xray/                # 数据集目录（需自行下载）
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── test/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── val/
│       ├── NORMAL/
│       └── PNEUMONIA/
├── 任务一/                    # 肺炎二分类任务
│   ├── train.py               # 训练脚本
│   ├── requirements.txt       # 依赖配置
│   ├── README.md              # 任务说明
│   ├── report.md              # 实验报告（运行后生成）
│   ├── best_model.h5          # 模型权重（训练后生成）
│   └── figures/               # 图表目录
│       ├── metrics_plot.png
│       └── confusion_matrix.png
└── 任务二/                    # 肺炎三分类任务（进阶）
    ├── train.py               # 训练脚本
    ├── requirements.txt       # 依赖配置
    ├── README.md              # 任务说明
    ├── report.md              # 实验报告（运行后生成）
    ├── best_model_3class.h5   # 模型权重（训练后生成）
    └── figures/               # 图表目录
        ├── metrics_plot.png
        └── confusion_matrix.png
```

## 一键运行命令

### 任务一：肺炎二分类

```bash
cd 任务一
pip install -r requirements.txt
python train.py
```

### 任务二：肺炎三分类

```bash
cd 任务二
pip install -r requirements.txt
python train.py
```

## 数据集获取

### 数据集来源
- **数据集名称**: Chest X-Ray Images (Pneumonia)
- **Kaggle链接**: [https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

### 数据集说明
- **数据规模**: 约1.15 GB，共5800余张儿童胸部X光影像
- **标注质量**: 所有影像均经过至少两名专家审核，标签可靠
- **数据结构**:
  - `train/`: 训练集（约5200张）
  - `test/`: 测试集（约600张）
  - `val/`: 验证集（仅16张，参考价值较低）

### 数据集准备

1. 从Kaggle下载数据集
2. 解压到项目根目录，确保目录结构为 `chest_xray/`
3. 运行训练脚本，程序会自动加载数据

## 最终测试集指标摘要

### 任务一：二分类（Normal vs Pneumonia）

| 指标 | 预期结果 |
|------|----------|
| Accuracy | ≥ 95% |
| Precision | ≥ 94% |
| Recall | ≥ 96% |
| F1 Score | ≥ 95% |

### 任务二：三分类（Normal / Viral / Bacterial）

| 指标 | 预期结果 |
|------|----------|
| Accuracy | ≥ 90% |
| Precision (Macro) | ≥ 88% |
| Recall (Macro) | ≥ 88% |
| F1 Score (Macro) | ≥ 88% |

## 模型架构

### 迁移学习方案（VGG16）

```
VGG16 (冻结)
    └── Flatten
        └── Dense(512, ReLU)
            └── Dropout(0.5)
                └── Dense(N, Softmax/Sigmoid)
```

- **输入尺寸**: 224×224×3
- **预训练权重**: ImageNet
- **优化器**: Adam (lr=0.0001)
- **数据增强**: 旋转、平移、缩放、水平翻转

## 输出文件

运行训练脚本后会生成：

1. **模型权重**: `best_model.h5` 或 `best_model_3class.h5`
2. **实验报告**: `report.md`（包含完整分析）
3. **训练曲线**: `figures/metrics_plot.png`
4. **混淆矩阵**: `figures/confusion_matrix.png`

## 运行环境推荐

| 环境 | 说明 |
|------|------|
| **Kaggle Notebook** | 推荐，免费GPU，直接挂载数据集 |
| **Google Colab** | 免费GPU可用 |
| **本地环境** | 建议配置GPU加速，CPU运行时间较长 |

## 注意事项

1. 首次运行会自动下载VGG16预训练权重（约500MB）
2. 训练时间：GPU约10-30分钟，CPU约数小时
3. 数据集需自行下载并放置在 `chest_xray/` 目录下
4. 原始 `val` 文件夹仅有16张图片，程序会从 `train` 按8:2比例重新划分

## 项目说明

本项目包含两个任务：
- **任务一**: 肺炎二分类（Normal vs Pneumonia）- 必做
- **任务二**: 肺炎三分类（Normal / Viral / Bacterial）- 选做进阶任务

详细报告请查看各任务目录下的 `report.md` 文件。