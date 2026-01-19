# Mouse Behavior Classification

小鼠社交行为分类项目 - 基于DeepLabCut追踪数据和Caltech Behavior Annotator标注

- 📹 Video: https://youtu.be/oTXjbmTi8IQ
- 📊 Dataset DOI: https://doi.org/10.6084/m9.figshare.30393298
- 💻 Environment: Python 3.9–3.11, VS Code / Kaggle

---

## 项目结构 (Project Structure)

```
Mouse-Behavior-Classifier-Train/
├── README.md                           # 项目文档
├── data/                               # 数据目录
│   ├── dlc_csv/                        # DeepLabCut追踪CSV文件 (58个视频)
│   │   └── *DLC_*.csv                  # 多动物追踪结果
│   ├── annotations/                    # Caltech Behavior Annotator标注文件 (58个)
│   │   └── *_annot.txt                 # 行为标注文件
│   ├── dataset58/                      # 预处理后的数据集
│   │   ├── feature8_58.xlsx            # 8特征矩阵 (用于8feature_src)
│   │   ├── feature_21.xlsx             # 21特征矩阵
│   │   ├── merged_labels.xlsx          # Behavior标签 (3分类)
│   │   └── labels_aggression.xlsx      # Aggression标签 (7分类)
│   └── processed/                      # 中间处理结果
│
├── src/                                # 源代码目录
│   ├── __init__.py                     # 包初始化
│   ├── label_parser.py                 # 标签解析器 (解析Caltech标注)
│   ├── feature_extraction.py           # 特征提取器 (从DLC提取26特征)
│   ├── data_loader.py                  # 数据加载器 (整合特征和标签)
│   ├── models.py                       # 模型定义 (MLP/LSTM/CNN/Transformer等)
│   ├── mouse_behavior_classification.ipynb  # 26特征实验主Notebook
│   │
│   └── 8feature_src/                   # 8特征模型训练代码
│       ├── kaggle_model_comparison.ipynb   # 8特征模型对比实验Notebook
│       ├── CNN.py                      # CNN模型
│       ├── LSTM.py                     # LSTM模型
│       ├── GMM.py                      # GMM模型
│       ├── HMM.py                      # HMM模型
│       ├── LightGBM.py                 # LightGBM模型
│       ├── XGBoost.py                  # XGBoost模型
│       ├── RandomForest.py             # RandomForest模型
│       ├── SVM.py                      # SVM模型
│       ├── data_load.py                # 数据加载工具
│       ├── data_solver.py              # 数据处理工具
│       ├── config.py                   # 配置文件
│       └── model_comparison_*.py       # 模型对比脚本
│
└── visualization/                      # 可视化输出
    └── visualization_*.html            # 交互式可视化结果
```

---

## 数据流水线 (Data Pipeline)

### 概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DATA PIPELINE                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────────────────┐   │
│  │  DLC CSV     │      │ Annotation   │      │     Feature Matrix       │   │
│  │  (58 files)  │      │  (58 files)  │      │                          │   │
│  └──────┬───────┘      └──────┬───────┘      │  26-feature: src/        │   │
│         │                     │              │  8-feature:  8feature_src│   │
│         ▼                     ▼              └──────────────────────────┘   │
│  ┌──────────────┐      ┌──────────────┐                                     │
│  │  feature_    │      │  label_      │                                     │
│  │  extraction  │      │  parser.py   │                                     │
│  │  .py         │      │              │                                     │
│  └──────┬───────┘      └──────┬───────┘                                     │
│         │                     │                                              │
│         │  26 features        │  Frame-level labels                         │
│         │                     │                                              │
│         └─────────┬───────────┘                                              │
│                   ▼                                                          │
│         ┌──────────────────────┐                                             │
│         │    data_loader.py    │                                             │
│         │  (Align & Combine)   │                                             │
│         └──────────┬───────────┘                                             │
│                    │                                                         │
│         ┌──────────┴──────────┐                                              │
│         ▼                     ▼                                              │
│  ┌─────────────┐      ┌─────────────┐                                        │
│  │  Behavior   │      │  Aggression │                                        │
│  │  (3-class)  │      │  (7-class)  │                                        │
│  └─────────────┘      └─────────────┘                                        │
│                                                                              │
│                    ▼                                                         │
│         ┌──────────────────────┐                                             │
│         │   Train/Val/Test     │                                             │
│         │   Stratified Split   │                                             │
│         └──────────┬───────────┘                                             │
│                    │                                                         │
│                    ▼                                                         │
│         ┌──────────────────────┐                                             │
│         │   8 Models Training  │                                             │
│         │   (CNN/LSTM/GMM/...) │                                             │
│         └──────────────────────┘                                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 两种特征模式

| 模式 | 特征数 | 代码位置 | 数据文件 | 说明 |
|------|--------|----------|----------|------|
| **26-feature** | 26 | `src/` | 从DLC实时提取 | 完整特征，从原始DLC CSV提取 |
| **8-feature** | 8 | `src/8feature_src/` | `dataset58/feature8_58.xlsx` | 精简特征，预处理好的Excel |

---

## 实验说明 (Experiments)

### S1: Behavior 实验 (3分类)
从S1标注中提取，排除base类别：

| 类别ID | 名称 | 说明 |
|--------|------|------|
| 0 | aggression | 攻击行为 |
| 1 | social | 社交行为 |
| 2 | nonsocial | 非社交行为 |

### S2: Aggression 实验 (7分类)
从S2标注中提取，排除base类别：

| 类别ID | 英文名 | 中文名 | 说明 |
|--------|--------|--------|------|
| 0 | lateralthreat | 侧向威胁 | 侧身展示威胁姿态 |
| 1 | keepdown | 压制 | 将对方压在身下 |
| 2 | clinch | 缠斗 | 激烈的肢体缠斗 |
| 3 | uprightposture | 直立姿态 | 直立对峙姿势 |
| 4 | freezing | 僵住 | 静止不动 |
| 5 | bite | 撕咬 | 咬攻击 |
| 6 | chase | 追逐 | 追赶对方 |

---

## 特征说明 (Features)

### 26特征 (从DLC提取)

从DLC多动物追踪结果中提取26个特征：

| 类别 | 数量 | 特征名 |
|------|------|--------|
| 速度特征 | 4 | top1_speed, top2_speed, body1_speed, body2_speed |
| 距离特征 | 4 | top_distance, body_distance, top1_tail2_distance, top2_tail1_distance |
| 角度特征 | 2 | angle_top1_tail1, angle_top2_tail2 |
| 坐标特征 | 12 | 两只小鼠各3个主要身体部位(top, body, tail)的x,y坐标 |
| 交互特征 | 4 | relative_angle, speed_ratio, approach_speed, body_speed_diff |

### 8特征 (精简版)

预提取的8个核心特征，用于快速实验：
- 距离特征 (身体部位间距离)
- 速度特征 (运动速度)
- 角度特征 (相对角度)

---

## 模型 (Models)

支持8种模型：

| 模型 | 类别 | 特点 |
|------|------|------|
| **MLP** | 深度学习 | 多层感知机，简单高效 |
| **LSTM** | 深度学习 | 双向长短期记忆网络，捕捉时序依赖 |
| **CNN** | 深度学习 | 1D卷积神经网络，提取局部特征 |
| **Transformer** | 深度学习 | 注意力机制，全局建模 |
| **LightGBM** | 集成学习 | 梯度提升树，快速高效 |
| **XGBoost** | 集成学习 | 极端梯度提升，鲁棒性强 |
| **RandomForest** | 集成学习 | 随机森林，防过拟合 |
| **SVM** | 传统ML | 支持向量机，适合小样本 |
| **GMM** | 概率模型 | 高斯混合模型，生成式 |
| **HMM** | 概率模型 | 隐马尔可夫模型，序列建模 |

---

## 工作流程 (Step-by-Step Workflow)

### 方式一: 26特征实验 (完整流程)

使用 `src/mouse_behavior_classification.ipynb`

```
Step 1: 环境设置
├── 安装依赖 (torch, lightgbm, xgboost, scikit-learn等)
└── 导入模块 (label_parser, feature_extraction, data_loader, models)

Step 2: 实验配置
├── 选择实验类型: EXPERIMENT = "behavior" 或 "aggression"
├── 设置数据路径: CSV_FOLDER, ANNOT_FOLDER
└── 设置训练参数: N_RUNS, N_EPOCHS, BATCH_SIZE

Step 3: 数据加载与预处理
├── prepare_dataset() 加载数据
│   ├── 从DLC CSV提取26个特征 (feature_extraction.py)
│   ├── 解析标注文件 (label_parser.py)
│   └── 对齐特征和标签，过滤无效样本
├── 可视化类别分布
└── 创建DataLoader (train/val/test split)

Step 4: 模型训练
├── 定义训练函数 train_pytorch_model()
├── 遍历多个模型 (MLP, LSTM, CNN, Transformer, LightGBM等)
├── 每个模型运行N_RUNS次 (不同随机种子)
└── 计算accuracy, weighted_f1, macro_f1

Step 5: 结果可视化
├── 生成模型对比图 (with error bars)
├── 生成Per-Class F1图
├── 生成混淆矩阵
└── 保存统计表格
```

**代码示例:**
```python
from src import prepare_dataset, create_data_loaders, get_pytorch_model

# 加载数据
X, y, feature_names, class_info = prepare_dataset(
    'data/dlc_csv', 
    'data/annotations',
    experiment='aggression'  # 或 'behavior'
)

# 创建数据加载器
train_loader, val_loader, test_loader, scaler = create_data_loaders(X, y)

# 创建模型
model = get_pytorch_model('mlp', n_features=26, n_classes=7)
```

---

### 方式二: 8特征实验 (快速实验)

使用 `src/8feature_src/kaggle_model_comparison.ipynb`

```
Step 1: 环境设置
├── 安装依赖包
├── 设置LOKY_MAX_CPU_COUNT (Windows兼容)
└── 检查CUDA可用性

Step 2: 数据加载
├── 加载特征文件: feature8_58.xlsx
├── 加载标签文件: merged_labels_aggression.xlsx
└── 对齐长度，检查类别分布

Step 3: 数据过滤与映射
├── EXPERIMENT_MODE = "behavior" 或 "aggression"
├── 移除class 0 (base类)
├── 重映射标签到连续范围 [0, n_classes-1]
└── 打印类别映射表

Step 4: 模型定义
├── PyTorch模型: BehaviorLSTM, BehaviorCNN
├── 传统ML模型: run_gmm_experiment, run_lightgbm_experiment, ...
└── 定义compute_metrics()计算评估指标

Step 5: 多次运行实验
├── run_multiple_experiments() 运行5次
├── 每次使用不同的split_seed
├── 收集accuracy, weighted_f1, macro_f1
└── 计算mean ± std

Step 6: 可视化与统计
├── create_comparison_graphs() 生成4张对比图
│   ├── overall.png: 总体性能对比
│   ├── per_class.png: Per-Class F1
│   ├── best_worst.png: 最佳/最差类别对比
│   └── stability.png: 稳定性(变异系数)
└── create_detailed_statistics_table() 打印详细统计表
```

---

## 安装与运行 (Installation & Usage)

### 1. 安装依赖
```bash
pip install torch lightgbm xgboost scikit-learn pandas numpy matplotlib seaborn hmmlearn openpyxl
```

### 2. 准备数据
将DLC CSV文件放入 `data/dlc_csv/`，标注文件放入 `data/annotations/`

或者下载预处理数据集放入 `data/dataset58/`

### 3. 运行实验

**26特征实验:**
```bash
# 在Jupyter中运行
jupyter notebook src/mouse_behavior_classification.ipynb
```

**8特征实验:**
```bash
# 在Jupyter/Kaggle中运行
jupyter notebook src/8feature_src/kaggle_model_comparison.ipynb

# 或运行Python脚本
cd src/8feature_src
python model_comparison_8models.py
```

### 4. Kaggle使用

修改notebook中的数据路径：
```python
# 8特征实验
feature_file = "/kaggle/input/mouse-behavior/dataset58/feature8_58.xlsx"
label_file = "/kaggle/input/mouse-behavior/dataset58/merged_labels_aggression.xlsx"

# 26特征实验
CSV_FOLDER = "/kaggle/input/mouse-behavior/dlc_csv"
ANNOT_FOLDER = "/kaggle/input/mouse-behavior/annotations"
```

---

## 标注文件格式 (Annotation Format)

```
Caltech Behavior Annotator - Annotation File

S1:	start	end	type
-----------------------------
   	1	943	base
   	944	1142	nonsocial
   	1143	1233	social
   	1234	1500	aggression
...

S2:	start	end	type
-----------------------------
   	1	6376	base
   	6377	6441	lateralthreat
   	6442	6500	keepdown
...
```

- **S1**: Behavior层级标注 (base/aggression/social/nonsocial)
- **S2**: Aggression细分标注 (lateralthreat/keepdown/clinch/uprightposture/freezing/bite/chase)

---

## 输出结果 (Outputs)

### 模型检查点
- PyTorch模型: `*.pth`
- LightGBM: `*.pkl` (model + scaler)

### 可视化图表
- `model_comparison_overall.png` - 总体性能对比
- `model_comparison_per_class.png` - Per-Class F1
- `model_comparison_best_worst.png` - 最佳/最差类别
- `model_comparison_stability.png` - 稳定性分析

### 统计表格
```
DETAILED STATISTICS TABLE (5 runs, mean ± std)
================================================================================
Model        Accuracy         Weighted F1      Macro F1         ...
--------------------------------------------------------------------------------
GMM          0.3521±0.0156    0.3412±0.0189    0.3156±0.0201    ...
LSTM         0.5623±0.0234    0.5512±0.0267    0.5234±0.0289    ...
CNN          0.5834±0.0198    0.5723±0.0223    0.5456±0.0245    ...
LightGBM     0.6123±0.0145    0.6012±0.0167    0.5789±0.0189    ...
...
```

---

## 可复现性 (Reproducibility)

- 每次实验运行5次，使用不同随机种子 (42, 43, 44, 45, 46)
- 使用分层抽样 (stratified split) 保持类别比例
- 标准化仅在训练集上fit，避免数据泄露
- Error bars表示标准差 (ddof=1)

---

## 许可证 (License)

MIT License

---

## 引用 (Citation)

如果使用本代码或数据集，请引用：

```bibtex
@misc{mouse_behavior_classification,
  author = {Blossom0913},
  title = {Mouse Behavior Classification},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/Blossom0913/Mouse-Behavior-Classifier-Train}
}
```

Dataset DOI: https://doi.org/10.6084/m9.figshare.30393298

- Dataset: [https://doi.org/10.6084/m9.figshare.30393298](https://doi.org/10.6084/m9.figshare.30393298.v2)
- Code: add your preferred software citation (e.g., Zenodo DOI if archived)

Example citation format:

```
Author(s). DeepLabVideo: Mouse Behavior Classification (Version YYYY.MM). Repository name. URL
Dataset: Figshare. DOI: 10.6084/m9.figshare.30393298
```

## License

Specify your license here (e.g., MIT). If none is provided, all rights reserved by default.
