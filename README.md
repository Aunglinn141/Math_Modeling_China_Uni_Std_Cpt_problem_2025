# Math Modeling Project

Choose your language: [English](#english) | [简体中文](#简体中文)

License: [MIT License](LICENSE.md)

<a id="english"></a>
## English

### Overview

This repository contains a math modeling project built around **NIPT (Non-Invasive Prenatal Testing)** data. It includes data cleaning, statistical analysis, visualization, grouped optimization, enhanced modeling, and classification tasks.

The project is organized into four problem-oriented modules, `Q_1` through `Q_4`, with shared preprocessing work located in `Data_Cleaned`.

### Project Workflow

1. Clean the raw Excel data, standardize fields, parse gestational week information, apply quality-control rules, and split male and female fetal samples.
2. Analyze the relationship between Y chromosome concentration, gestational week, and BMI in male fetal samples.
3. Estimate more suitable testing weeks for different BMI groups.
4. Extend the analysis with bootstrap methods, threshold sensitivity analysis, personalized recommendations, and multiple models.
5. Build a chromosome abnormality detection model for female fetal samples and export performance reports.

### Directory Structure

```text
Math_Modeling/
├─ Data_Cleaned/                 # Raw data cleaning and preprocessing
│  ├─ 附件.xlsx
│  ├─ data_cleaned.py
│  └─ output/
├─ Q_1/                          # Problem 1: Y chromosome concentration analysis
│  ├─ Question_1.py
│  └─ output/
├─ Q_2/                          # Problem 2: Optimal testing week by BMI group
│  ├─ question_2_1.py
│  └─ nipt_outputs/
├─ Q_3/                          # Problem 3: Enhanced modeling and personalization
│  ├─ question_3.py
│  └─ nipt_enhanced_outputs/
├─ Q_4/                          # Problem 4: Female fetal abnormality detection
│  ├─ question_4.py
│  ├─ problem4_flowchart.svg
│  └─ q4_outputs_fixed/
├─ README.md
└─ license.md
```

### Module Summary

#### `Data_Cleaned`

Main responsibilities:

- Standardize Chinese and English column mappings
- Parse date and gestational-week fields
- Normalize height, weight, and BMI values
- Process laboratory indicators and quality-control conditions
- Infer fetal sex and split male and female samples
- Export cleaned full and subset datasets

Main outputs:

- `output/nipt_cleaned.csv`
- `output/male_tests.csv`
- `output/female_tests.csv`
- `output/earliest_male_threshold.csv`

#### `Q_1`

Focus:

- Descriptive statistics
- Correlation analysis
- Multiple linear regression
- BMI group comparison
- 2D and 3D visualization

Main outputs:

- `output/3d_scatter_plot.png`
- `output/3d_surface_plot.png`
- `output/modern_2d_analysis.png`
- `output/correlation_network.png`
- `output/bmi_group_analysis.png`
- `Y染色体浓度分析结果_中文版本.xlsx`

#### `Q_2`

Focus:

- BMI grouping and dynamic binning
- Probability modeling for target attainment
- Bootstrap resampling
- Optimal testing-week estimation by group
- Exported plots and summary reports

Main outputs:

- `nipt_outputs/per_group_optimal_weeks.csv`
- `nipt_outputs/group_analysis_reports.json`
- `nipt_outputs/group_analysis_report.txt`
- `nipt_outputs/overview/`
- `nipt_outputs/plots/`
- `nipt_outputs/bootstrap/`

#### `Q_3`

Focus:

- End-to-end feature engineering
- Quality scoring and threshold sensitivity analysis
- Bootstrap confidence interval estimation
- Logistic regression, random forest, neural network, and survival analysis
- Personalized testing recommendations

Main outputs:

- `nipt_enhanced_outputs/model_performance_summary.csv`
- `nipt_enhanced_outputs/optimal_weeks_by_bmi.csv`
- `nipt_enhanced_outputs/personalized_recommendations.csv`
- `nipt_enhanced_outputs/threshold_sensitivity.csv`
- `nipt_enhanced_outputs/detection_error_analysis.csv`
- `nipt_enhanced_outputs/reports/comprehensive_analysis_report.md`
- `nipt_enhanced_outputs/plots/`

#### `Q_4`

Focus:

- Automatic feature and label identification
- Abnormal-label construction using Z-score, percentile, or risk score
- Model training with logistic regression, XGBoost, LightGBM, and related methods
- Performance evaluation under target false-positive-rate constraints
- ROC, PR, and feature-importance exports

Main outputs:

- `q4_outputs_fixed/model_summary.json`
- `q4_outputs_fixed/model_config.json`
- `q4_outputs_fixed/performance_report.txt`
- `q4_outputs_fixed/classification_report.txt`
- `q4_outputs_fixed/confusion_matrix.csv`
- `q4_outputs_fixed/feature_importance.csv`
- `q4_outputs_fixed/female_roc_curve.png`
- `q4_outputs_fixed/female_pr_curve.png`

### Environment

Recommended environment:

- Python 3.10 or later
- Windows is recommended for the current scripts
- Chinese fonts such as `SimHei` or `Microsoft YaHei` may be needed for figure rendering

Recommended dependencies:

```bash
pip install pandas numpy scipy matplotlib seaborn statsmodels scikit-learn openpyxl
pip install torch xgboost lightgbm lifelines
```

Notes:

- `torch`, `xgboost`, `lightgbm`, and `lifelines` are optional for enhanced models.
- If you only need the basic analysis pipeline, the first install command is usually enough.

### How to Run

The scripts rely heavily on **relative paths**, so run them from their own directories.

#### 1. Data Cleaning

```bash
cd Data_Cleaned
python data_cleaned.py
```

#### 2. Problem 1

```bash
cd Q_1
python Question_1.py
```

#### 3. Problem 2

```bash
cd Q_2
python question_2_1.py
```

#### 4. Problem 3

```bash
cd Q_3
python question_3.py
```

#### 5. Problem 4

```bash
cd Q_4
python question_4.py
```

To view supported command-line arguments for Problem 4:

```bash
cd Q_4
python question_4.py --help
```

### Data and Results

- The raw source file is `Data_Cleaned/附件.xlsx`.
- Cleaned intermediate data is stored in `Data_Cleaned/output/`.
- Each problem folder includes its own output directory for independent execution and reproduction.
- Some generated result files are already kept in the repository for direct inspection.

### Suggested Usage Order

Run the project in this order:

`Data_Cleaned -> Q_1 -> Q_2 -> Q_3 -> Q_4`

### License

This project is released under the MIT License. See [license.md](LICENSE.md).

---

<a id="简体中文"></a>
## 简体中文

### 项目简介

本仓库是一个围绕 **NIPT（无创产前检测）数据** 展开的数学建模项目，包含数据清洗、统计分析、可视化、分组优化、增强建模以及分类预测等内容。

项目按题目拆分为 `Q_1` 到 `Q_4` 四个模块，通用的数据预处理工作位于 `Data_Cleaned` 目录。

### 项目流程

1. 清洗原始 Excel 数据，统一字段，解析孕周信息，执行质控规则，并拆分男胎与女胎样本。
2. 基于男胎样本分析 Y 染色体浓度与孕周、BMI 等因素之间的关系。
3. 按 BMI 分组，估计不同分组下更合适的检测孕周。
4. 在增强分析中加入 Bootstrap、阈值敏感性分析、个体化建议和多模型建模。
5. 面向女胎样本构建染色体异常检测模型，并输出性能评估结果。

### 目录结构

```text
Math_Modeling/
├─ Data_Cleaned/                 # 原始数据清洗与预处理
│  ├─ 附件.xlsx
│  ├─ data_cleaned.py
│  └─ output/
├─ Q_1/                          # 问题一：Y染色体浓度分析
│  ├─ Question_1.py
│  └─ output/
├─ Q_2/                          # 问题二：BMI分组下的最优检测孕周
│  ├─ question_2_1.py
│  └─ nipt_outputs/
├─ Q_3/                          # 问题三：增强建模与个体化推荐
│  ├─ question_3.py
│  └─ nipt_enhanced_outputs/
├─ Q_4/                          # 问题四：女胎染色体异常检测
│  ├─ question_4.py
│  ├─ problem4_flowchart.svg
│  └─ q4_outputs_fixed/
├─ README.md
└─ license.md
```

### 模块说明

#### `Data_Cleaned`

主要功能：

- 统一中英文列名映射
- 解析日期与孕周字段
- 规范身高、体重和 BMI 等体征数据
- 处理实验室指标与质控条件
- 推断胎儿性别并拆分男胎和女胎样本
- 导出清洗后的总表及子样本表

主要输出：

- `output/nipt_cleaned.csv`
- `output/male_tests.csv`
- `output/female_tests.csv`
- `output/earliest_male_threshold.csv`

#### `Q_1`

主要内容：

- 描述性统计分析
- 相关性分析
- 多元线性回归
- BMI 分组比较
- 2D 与 3D 可视化输出

主要输出：

- `output/3d_scatter_plot.png`
- `output/3d_surface_plot.png`
- `output/modern_2d_analysis.png`
- `output/correlation_network.png`
- `output/bmi_group_analysis.png`
- `Y染色体浓度分析结果_中文版本.xlsx`

#### `Q_2`

主要内容：

- BMI 分组与动态分箱
- 达标概率建模
- Bootstrap 重采样
- 分组最优检测周估计
- 图形与汇总报告导出

主要输出：

- `nipt_outputs/per_group_optimal_weeks.csv`
- `nipt_outputs/group_analysis_reports.json`
- `nipt_outputs/group_analysis_report.txt`
- `nipt_outputs/overview/`
- `nipt_outputs/plots/`
- `nipt_outputs/bootstrap/`

#### `Q_3`

主要内容：

- 完整特征工程流程
- 质量评分与阈值敏感性分析
- Bootstrap 置信区间估计
- 逻辑回归、随机森林、神经网络、生存分析等模型
- 个体化检测建议生成

主要输出：

- `nipt_enhanced_outputs/model_performance_summary.csv`
- `nipt_enhanced_outputs/optimal_weeks_by_bmi.csv`
- `nipt_enhanced_outputs/personalized_recommendations.csv`
- `nipt_enhanced_outputs/threshold_sensitivity.csv`
- `nipt_enhanced_outputs/detection_error_analysis.csv`
- `nipt_enhanced_outputs/reports/comprehensive_analysis_report.md`
- `nipt_enhanced_outputs/plots/`

#### `Q_4`

主要内容：

- 自动识别特征列与标签列
- 基于 Z-score、百分位数或风险评分构造异常标签
- 使用逻辑回归、XGBoost、LightGBM 等模型训练
- 在目标假阳性率约束下评估模型性能
- 输出 ROC、PR 曲线和特征重要性结果

主要输出：

- `q4_outputs_fixed/model_summary.json`
- `q4_outputs_fixed/model_config.json`
- `q4_outputs_fixed/performance_report.txt`
- `q4_outputs_fixed/classification_report.txt`
- `q4_outputs_fixed/confusion_matrix.csv`
- `q4_outputs_fixed/feature_importance.csv`
- `q4_outputs_fixed/female_roc_curve.png`
- `q4_outputs_fixed/female_pr_curve.png`

### 运行环境

建议环境：

- Python 3.10 及以上
- 当前脚本更适合在 Windows 环境下运行
- 若图表涉及中文显示，建议安装 `SimHei`、`Microsoft YaHei` 等字体

建议安装依赖：

```bash
pip install pandas numpy scipy matplotlib seaborn statsmodels scikit-learn openpyxl
pip install torch xgboost lightgbm lifelines
```

说明：

- `torch`、`xgboost`、`lightgbm`、`lifelines` 属于增强分析所需的可选依赖。
- 如果只运行基础分析流程，通常第一条安装命令即可满足需求。

### 运行方式

各脚本大量依赖**相对路径**，建议进入对应目录后再执行。

#### 1. 数据清洗

```bash
cd Data_Cleaned
python data_cleaned.py
```

#### 2. 问题一

```bash
cd Q_1
python Question_1.py
```

#### 3. 问题二

```bash
cd Q_2
python question_2_1.py
```

#### 4. 问题三

```bash
cd Q_3
python question_3.py
```

#### 5. 问题四

```bash
cd Q_4
python question_4.py
```

如需查看问题四脚本支持的参数：

```bash
cd Q_4
python question_4.py --help
```

### 数据与结果

- 原始数据文件位于 `Data_Cleaned/附件.xlsx`。
- 清洗后的中间数据位于 `Data_Cleaned/output/`。
- 每个题目目录都包含各自的输出文件夹，便于独立运行和复现。
- 仓库中已保留部分结果文件，可直接查看已有图表与报告。

### 推荐运行顺序

建议按以下顺序执行：

`Data_Cleaned -> Q_1 -> Q_2 -> Q_3 -> Q_4`

### 许可证

本项目采用 MIT 许可证，详见 [license.md](LICENSE.md)。
