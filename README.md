# 数学建模项目说明

## 项目简介

本项目围绕 **NIPT（无创产前检测）数据** 展开，包含数据清洗、统计分析、可视化建模、分组优化分析以及分类预测等内容。仓库按照题目拆分为 `Q_1` 到 `Q_4` 四个部分，前置的数据清洗工作位于 `Data_Cleaned` 目录。

项目整体流程如下：

1. 对原始 Excel 数据进行字段统一、缺失处理、孕周解析、质控判断与男女胎样本拆分。
2. 基于男胎样本分析 Y 染色体浓度与孕周、BMI 等因素的关系。
3. 按 BMI 分组，估计不同分组下更合适的检测孕周。
4. 在增强分析中引入 Bootstrap、阈值敏感性、个体化建议和多模型建模。
5. 针对女胎样本构建染色体异常检测模型并输出性能评估结果。

## 目录结构

```text
Math_Modeling/
├─ Data_Cleaned/                 # 原始数据清洗与预处理
│  ├─ 附件.xlsx
│  ├─ data_cleaned.py
│  └─ output/
├─ Q_1/                          # 问题一：Y染色体浓度与孕周/BMI关系分析
│  ├─ Question_1.py
│  └─ output/
├─ Q_2/                          # 问题二：BMI分组下的最优检测孕周分析
│  ├─ question_2_1.py
│  └─ nipt_outputs/
├─ Q_3/                          # 问题三：增强建模与个体化推荐
│  ├─ question_3.py
│  └─ nipt_enhanced_outputs/
├─ Q_4/                          # 问题四：女胎染色体异常检测模型
│  ├─ question_4.py
│  ├─ problem4_flowchart.svg
│  └─ q4_outputs_fixed/
└─ README.md
```

## 各部分说明

### 1. `Data_Cleaned`

用于原始数据预处理，主要功能包括：

- 统一中英文列名映射
- 解析日期与孕周字段
- 规范身高、体重、BMI 等体征数据
- 处理实验室指标与质控条件
- 推断胎儿性别并拆分男胎/女胎样本
- 导出清洗后的总表及子样本表

主要输出文件：

- `output/nipt_cleaned.csv`
- `output/male_tests.csv`
- `output/female_tests.csv`
- `output/earliest_male_threshold.csv`

### 2. `Q_1`

问题一重点研究 **Y 染色体浓度、孕周、BMI** 之间的关系，包含：

- 描述性统计分析
- 相关性分析
- 多元线性回归
- BMI 分组比较
- 2D / 3D 可视化图表输出

主要输出文件：

- `output/3d_scatter_plot.png`
- `output/3d_surface_plot.png`
- `output/modern_2d_analysis.png`
- `output/correlation_network.png`
- `output/bmi_group_analysis.png`
- `Y染色体浓度分析结果_中文版本.xlsx`

### 3. `Q_2`

问题二在男胎样本基础上，按 BMI 进行分箱和分组分析，目标是寻找更合理的检测孕周。脚本包含：

- BMI 分组与动态分箱
- 达标概率建模
- Bootstrap 重采样
- 分组最优检测周估计
- 不同分组的图形化结果导出

主要输出目录：

- `nipt_outputs/per_group_optimal_weeks.csv`
- `nipt_outputs/group_analysis_reports.json`
- `nipt_outputs/group_analysis_report.txt`
- `nipt_outputs/overview/`
- `nipt_outputs/plots/`
- `nipt_outputs/bootstrap/`

### 4. `Q_3`

问题三是在问题二基础上的增强分析版本，强调更丰富的特征工程与模型融合，主要包括：

- 完整数据处理与特征工程
- 质量评分与阈值敏感性分析
- Bootstrap 置信区间估计
- 逻辑回归、随机森林、神经网络、生存分析等模型
- 个体化检测建议生成

主要输出目录：

- `nipt_enhanced_outputs/model_performance_summary.csv`
- `nipt_enhanced_outputs/optimal_weeks_by_bmi.csv`
- `nipt_enhanced_outputs/personalized_recommendations.csv`
- `nipt_enhanced_outputs/threshold_sensitivity.csv`
- `nipt_enhanced_outputs/detection_error_analysis.csv`
- `nipt_enhanced_outputs/reports/comprehensive_analysis_report.md`
- `nipt_enhanced_outputs/plots/`

### 5. `Q_4`

问题四面向 **女胎染色体异常检测**，核心工作包括：

- 自动识别字段与标签列
- 基于 Z-score、百分位数或风险评分构造异常标签
- 使用逻辑回归/XGBoost/LightGBM 等模型训练
- 在目标假阳性率约束下评估模型性能
- 输出 ROC、PR 曲线与特征重要性分析

主要输出目录：

- `q4_outputs_fixed/model_summary.json`
- `q4_outputs_fixed/model_config.json`
- `q4_outputs_fixed/performance_report.txt`
- `q4_outputs_fixed/classification_report.txt`
- `q4_outputs_fixed/confusion_matrix.csv`
- `q4_outputs_fixed/feature_importance.csv`
- `q4_outputs_fixed/female_roc_curve.png`
- `q4_outputs_fixed/female_pr_curve.png`

## 运行环境

建议环境：

- Python 3.10 或以上
- Windows 环境下运行体验更好
- 已安装中文字体，如 `SimHei`、`Microsoft YaHei`

建议安装依赖：

```bash
pip install pandas numpy scipy matplotlib seaborn statsmodels scikit-learn openpyxl
pip install torch xgboost lightgbm lifelines
```

说明：

- `torch`、`xgboost`、`lightgbm`、`lifelines` 属于增强功能依赖，未安装时部分模型会自动关闭。
- 若仅运行基础分析，前一条依赖通常即可满足。

## 运行方式

注意：各脚本大量使用**相对路径**，建议进入对应目录后再运行。

### 1. 数据清洗

```bash
cd Data_Cleaned
python data_cleaned.py
```

### 2. 问题一分析

```bash
cd Q_1
python Question_1.py
```

### 3. 问题二分析

```bash
cd Q_2
python question_2_1.py
```

### 4. 问题三增强分析

```bash
cd Q_3
python question_3.py
```

### 5. 问题四分类建模

```bash
cd Q_4
python question_4.py
```

如需查看问题四脚本支持的参数，可执行：

```bash
cd Q_4
python question_4.py --help
```

## 数据与结果说明

- 原始数据文件位于 `Data_Cleaned/附件.xlsx`。
- 清洗后的中间数据位于 `Data_Cleaned/output/`。
- 每个题目目录都包含各自的输出文件夹，便于独立运行和复现。
- 仓库中已保留部分结果文件，可直接查看已有分析图表与报告。

## 使用建议

- 推荐按 `Data_Cleaned -> Q_1 -> Q_2 -> Q_3 -> Q_4` 的顺序运行。
- 如果出现中文乱码，优先检查终端编码、文件编码和 matplotlib 中文字体配置。
- 如果某些高级模型不可用，先确认对应依赖是否安装成功。

## 项目特点

- 以真实建模流程组织代码，包含数据清洗、统计分析、机器学习与结果可视化。
- 每道题目相对独立，便于拆分展示、答辩汇报或单独调试。
- 输出结果较完整，适合用于论文写作、图表整理和建模报告撰写。
