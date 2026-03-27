from __future__ import annotations

import argparse
import json
import logging
import os
import re
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import DataConversionWarning

# 可选模型 (如果安装了就使用)
try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier

    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

# 忽略警告
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """配置类，包含所有模型参数和列名映射"""
    # 特征列 (自动检测; 如果未找到，需要手动设置)
    z13: Optional[str] = None
    z18: Optional[str] = None
    z21: Optional[str] = None
    zx: Optional[str] = None
    gc: Optional[str] = None
    reads: Optional[str] = None
    reads_ratio: Optional[str] = None
    bmi: Optional[str] = None
    maternal_age: Optional[str] = None
    gestational_weeks: Optional[str] = None

    # 标签和性别列
    label: Optional[str] = None  # 1=异常, 0=正常
    fetal_sex: Optional[str] = None  # 'F'/'M' 或 '女'/'男'

    # 新增：异常检测参数
    zscore_threshold: float = 2.5  # Z-score阈值
    abnormal_detection_method: str = "zscore"  # 'zscore' | 'percentile' | 'risk_score'
    percentile_threshold: float = 95.0  # 百分位数阈值
    force_create_labels: bool = False  # 强制创建异常标签

    # 模型参数
    target_fpr: float = 0.01  # 目标假阳性率上限
    model: str = "lr"  # 'lr' | 'xgb' | 'lgbm'
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42

    def __post_init__(self):
        """验证配置参数"""
        if not 0 < self.target_fpr < 1:
            raise ValueError("target_fpr 必须在 (0, 1) 范围内")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size 必须在 (0, 1) 范围内")
        if self.cv_folds < 2:
            raise ValueError("cv_folds 必须 >= 2")


class DataProcessor:
    """数据处理器，负责数据清洗和预处理"""

    @staticmethod
    def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """标准化列名：去除空格，转换为小写"""
        df = df.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]
        return df

    @staticmethod
    def guess_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """
        灵活匹配列名，支持正则表达式和模糊匹配

        Args:
            df: DataFrame
            candidates: 候选列名模式列表

        Returns:
            匹配到的列名，如果没找到返回 None
        """
        cols = list(df.columns)

        # 1. 精确匹配 (不区分大小写)
        for candidate in candidates:
            for col in cols:
                if col.lower() == candidate.lower():
                    return col

        # 2. 正则表达式匹配
        for candidate in candidates:
            try:
                pattern = re.compile(candidate, flags=re.IGNORECASE)
                for col in cols:
                    if pattern.search(col):
                        return col
            except re.error:
                continue

        return None

    @staticmethod
    def map_label_series(y: pd.Series, label_col_name: str) -> pd.Series:
        """
        将各种可能的标签转换为 {0, 1}，其中 1 = 异常/不健康

        Args:
            y: 标签 Series
            label_col_name: 标签列名，用于推断标签含义

        Returns:
            转换后的标签 Series
        """
        y = y.copy()

        logger.info(f"处理标签列 '{label_col_name}', 原始数据类型: {y.dtype}")
        logger.info(f"原始标签唯一值: {sorted(y.dropna().unique())}")
        logger.info(f"原始标签分布:\n{y.value_counts()}")

        # 如果已经是数值型且在 {0, 1} 范围内
        if pd.api.types.is_numeric_dtype(y):
            y_clean = y.dropna()
            if len(y_clean) > 0 and y_clean.isin([0, 1]).all():
                # 检查是否需要反转 - 扩展健康相关关键词
                health_keywords = [
                    'health', 'healthy', '健康', 'after_birth', 'outcome',
                    'normal', 'good', 'well', 'fine', 'birth'
                ]
                needs_invert = any(keyword in label_col_name.lower() for keyword in health_keywords)

                if needs_invert:
                    logger.info(f"检测到健康相关标签列 '{label_col_name}'，将进行反转 (健康=0 -> 异常=1)")
                    y_inverted = y.map(lambda x: 1 - x if pd.notna(x) else np.nan)
                    logger.info(f"反转后标签分布:\n{pd.Series(y_inverted).value_counts()}")
                    return y_inverted.astype('Int64')
                else:
                    logger.info("未检测到健康相关关键词，保持原始标签")
                    return y.astype('Int64')

        # 定义映射字典
        positive_mapping = {
            # 异常相关
            "abnormal": 1, "abn": 1, "pos": 1, "positive": 1,
            "异常": 1, "不健康": 1, "阳性": 1, "trisomy": 1,
            "disease": 1, "sick": 1, "unhealthy": 1,
            # 中文
            "是": 1, "有": 1, "存在": 1,
        }

        negative_mapping = {
            # 正常相关
            "normal": 0, "neg": 0, "negative": 0,
            "正常": 0, "健康": 0, "阴性": 0,
            "healthy": 0, "good": 0,
            # 中文
            "否": 0, "无": 0, "不存在": 0,
        }

        mapping = {**positive_mapping, **negative_mapping}

        def map_value(v):
            if pd.isna(v):
                return np.nan

            v_str = str(v).strip().lower()

            # 直接映射
            if v_str in mapping:
                return mapping[v_str]

            # 模糊匹配
            if any(pos_key in v_str for pos_key in positive_mapping.keys()):
                return 1
            if any(neg_key in v_str for neg_key in negative_mapping.keys()):
                return 0

            # 数值转换
            try:
                float_val = float(v_str)
                if float_val in [0.0, 1.0]:
                    return int(float_val)
            except (ValueError, TypeError):
                pass

            return np.nan

        y_mapped = y.map(map_value)

        # 如果标签列名暗示这是"健康"标签，需要反转
        health_keywords = [
            'health', 'healthy', '健康', 'after_birth', 'outcome',
            'normal', 'good', 'well', 'fine', 'birth'
        ]
        if any(keyword in label_col_name.lower() for keyword in health_keywords):
            logger.info("检测到健康相关标签，将进行反转")
            y_mapped = y_mapped.map(lambda x: 1 - x if pd.notna(x) else np.nan)

        logger.info(f"映射后标签分布:\n{pd.Series(y_mapped).value_counts()}")
        return y_mapped.astype('Int64')

    @staticmethod
    def filter_female_samples(df: pd.DataFrame, sex_col: str) -> pd.DataFrame:
        """筛选女胎样本"""

        def is_female(value):
            if pd.isna(value):
                return False
            v_str = str(value).strip().lower()
            female_indicators = {"f", "female", "女", "woman", "0", "girl"}
            return v_str in female_indicators

        mask = df[sex_col].map(is_female)
        filtered_df = df[mask].copy()

        logger.info(f"筛选前样本数: {len(df)}, 筛选后女胎样本数: {len(filtered_df)}")
        return filtered_df


class AbnormalityLabelCreator:
    """异常标签创建器 - 解决单类别问题的核心模块"""

    @staticmethod
    def detect_single_class_problem(y: pd.Series) -> bool:
        """检测是否存在单类别问题"""
        unique_classes = y.dropna().nunique()
        return unique_classes <= 1

    @staticmethod
    def create_zscore_based_labels(df: pd.DataFrame, config: Config) -> pd.DataFrame:
        """基于Z-score创建异常标签"""
        logger.info(f"基于Z-score创建异常标签 (阈值={config.zscore_threshold})")

        df = df.copy()

        # 找到Z-score列
        z_cols = [col for col in [config.z13, config.z18, config.z21, config.zx]
                  if col is not None and col in df.columns]

        if not z_cols:
            logger.warning("未找到Z-score列，无法创建异常标签")
            return df

        # 计算每个样本是否有任何染色体异常
        abnormal_mask = pd.Series(False, index=df.index)

        abnormal_counts = {}
        for col in z_cols:
            col_abnormal = abs(df[col]) > config.zscore_threshold
            abnormal_mask = abnormal_mask | col_abnormal
            abnormal_counts[col] = col_abnormal.sum()
            logger.info(f"{col} 异常样本数: {col_abnormal.sum()}")

        # 创建新的标签列
        df['zscore_abnormal'] = abnormal_mask.astype(int)

        total_abnormal = abnormal_mask.sum()
        abnormal_rate = abnormal_mask.mean()

        logger.info(f"总异常样本数: {total_abnormal}")
        logger.info(f"异常率: {abnormal_rate:.3%}")

        if total_abnormal == 0:
            logger.warning("基于当前阈值未发现异常样本，尝试降低阈值")
            # 自动降低阈值
            for new_threshold in [2.0, 1.5, 1.0]:
                abnormal_mask = pd.Series(False, index=df.index)
                for col in z_cols:
                    col_abnormal = abs(df[col]) > new_threshold
                    abnormal_mask = abnormal_mask | col_abnormal

                if abnormal_mask.sum() > 0:
                    df['zscore_abnormal'] = abnormal_mask.astype(int)
                    logger.info(f"使用阈值 {new_threshold}，异常样本数: {abnormal_mask.sum()}")
                    break

        return df

    @staticmethod
    def create_percentile_based_labels(df: pd.DataFrame, config: Config) -> pd.DataFrame:
        """基于百分位数创建异常标签"""
        logger.info(f"基于{config.percentile_threshold}%分位数创建异常标签")

        df = df.copy()

        z_cols = [col for col in [config.z13, config.z18, config.z21, config.zx]
                  if col is not None and col in df.columns]

        if not z_cols:
            return df

        abnormal_mask = pd.Series(False, index=df.index)

        for col in z_cols:
            threshold = np.percentile(abs(df[col].dropna()), config.percentile_threshold)
            col_abnormal = abs(df[col]) > threshold
            abnormal_mask = abnormal_mask | col_abnormal
            logger.info(f"{col} 阈值: {threshold:.3f}, 异常样本数: {col_abnormal.sum()}")

        df['percentile_abnormal'] = abnormal_mask.astype(int)

        logger.info(f"总异常样本数: {abnormal_mask.sum()}")
        logger.info(f"异常率: {abnormal_mask.mean():.3%}")

        return df

    @staticmethod
    def create_risk_score_labels(df: pd.DataFrame, config: Config) -> pd.DataFrame:
        """基于综合风险评分创建标签"""
        logger.info("基于综合风险评分创建标签")

        df = df.copy()

        z_cols = [col for col in [config.z13, config.z18, config.z21, config.zx]
                  if col is not None and col in df.columns]

        if len(z_cols) < 2:
            logger.warning("Z-score列不足，无法计算综合风险")
            return df

        # 计算综合风险评分 (所有Z-score的平方和的平方根)
        risk_scores = np.zeros(len(df))

        for col in z_cols:
            z_values = df[col].fillna(0)
            risk_scores += z_values ** 2

        risk_scores = np.sqrt(risk_scores)
        df['risk_score'] = risk_scores

        # 基于风险评分的上分位数创建标签
        threshold_95 = np.percentile(risk_scores, 95)

        df['risk_score_abnormal'] = (risk_scores > threshold_95).astype(int)

        abnormal_count = (risk_scores > threshold_95).sum()
        logger.info(f"高风险样本数: {abnormal_count}")
        logger.info(f"高风险率: {abnormal_count / len(df):.3%}")

        return df


class FeatureEngineer:
    """特征工程类"""

    def __init__(self, config: Config):
        self.config = config

    def autodetect_columns(self, df: pd.DataFrame) -> Config:
        """自动检测列名"""
        df = DataProcessor.normalize_columns(df)
        cfg = self.config

        # Z-scores 检测
        cfg.z13 = DataProcessor.guess_column(df, [r"^z.*13$", "z13", "chr13.*z", "z_13", r"13.*z"])
        cfg.z18 = DataProcessor.guess_column(df, [r"^z.*18$", "z18", "chr18.*z", "z_18", r"18.*z"])
        cfg.z21 = DataProcessor.guess_column(df, [r"^z.*21$", "z21", "chr21.*z", "z_21", r"21.*z"])
        cfg.zx = DataProcessor.guess_column(df, [r"^z.*x$", "zx", "chrx.*z", "z_x", r"x.*z"])

        # GC 含量和测序深度
        cfg.gc = DataProcessor.guess_column(df, [r"^gc(_?content)?$", r"gc.*含量", "gc含量"])
        cfg.reads = DataProcessor.guess_column(df, [
            "^reads$", "read_count", "total_reads", "n_reads",
            "测序读数", "reads_count", "sequencing_depth"
        ])
        cfg.reads_ratio = DataProcessor.guess_column(df, [
            "reads_ratio", "mapping_ratio", "unique_ratio",
            "比对率", "通过率", "mapping_rate"
        ])

        # 临床特征
        cfg.bmi = DataProcessor.guess_column(df, ["^bmi$", "body_mass_index", "体重指数"])
        cfg.maternal_age = DataProcessor.guess_column(df, [
            "maternal_age", "^age$", "mother_age",
            "母亲年龄", "孕妇年龄", "妈妈年龄"
        ])
        cfg.gestational_weeks = DataProcessor.guess_column(df, [
            "gestational_age", "gestational_weeks", "ga",
            "孕周", "妊娠周", "怀孕周数"
        ])

        # 标签和性别
        cfg.label = DataProcessor.guess_column(df, [
            "^label$", "is_abnormal", "abnormal", "health", "healthy",
            "是否健康", "是否异常", "result", "最终结果",
            "产后结果", "分娩结果", "diagnosis", "outcome",
            # 新增异常标签列
            "zscore_abnormal", "percentile_abnormal", "risk_score_abnormal"
        ])
        cfg.fetal_sex = DataProcessor.guess_column(df, [
            "fetal_sex", "sex", "gender", "胎儿性别",
            "胎性别", "baby_sex", "child_gender"
        ])

        # 记录检测结果
        detected_features = []
        for field_name, field_value in asdict(cfg).items():
            if (field_value is not None and
                    field_name not in ['target_fpr', 'model', 'test_size', 'cv_folds', 'random_state',
                                       'zscore_threshold', 'abnormal_detection_method', 'percentile_threshold',
                                       'force_create_labels']):
                detected_features.append(f"{field_name}: {field_value}")

        logger.info(f"自动检测到的列: {', '.join(detected_features)}")
        return cfg

    def build_features_and_labels(self, df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """构建特征矩阵和标签向量"""

        # 首先尝试处理原始标签
        original_label_processed = False
        y_clean = None

        if cfg.label is not None and cfg.label in df.columns:
            try:
                y = DataProcessor.map_label_series(df[cfg.label], cfg.label)

                # 检查是否存在单类别问题
                if not AbnormalityLabelCreator.detect_single_class_problem(y):
                    # 没有单类别问题，使用原始标签
                    valid_mask = ~y.isna()
                    if valid_mask.any():
                        y_clean = y.loc[valid_mask].copy().astype(int)
                        df = df.loc[valid_mask].copy()
                        original_label_processed = True
                        logger.info("使用原始标签，未发现单类别问题")
            except Exception as e:
                logger.warning(f"处理原始标签失败: {e}")

        # 如果原始标签有问题或强制创建新标签，则创建基于Z-score的标签
        if not original_label_processed or cfg.force_create_labels:
            logger.info("检测到单类别问题或强制创建标签，基于Z-score创建异常标签")

            # 创建异常标签
            if cfg.abnormal_detection_method == "zscore":
                df = AbnormalityLabelCreator.create_zscore_based_labels(df, cfg)
                new_label_col = 'zscore_abnormal'
            elif cfg.abnormal_detection_method == "percentile":
                df = AbnormalityLabelCreator.create_percentile_based_labels(df, cfg)
                new_label_col = 'percentile_abnormal'
            elif cfg.abnormal_detection_method == "risk_score":
                df = AbnormalityLabelCreator.create_risk_score_labels(df, cfg)
                new_label_col = 'risk_score_abnormal'
            else:
                # 默认使用zscore方法
                df = AbnormalityLabelCreator.create_zscore_based_labels(df, cfg)
                new_label_col = 'zscore_abnormal'

            if new_label_col not in df.columns:
                raise ValueError(f"创建异常标签失败，未找到列 {new_label_col}")

            y_clean = df[new_label_col].astype(int)
            cfg.label = new_label_col  # 更新配置中的标签列名

        # 检查最终标签是否仍有单类别问题
        if AbnormalityLabelCreator.detect_single_class_problem(y_clean):
            logger.warning("仍然存在单类别问题，尝试降低阈值")
            # 尝试更宽松的阈值
            cfg.zscore_threshold = 1.5
            df = AbnormalityLabelCreator.create_zscore_based_labels(df, cfg)
            if 'zscore_abnormal' in df.columns:
                y_clean = df['zscore_abnormal'].astype(int)
                cfg.label = 'zscore_abnormal'

        # 最终检查
        if AbnormalityLabelCreator.detect_single_class_problem(y_clean):
            raise ValueError(
                "创建异常标签后仍然存在单类别问题。请检查数据或调整参数。\n"
                f"当前标签分布: {y_clean.value_counts()}"
            )

        # 收集所有特征列
        feature_candidates = [
            cfg.z13, cfg.z18, cfg.z21, cfg.zx, cfg.gc,
            cfg.reads, cfg.reads_ratio, cfg.bmi,
            cfg.maternal_age, cfg.gestational_weeks
        ]

        feature_cols = [col for col in feature_candidates
                        if col is not None and col in df.columns]

        if not feature_cols:
            raise ValueError(
                "未找到任何有效特征列。请检查数据文件或在 Config 中手动指定列名。\n"
                f"可用列名: {list(df.columns)}"
            )

        # 构建特征矩阵
        X = df[feature_cols].copy()

        # 检查特征缺失情况
        missing_info = X.isnull().sum()
        if missing_info.any():
            logger.info(f"特征缺失情况:\n{missing_info[missing_info > 0]}")

        logger.info(f"最终特征数: {len(feature_cols)}, 样本数: {len(X)}")
        logger.info(f"标签分布: {y_clean.value_counts().sort_index()}")

        return X, y_clean, feature_cols


# 其余的类保持不变，这里只展示修改的部分...
# (ModelBuilder, ThresholdOptimizer, BootstrapAnalyzer, Visualizer, ModelTrainer 等类保持原样)

class ModelBuilder:
    """模型构建器"""

    @staticmethod
    def create_model(cfg: Config, n_features: int) -> Tuple[str, Pipeline]:
        """创建机器学习模型管道"""

        if cfg.model == "lr":
            model = LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=cfg.random_state,
                solver='lbfgs'
            )
            pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", model)
            ])
            return "LogisticRegression", pipeline

        elif cfg.model == "xgb" and HAS_XGB:
            model = XGBClassifier(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                eval_metric="logloss",
                random_state=cfg.random_state,
                n_jobs=-1,
                verbosity=0
            )
            pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", model)
            ])
            return "XGBClassifier", pipeline

        elif cfg.model == "lgbm" and HAS_LGBM:
            model = LGBMClassifier(
                n_estimators=800,
                learning_rate=0.05,
                max_depth=-1,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.8,
                class_weight="balanced",
                random_state=cfg.random_state,
                n_jobs=-1,
                verbose=-1
            )
            pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", model)
            ])
            return "LGBMClassifier", pipeline

        else:
            # 回退到逻辑回归
            if cfg.model != "lr":
                logger.warning(f"模型 {cfg.model} 不可用，回退到逻辑回归")

            model = LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=cfg.random_state
            )
            pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", model)
            ])
            return "LogisticRegression", pipeline


class ThresholdOptimizer:
    """阈值优化器，用于在指定假阳性率约束下优化灵敏度"""

    @staticmethod
    def find_optimal_threshold(
            y_true: np.ndarray,
            y_prob: np.ndarray,
            target_fpr: float
    ) -> Tuple[float, Dict[str, float]]:
        """
        在目标假阳性率约束下找到最优阈值

        Args:
            y_true: 真实标签
            y_prob: 预测概率
            target_fpr: 目标假阳性率上限

        Returns:
            (最优阈值, 性能指标字典)
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)

        # 找到满足 FPR 约束的阈值
        valid_indices = fpr <= target_fpr

        if not np.any(valid_indices):
            # 如果没有满足条件的阈值，选择 FPR 最小的
            best_idx = np.argmin(fpr)
            logger.warning(f"无法找到 FPR <= {target_fpr} 的阈值，选择 FPR 最小值 {fpr[best_idx]:.4f}")
        else:
            # 在满足 FPR 约束的阈值中，选择 TPR 最大的
            valid_tpr = tpr[valid_indices]
            best_valid_idx = np.argmax(valid_tpr)
            best_idx = np.arange(len(thresholds))[valid_indices][best_valid_idx]

        optimal_threshold = thresholds[best_idx]

        metrics = {
            "threshold": float(optimal_threshold),
            "fpr": float(fpr[best_idx]),
            "tpr": float(tpr[best_idx]),
            "specificity": float(1 - fpr[best_idx]),
            "sensitivity": float(tpr[best_idx]),
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "pr_auc": float(average_precision_score(y_true, y_prob))
        }

        return float(optimal_threshold), metrics


class BootstrapAnalyzer:
    """Bootstrap 分析器，用于计算置信区间"""

    @staticmethod
    def calculate_confidence_intervals(
            y_true: np.ndarray,
            y_prob: np.ndarray,
            threshold: float,
            n_bootstrap: int = 1000,
            confidence_level: float = 0.95,
            random_state: int = 42
    ) -> Dict[str, Tuple[float, float]]:
        """
        使用 Bootstrap 方法计算性能指标的置信区间

        Args:
            y_true: 真实标签
            y_prob: 预测概率
            threshold: 分类阈值
            n_bootstrap: Bootstrap 采样次数
            confidence_level: 置信水平
            random_state: 随机种子

        Returns:
            各指标的置信区间字典
        """
        rng = np.random.default_rng(random_state)
        n_samples = len(y_true)

        bootstrap_metrics = {
            'sensitivity': [],
            'specificity': [],
            'roc_auc': [],
            'pr_auc': []
        }

        for _ in range(n_bootstrap):
            # Bootstrap 采样
            indices = rng.integers(0, n_samples, size=n_samples)
            y_boot = y_true[indices]
            p_boot = y_prob[indices]

            # 跳过单类样本
            if len(np.unique(y_boot)) < 2:
                continue

            try:
                # 计算指标
                y_pred_boot = (p_boot >= threshold).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_boot, y_pred_boot, labels=[0, 1]).ravel()

                if tp + fn > 0:
                    sensitivity = tp / (tp + fn)
                    bootstrap_metrics['sensitivity'].append(sensitivity)

                if tn + fp > 0:
                    specificity = tn / (tn + fp)
                    bootstrap_metrics['specificity'].append(specificity)

                bootstrap_metrics['roc_auc'].append(roc_auc_score(y_boot, p_boot))
                bootstrap_metrics['pr_auc'].append(average_precision_score(y_boot, p_boot))

            except Exception:
                continue

        # 计算置信区间
        alpha = 1 - confidence_level
        confidence_intervals = {}

        for metric, values in bootstrap_metrics.items():
            if values:
                values_array = np.array(values)
                lower = np.percentile(values_array, 100 * alpha / 2)
                upper = np.percentile(values_array, 100 * (1 - alpha / 2))
                confidence_intervals[f"{metric}_ci"] = (float(lower), float(upper))

        return confidence_intervals


class Visualizer:
    """可视化器"""

    @staticmethod
    def plot_roc_pr_curves(
            y_true: np.ndarray,
            y_prob: np.ndarray,
            output_dir: str,
            prefix: str = "female"
    ):
        """绘制 ROC 和 PR 曲线"""

        # ROC 曲线
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.7)
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('ROC Curve - 女胎异常检测')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / f"{prefix}_roc_curve.png", dpi=300, bbox_inches='tight')
        plt.close()

        # PR 曲线
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, label=f'PR Curve (AP = {pr_auc:.3f})')
        baseline = np.sum(y_true) / len(y_true)
        plt.axhline(y=baseline, color='gray', linestyle='--', alpha=0.7, label=f'Baseline (AP = {baseline:.3f})')
        plt.xlabel('Recall (Sensitivity)')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - 女胎异常检测')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / f"{prefix}_pr_curve.png", dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_feature_importance(
            importance_series: pd.Series,
            output_dir: str,
            title: str = "Feature Importance"
    ):
        """绘制特征重要性图"""
        if len(importance_series) == 0:
            return

        # 取前15个最重要的特征
        top_features = importance_series.head(15)

        plt.figure(figsize=(10, max(6, len(top_features) * 0.5)))
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))

        bars = plt.barh(range(len(top_features)), top_features.values, color=colors)
        plt.yticks(range(len(top_features)), top_features.index)
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.title(title)
        plt.grid(True, alpha=0.3, axis='x')

        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, top_features.values)):
            plt.text(bar.get_width() + max(top_features.values) * 0.01,
                     bar.get_y() + bar.get_height() / 2,
                     f'{value:.3f}',
                     ha='left', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(Path(output_dir) / "feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()


class ModelTrainer:
    """模型训练器 - 主要的训练和评估逻辑"""

    def __init__(self, config: Config):
        self.config = config
        self.feature_engineer = FeatureEngineer(config)

    def train_and_evaluate(self, df: pd.DataFrame, output_dir: str) -> Dict[str, Any]:
        """
        完整的训练和评估流程

        Args:
            df: 输入数据
            output_dir: 输出目录

        Returns:
            评估结果字典
        """
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("开始训练女胎异常检测模型...")

        # 1. 自动检测列名
        config = self.feature_engineer.autodetect_columns(df)

        # 2. 筛选女胎样本
        if config.fetal_sex and config.fetal_sex in df.columns:
            df = DataProcessor.filter_female_samples(df, config.fetal_sex)
            if len(df) == 0:
                raise ValueError("筛选后没有女胎样本")
        else:
            logger.warning("未找到胎儿性别列，将使用所有样本")

        # 3. 构建特征和标签
        X, y, feature_cols = self.feature_engineer.build_features_and_labels(df, config)

        # 4. 检查样本平衡性
        class_counts = y.value_counts().sort_index()
        logger.info(f"样本分布 - 正常: {class_counts.get(0, 0)}, 异常: {class_counts.get(1, 0)}")

        if len(class_counts) < 2:
            raise ValueError("数据中只有一个类别，无法进行分类")

        # 5. 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.test_size,
            stratify=y,
            random_state=config.random_state
        )

        logger.info(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")

        # 6. 创建模型
        model_name, pipeline = ModelBuilder.create_model(config, X.shape[1])
        logger.info(f"使用模型: {model_name}")

        # 7. 交叉验证
        cv_results = self._cross_validate(pipeline, X_train, y_train, config)

        # 8. 训练最终模型
        pipeline.fit(X_train, y_train)

        # 9. 测试集预测
        test_probabilities = pipeline.predict_proba(X_test)[:, 1]

        # 10. 阈值优化
        optimal_threshold, threshold_metrics = ThresholdOptimizer.find_optimal_threshold(
            y_test.values, test_probabilities, config.target_fpr
        )

        # 11. 混淆矩阵和分类报告
        y_pred = (test_probabilities >= optimal_threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        classification_rep = classification_report(
            y_test, y_pred,
            target_names=['正常', '异常'],
            digits=4,
            output_dict=False
        )

        # 12. Bootstrap 置信区间
        confidence_intervals = BootstrapAnalyzer.calculate_confidence_intervals(
            y_test.values, test_probabilities, optimal_threshold,
            n_bootstrap=1000, random_state=config.random_state
        )

        # 13. 特征重要性
        feature_importance = self._extract_feature_importance(pipeline, feature_cols)

        # 14. 生成可视化
        Visualizer.plot_roc_pr_curves(
            y_test.values, test_probabilities, output_dir, prefix="female"
        )

        if feature_importance is not None:
            Visualizer.plot_feature_importance(
                feature_importance, output_dir, "特征重要性 - 女胎异常检测模型"
            )

        # 15. 保存结果
        results = self._save_results(
            output_path, model_name, feature_cols, cv_results,
            threshold_metrics, confidence_intervals, cm,
            classification_rep, feature_importance, config
        )

        logger.info("模型训练和评估完成!")
        return results

    def _cross_validate(self, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, config: Config) -> Dict[str, float]:
        """执行交叉验证"""
        logger.info(f"开始 {config.cv_folds} 折交叉验证...")

        skf = StratifiedKFold(
            n_splits=config.cv_folds,
            shuffle=True,
            random_state=config.random_state
        )

        cv_scores = {'roc_auc': [], 'pr_auc': []}

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

            # 训练
            pipeline.fit(X_fold_train, y_fold_train)

            # 预测
            val_probs = pipeline.predict_proba(X_fold_val)[:, 1]

            # 评估
            try:
                roc_auc = roc_auc_score(y_fold_val, val_probs)
                pr_auc = average_precision_score(y_fold_val, val_probs)

                cv_scores['roc_auc'].append(roc_auc)
                cv_scores['pr_auc'].append(pr_auc)

                logger.info(f"Fold {fold}: ROC-AUC = {roc_auc:.4f}, PR-AUC = {pr_auc:.4f}")
            except Exception as e:
                logger.warning(f"Fold {fold} 评估失败: {e}")

        # 计算均值和标准差
        cv_results = {}
        for metric, scores in cv_scores.items():
            if scores:
                cv_results[f"cv_{metric}_mean"] = float(np.mean(scores))
                cv_results[f"cv_{metric}_std"] = float(np.std(scores))
            else:
                cv_results[f"cv_{metric}_mean"] = 0.0
                cv_results[f"cv_{metric}_std"] = 0.0

        logger.info(f"交叉验证结果: ROC-AUC = {cv_results['cv_roc_auc_mean']:.4f} ± {cv_results['cv_roc_auc_std']:.4f}")
        return cv_results

    def _extract_feature_importance(self, pipeline: Pipeline, feature_cols: List[str]) -> Optional[pd.Series]:
        """提取特征重要性"""
        try:
            classifier = pipeline.named_steps.get("classifier")
            if classifier is None:
                return None

            importance_values = None

            # XGBoost 和 LightGBM
            if hasattr(classifier, "feature_importances_"):
                importance_values = classifier.feature_importances_

            # 逻辑回归 - 使用系数的绝对值
            elif hasattr(classifier, "coef_"):
                importance_values = np.abs(classifier.coef_).ravel()

            if importance_values is not None:
                importance_series = pd.Series(
                    importance_values,
                    index=feature_cols
                ).sort_values(ascending=False)

                logger.info(f"特征重要性前5: {importance_series.head().to_dict()}")
                return importance_series

        except Exception as e:
            logger.warning(f"提取特征重要性失败: {e}")

        return None

    def _save_results(
            self,
            output_path: Path,
            model_name: str,
            feature_cols: List[str],
            cv_results: Dict[str, float],
            threshold_metrics: Dict[str, float],
            confidence_intervals: Dict[str, Tuple[float, float]],
            confusion_matrix_array: np.ndarray,
            classification_report_str: str,
            feature_importance: Optional[pd.Series],
            config: Config
    ) -> Dict[str, Any]:
        """保存所有结果到文件"""

        # 1. 保存主要结果摘要
        summary = {
            "model_info": {
                "model_type": model_name,
                "features_used": feature_cols,
                "n_features": len(feature_cols),
                "target_fpr": config.target_fpr,
                "abnormal_detection_method": config.abnormal_detection_method,
                "zscore_threshold": config.zscore_threshold
            },
            "cross_validation": cv_results,
            "test_performance": threshold_metrics,
            "confidence_intervals": confidence_intervals,
            "confusion_matrix": confusion_matrix_array.tolist()
        }

        with open(output_path / "model_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # 2. 保存混淆矩阵 (带标签)
        cm_df = pd.DataFrame(
            confusion_matrix_array,
            index=["实际正常", "实际异常"],
            columns=["预测正常", "预测异常"]
        )
        cm_df.to_csv(output_path / "confusion_matrix.csv", encoding="utf-8")

        # 3. 保存分类报告
        with open(output_path / "classification_report.txt", "w", encoding="utf-8") as f:
            f.write("女胎异常检测模型 - 分类报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"异常检测方法: {config.abnormal_detection_method}\n")
            f.write(f"Z-score阈值: {config.zscore_threshold}\n\n")
            f.write(classification_report_str)

        # 4. 保存特征重要性
        if feature_importance is not None:
            feature_importance.to_csv(output_path / "feature_importance.csv",
                                      header=["importance"], encoding="utf-8")

        # 5. 保存配置信息
        config_dict = asdict(config)
        with open(output_path / "model_config.json", "w", encoding="utf-8") as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)

        # 6. 创建性能报告
        self._create_performance_report(output_path, summary, cm_df, config)

        # 返回主要结果用于控制台显示
        return {
            "model": model_name,
            "features_used": feature_cols,
            "test_roc_auc": threshold_metrics["roc_auc"],
            "test_pr_auc": threshold_metrics["pr_auc"],
            "optimal_threshold": threshold_metrics["threshold"],
            "sensitivity": threshold_metrics["sensitivity"],
            "specificity": threshold_metrics["specificity"],
            "fpr": threshold_metrics["fpr"],
            "confusion_matrix": confusion_matrix_array.tolist(),
            "confidence_intervals": confidence_intervals,
            "abnormal_detection_method": config.abnormal_detection_method,
            "zscore_threshold": config.zscore_threshold
        }

    def _create_performance_report(
            self,
            output_path: Path,
            summary: Dict,
            cm_df: pd.DataFrame,
            config: Config
    ):
        """创建可读的性能报告"""

        report_lines = [
            "女胎染色体异常检测模型 - 性能报告",
            "=" * 60,
            "",
            f"模型类型: {summary['model_info']['model_type']}",
            f"使用特征数: {summary['model_info']['n_features']}",
            f"目标假阳性率: {config.target_fpr:.1%}",
            f"异常检测方法: {config.abnormal_detection_method}",
            f"Z-score阈值: {config.zscore_threshold}",
            "",
            "=== 交叉验证结果 ===",
            f"ROC-AUC: {summary['cross_validation']['cv_roc_auc_mean']:.4f} ± {summary['cross_validation']['cv_roc_auc_std']:.4f}",
            f"PR-AUC:  {summary['cross_validation']['cv_pr_auc_mean']:.4f} ± {summary['cross_validation']['cv_pr_auc_std']:.4f}",
            "",
            "=== 测试集性能 ===",
            f"ROC-AUC: {summary['test_performance']['roc_auc']:.4f}",
            f"PR-AUC:  {summary['test_performance']['pr_auc']:.4f}",
            f"最优阈值: {summary['test_performance']['threshold']:.4f}",
            "",
            "=== 在最优阈值下的性能 ===",
            f"灵敏度 (Sensitivity): {summary['test_performance']['sensitivity']:.4f}",
            f"特异度 (Specificity): {summary['test_performance']['specificity']:.4f}",
            f"假阳性率 (FPR): {summary['test_performance']['fpr']:.4f}",
            "",
            "=== 混淆矩阵 ===",
            str(cm_df),
            "",
            "=== 95% 置信区间 ==="
        ]

        # 添加置信区间信息
        for metric, (lower, upper) in summary['confidence_intervals'].items():
            metric_name = metric.replace('_ci', '').replace('_', ' ').title()
            report_lines.append(f"{metric_name}: [{lower:.4f}, {upper:.4f}]")

        report_lines.extend([
            "",
            "=== 使用的特征 ===",
            ", ".join(summary['model_info']['features_used']),
            "",
            "=== 模型解释 ===",
            "1. 该模型专门针对女胎样本训练",
            "2. 使用代价敏感学习，优先保证高灵敏度",
            f"3. 在 {config.target_fpr:.1%} 假阳性率约束下优化检测性能",
            "4. 通过交叉验证确保模型稳定性",
            "5. 提供Bootstrap置信区间评估不确定性",
            f"6. 基于{config.abnormal_detection_method}方法创建异常标签",
            f"7. Z-score阈值: {config.zscore_threshold}"
        ])

        # 写入文件
        with open(output_path / "performance_report.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))


class DataLoader:
    """数据加载器"""

    @staticmethod
    def load_data(file_paths: List[str]) -> pd.DataFrame:
        """
        尝试从多个可能的路径加载数据

        Args:
            file_paths: 候选文件路径列表

        Returns:
            加载的 DataFrame
        """
        for file_path in file_paths:
            if not file_path:
                continue

            path_obj = Path(file_path)
            if not path_obj.exists():
                continue

            try:
                logger.info(f"尝试加载文件: {file_path}")

                # 根据文件扩展名选择加载方法
                if path_obj.suffix.lower() in ['.csv']:
                    df = pd.read_csv(file_path, encoding='utf-8')
                elif path_obj.suffix.lower() in ['.xlsx', '.xls']:
                    df = pd.read_excel(file_path)
                else:
                    # 尝试作为 CSV 加载
                    df = pd.read_csv(file_path, encoding='utf-8')

                logger.info(f"成功加载数据: {df.shape[0]} 行, {df.shape[1]} 列")
                return df

            except Exception as e:
                logger.warning(f"加载文件 {file_path} 失败: {e}")
                continue

        raise FileNotFoundError(
            f"无法从以下路径加载数据文件:\n" + "\n".join(f"  - {p}" for p in file_paths if p)
        )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="女胎染色体异常检测模型 (代价敏感) - 修复版",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--data",
        type=str,
        default="",
        help="数据文件路径"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="q4_outputs_fixed",
        help="输出目录"
    )
    parser.add_argument(
        "--target_fpr",
        type=float,
        default=0.01,
        help="目标假阳性率上限 (0-1之间)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lr",
        choices=["lr", "xgb", "lgbm"],
        help="模型类型"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="测试集比例"
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=5,
        help="交叉验证折数"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="随机种子"
    )
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="手动指定标签列名"
    )
    parser.add_argument(
        "--use_all_samples",
        action="store_true",
        help="使用所有样本（不仅限女胎）"
    )
    parser.add_argument(
        "--explore_data",
        action="store_true",
        help="仅探索数据，不训练模型"
    )
    # 新增参数
    parser.add_argument(
        "--zscore_threshold",
        type=float,
        default=2.5,
        help="Z-score异常检测阈值"
    )
    parser.add_argument(
        "--abnormal_detection_method",
        type=str,
        default="zscore",
        choices=["zscore", "percentile", "risk_score"],
        help="异常检测方法"
    )
    parser.add_argument(
        "--percentile_threshold",
        type=float,
        default=95.0,
        help="百分位数异常检测阈值"
    )
    parser.add_argument(
        "--force_create_labels",
        action="store_true",
        help="强制基于Z-score创建异常标签"
    )

    args = parser.parse_args()

    try:
        # 数据文件路径
        candidate_paths = [
            args.data,
            "../Data_Cleaned/output/nipt_cleaned.csv",
            "../Data_Cleaned/output/female_tests.csv",
        ]

        # 加载数据
        df = DataLoader.load_data(candidate_paths)
        df = DataProcessor.normalize_columns(df)

        # 探索数据模式
        if args.explore_data:
            print("\n" + "=" * 70)
            print("数据探索模式")
            print("=" * 70)
            print(f"数据形状: {df.shape}")
            print(f"列名: {list(df.columns)}")

            # 查找可能的标签列
            label_candidates = []
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['label', 'health', 'result', 'abnormal', 'diagnosis']):
                    label_candidates.append(col)
                    print(f"\n可能的标签列 '{col}':")
                    print(df[col].value_counts())

            # 查找性别列
            sex_candidates = []
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['sex', 'gender', '性别']):
                    sex_candidates.append(col)
                    print(f"\n可能的性别列 '{col}':")
                    print(df[col].value_counts())

            # 分析Z-score分布
            print(f"\nZ-score分布分析:")
            z_cols = [col for col in df.columns if col.lower().startswith('z_')]
            for col in z_cols:
                values = df[col].dropna()
                print(f"\n{col}:")
                print(f"  均值: {values.mean():.3f}")
                print(f"  标准差: {values.std():.3f}")
                print(f"  |Z| > 2的样本: {(abs(values) > 2).sum()}")
                print(f"  |Z| > 3的样本: {(abs(values) > 3).sum()}")

            print(f"\n建议的标签列: {label_candidates}")
            print(f"建议的性别列: {sex_candidates}")
            print(f"\n建议使用 --force_create_labels 参数基于Z-score创建异常标签")
            return

        # 创建配置
        config = Config(
            target_fpr=args.target_fpr,
            model=args.model,
            test_size=args.test_size,
            cv_folds=args.cv_folds,
            random_state=args.random_state,
            zscore_threshold=args.zscore_threshold,
            abnormal_detection_method=args.abnormal_detection_method,
            percentile_threshold=args.percentile_threshold,
            force_create_labels=args.force_create_labels
        )

        # 手动指定标签列
        if args.label:
            config.label = args.label

        # 创建训练器
        trainer = ModelTrainer(config)

        # 如果使用所有样本，跳过性别筛选
        if args.use_all_samples:
            config.fetal_sex = None
            logger.info("使用所有样本进行训练（不限制性别）")

        # 执行训练
        results = trainer.train_and_evaluate(df, args.outdir)

        # 打印结果摘要
        print("\n" + "=" * 70)
        print("女胎染色体异常检测模型训练完成")
        print("=" * 70)
        print(f"模型类型: {results['model']}")
        print(f"异常检测方法: {results['abnormal_detection_method']}")
        print(f"Z-score阈值: {results['zscore_threshold']}")
        print(f"使用特征: {len(results['features_used'])} 个")
        print(f"测试集 ROC-AUC: {results['test_roc_auc']:.4f}")
        print(f"测试集 PR-AUC: {results['test_pr_auc']:.4f}")
        print(f"最优阈值: {results['optimal_threshold']:.4f}")
        print(f"灵敏度: {results['sensitivity']:.4f}")
        print(f"特异度: {results['specificity']:.4f}")
        print(f"假阳性率: {results['fpr']:.4f}")

        print(f"\n输出文件保存在: {Path(args.outdir).absolute()}")
        print("   包含以下文件:")
        print("   ├── model_summary.json      (模型摘要)")
        print("   ├── performance_report.txt  (性能报告)")
        print("   ├── confusion_matrix.csv    (混淆矩阵)")
        print("   ├── classification_report.txt (分类报告)")
        print("   ├── feature_importance.csv  (特征重要性)")
        print("   ├── female_roc_curve.png    (ROC曲线)")
        print("   ├── female_pr_curve.png     (PR曲线)")
        print("   └── feature_importance.png  (特征重要性图)")

        # 显示置信区间
        if results['confidence_intervals']:
            print(f"\n95% 置信区间:")
            for metric, (lower, upper) in results['confidence_intervals'].items():
                metric_display = metric.replace('_ci', '').replace('_', ' ').title()
                print(f"   {metric_display}: [{lower:.4f}, {upper:.4f}]")

        print(f"\n训练成功完成!")

    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        raise


if __name__ == "__main__":
    main()