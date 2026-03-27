import os, re, math, warnings, json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont, FontManager
from scipy import stats
from scipy.optimize import minimize_scalar

import torch
from torch import nn
from torch.optim import LBFGS, Adam

# ======================= 基本参数（你可以按需修改） =======================

# 你的数据路径（会按顺序尝试读取，找到第一个存在的文件）
DATA_CANDIDATES = ["../Data_Cleaned/output/nipt_cleaned.csv"]

MALE_TESTS_OPTIONAL = ["../Data_Cleaned/output/male_tests.csv"]
EARLIEST_FILE_OPTIONAL = ["../Data_Cleaned/output/earliest_male_threshold.csv"]

# 列名配置（按你的英文列名）
MOTHER_ID = "mother_id"
WEEK_COL = "gestation_weeks"
BMI_COL = "bmi"
Y_FRAC_COL = "y_frac"
GC_COL = "gc_total"  # 可缺省
MAPPING_RATE = "reads_align"  # 可缺省（或改成 reads_align_ratio 等）
FILTERED_RATIO = "reads_filtered"  # 可缺省

# 达标与质控阈值 - 新增：多阈值敏感性分析
THRESH_Y_FRAC = 0.04
THRESH_SENSITIVITY = [0.035, 0.04, 0.045]  # 敏感性分析的多个阈值
QC_THRESH = {"gc_min": 0.40, "gc_max": 0.60, "map_min": 0.85, "filt_max": 0.60}

# 质控波动参数（新增）
QC_RELIABILITY = 0.95  # 质控通过的基础概率
QC_NOISE_LEVEL = 0.1  # 质控波动的标准差

# Bootstrap参数（新增）
N_BOOTSTRAP = 1000  # Bootstrap重采样次数
BOOTSTRAP_SEED = 42  # Bootstrap随机种子

# 逻辑回归与优化
MAX_ITERS_SEG_FIT = 400
L2_REG = 1e-4

# DP 分箱（越大越细；注意样本量约束）
K_MAX = 6
MIN_MOTHERS_PER_BIN = 20
N_BMI_KNOTS_MAX = 120
NLL_BIG = 1e9

# 外层迭代（模型⇄分箱）
N_OUTER_ITERS = 5

# 风险权重与孕周网格 - 修改：将使用每组独立的真实数据范围
EARLY_END, MID_END = 12.0, 27.0
RISK_WEIGHT = {"early": 0.1, "mid": 1.0, "late": 3.0}

# 医学合理范围（用作安全边界）
MEDICAL_MIN_WEEK = 6.0  # 医学上最早可能的检测孕周
MEDICAL_MAX_WEEK = 35.0  # 医学上最晚合理的检测孕周

# 输出目录
OUT_DIR = "./nipt_outputs"
PLOT_DIR = os.path.join(OUT_DIR, "plots")
OVERVIEW_DIR = os.path.join(OUT_DIR, "overview")
BOOTSTRAP_DIR = os.path.join(OUT_DIR, "bootstrap")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(OVERVIEW_DIR, exist_ok=True)
os.makedirs(BOOTSTRAP_DIR, exist_ok=True)

# 每张图都要"单独"设置的中文字体（名字或绝对路径都支持）
# Windows: "SimHei" 或 "Microsoft YaHei"
# macOS: "PingFang SC"
# Linux: "Noto Sans CJK SC"
CHINESE_FONT_NAME = "SimHei"
CHINESE_FONT_PATH = None  # 若你有指定字体文件 *.ttf，可在此填入绝对路径

# 随机种子
np.random.seed(2025)
torch.manual_seed(2025)

# 中文标签字典 - 新增统一的中文标签管理
CHINESE_LABELS = {
    # 基础标签
    "week": "孕周",
    "reach_status": "是否达标（抖动可视）",
    "not_reached": "未达标",
    "reached": "达标",
    "bmi": "BMI",
    "y_frac": "Y染色体浓度 y_frac",
    "count": "人数",
    "probability_density": "概率密度",
    "detection_week": "检测孕周",

    # 模型相关
    "logistic_prob": "逻辑回归达标概率",
    "target_90pct": "90%通过率目标",
    "optimal_week": "最佳检测周",
    "bootstrap_dist": "Bootstrap分布",
    "bootstrap_mean": "Bootstrap均值",
    "confidence_interval_95": "95%置信区间",
    "confidence_interval_90": "90%置信区间",

    # 阈值相关
    "main_threshold": "主阈值",
    "sensitivity_threshold": "敏感性阈值",
    "target_85pct": "85%目标",
    "target_95pct": "95%目标",

    # 范围相关
    "analysis_range": "该组分析范围",
    "data_range": "数据范围",
    "analysis_extension": "分析扩展",

    # 图表标题
    "prob_curve_title": "达标概率曲线（通过率阈值法）",
    "data_coverage_title": "数据分布与检测策略",
    "bootstrap_title": "Bootstrap重采样分布",
    "bmi_dist_title": "BMI分布与DP+BIC分箱边界",
    "optimal_weeks_title": "各BMI组最佳检测周（含独立分析范围）",
    "analysis_ranges_title": "各BMI组分析范围对比",

    # 说明文字
    "boundary_warning_lower": "⚠️下限限制",
    "boundary_warning_upper": "⚠️上限限制",
    "boundary_limited": "⚠️边界限制",
    "narrow_range": "(窄)",
    "wide_range": "(宽)",
    "quality_reminder": "质量提醒",
    "actual_achieved": "实际达到",

    # 图例说明
    "analysis_range_legend": "分析范围",
    "optimal_detection_week": "最优检测周",
    "main_threshold_line": "主阈值线",
    "optimal_week_line": "最优检测周线"
}


# ======================= 设备选择（支持 4060） =======================

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEVICE = get_device()
print(f"[INFO] Using device: {DEVICE}")


# ======================= 工具函数：中文字体（每图单独设置） =======================

def get_chinese_font() -> FontProperties:
    """
    为每一张图创建一个 FontProperties，优先用 CHINESE_FONT_PATH（若存在），否则用 CHINESE_FONT_NAME。
    若给定名字找不到，会尽量用系统已安装字体回退。
    """
    if CHINESE_FONT_PATH and os.path.exists(CHINESE_FONT_PATH):
        return FontProperties(fname=CHINESE_FONT_PATH)
    # 直接返回名字；matplotlib 会用 font cache 解析
    return FontProperties(family=CHINESE_FONT_NAME)


# ======================= 每组独立的数据范围分析 =======================

def determine_week_range_per_group(tests_in_bin: pd.DataFrame, bmi_min: float, bmi_max: float) -> Tuple[
    float, float, Dict]:
    """
    为每个BMI组单独确定分析范围，基于该组的实际数据分布

    Returns:
        group_min: 该组分析使用的最小孕周
        group_max: 该组分析使用的最大孕周
        group_report: 该组数据质量报告
    """
    actual_weeks = tests_in_bin[WEEK_COL].dropna()

    if len(actual_weeks) == 0:
        # 如果该组没有数据，使用医学默认范围
        return MEDICAL_MIN_WEEK, MEDICAL_MAX_WEEK, {
            "bmi_range": f"[{bmi_min:.1f}, {bmi_max:.1f}]",
            "sample_count": 0,
            "error": "no_data"
        }

    # 该BMI组的数据统计
    data_min = actual_weeks.min()
    data_max = actual_weeks.max()
    data_mean = actual_weeks.mean()
    data_std = actual_weeks.std()

    # 为该组设置分析范围：在数据基础上适当扩展，但不超过医学范围
    range_buffer = max(0.3, data_std * 0.15) if data_std > 0 else 0.5
    group_min = max(MEDICAL_MIN_WEEK, data_min - range_buffer)
    group_max = min(MEDICAL_MAX_WEEK, data_max + range_buffer)

    # 如果该组数据范围太窄（<2周），适当扩展
    if group_max - group_min < 2.0:
        center = (group_min + group_max) / 2
        group_min = max(MEDICAL_MIN_WEEK, center - 1.0)
        group_max = min(MEDICAL_MAX_WEEK, center + 1.0)

    # 计算该组的覆盖分析
    reach_rate = tests_in_bin['reach'].mean() if 'reach' in tests_in_bin.columns else 0
    early_count = (actual_weeks < 12).sum()
    late_count = (actual_weeks >= 20).sum()

    group_report = {
        "bmi_range": f"[{bmi_min:.1f}, {bmi_max:.1f}]",
        "sample_count": len(actual_weeks),
        "data_range": [float(data_min), float(data_max)],
        "analysis_range": [float(group_min), float(group_max)],
        "statistics": {
            "mean": float(data_mean),
            "std": float(data_std),
            "reach_rate": float(reach_rate)
        },
        "coverage": {
            "early_samples": int(early_count),
            "late_samples": int(late_count),
            "total_weeks_span": float(data_max - data_min)
        },
        "analysis_notes": []
    }

    # 分析质量评估
    if len(actual_weeks) < 20:
        group_report["analysis_notes"].append("样本量较少，结果可靠性有限")
    if data_max - data_min < 3:
        group_report["analysis_notes"].append("数据跨度较窄，可能影响拟合精度")
    if reach_rate < 0.05:
        group_report["analysis_notes"].append("达标率很低，需验证检测可行性")
    if early_count == 0:
        group_report["analysis_notes"].append("无早期数据，无法评估早期检测")

    return group_min, group_max, group_report


def save_group_analysis_reports(group_reports: List[Dict], out_dir: str):
    """保存每组的数据分析报告"""

    # 保存JSON格式
    with open(os.path.join(out_dir, "group_analysis_reports.json"), 'w', encoding='utf-8') as f:
        json.dump(group_reports, f, ensure_ascii=False, indent=2)

    # 保存文本格式
    with open(os.path.join(out_dir, "group_analysis_report.txt"), 'w', encoding='utf-8') as f:
        f.write("NIPT 每组数据分析报告\n")
        f.write("=" * 60 + "\n\n")

        for i, report in enumerate(group_reports):
            if "error" in report:
                f.write(f"组 {i + 1}: {report['bmi_range']} - 无数据\n")
                continue

            f.write(f"组 {i + 1}: {report['bmi_range']}\n")
            f.write("-" * 40 + "\n")
            f.write(f"样本数量: {report['sample_count']}\n")
            f.write(f"数据范围: {report['data_range'][0]:.1f} - {report['data_range'][1]:.1f} 周\n")
            f.write(f"分析范围: {report['analysis_range'][0]:.1f} - {report['analysis_range'][1]:.1f} 周\n")
            f.write(f"平均孕周: {report['statistics']['mean']:.1f} ± {report['statistics']['std']:.1f}\n")
            f.write(f"达标率: {report['statistics']['reach_rate']:.3f}\n")
            f.write(f"早期样本(<12周): {report['coverage']['early_samples']}\n")
            f.write(f"晚期样本(≥20周): {report['coverage']['late_samples']}\n")

            if report['analysis_notes']:
                f.write("注意事项:\n")
                for note in report['analysis_notes']:
                    f.write(f"  • {note}\n")
            f.write("\n")

    print(f"[OK] 每组分析报告已保存至 {out_dir}")


# ======================= 数据加载与清洗 =======================

def parse_week(x):
    """
    支持 12.5 / 12,5 / '12+3' → 12 + 3/7
    """
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    m = re.match(r"^(\d+)\s*\+\s*(\d+)$", s)
    if m:
        return int(m.group(1)) + int(m.group(2)) / 7.0
    s = s.replace(",", ".")
    try:
        return float(s)
    except:
        return np.nan


def qc_pass_row(row, qc_noise=False):
    """
    质控判定，新增可选的质控波动建模
    """
    ok = True
    if GC_COL in row and not pd.isna(row[GC_COL]):
        ok = ok and (QC_THRESH["gc_min"] <= float(row[GC_COL]) <= QC_THRESH["gc_max"])
    if MAPPING_RATE in row and not pd.isna(row[MAPPING_RATE]):
        ok = ok and (float(row[MAPPING_RATE]) >= QC_THRESH["map_min"])
    if FILTERED_RATIO in row and not pd.isna(row[FILTERED_RATIO]):
        ok = ok and (float(row[FILTERED_RATIO]) <= QC_THRESH["filt_max"])

    # 新增：质控波动建模
    if qc_noise and ok:
        # 即使满足质控条件，也有一定概率因为系统波动而"失败"
        noise = np.random.normal(0, QC_NOISE_LEVEL)
        effective_reliability = QC_RELIABILITY + noise
        ok = ok and (np.random.random() < np.clip(effective_reliability, 0, 1))

    return ok


def try_read_first(paths: List[str]) -> Optional[pd.DataFrame]:
    for p in paths:
        if os.path.exists(p):
            print(f"[INFO] Loaded: {p}")
            return pd.read_csv(p)
    return None


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = try_read_first(DATA_CANDIDATES)
    if base is None:
        raise FileNotFoundError("未找到 nipt_cleaned.csv，请检查 DATA_CANDIDATES。")

    male_extra = try_read_first(MALE_TESTS_OPTIONAL)

    if male_extra is not None:
        base = pd.concat([base, male_extra], axis=0, ignore_index=True).drop_duplicates()

    # 统一所需列名（若你文件里命名不同，在顶部常量处改变量即可）
    # 解析孕周
    base[WEEK_COL] = base[WEEK_COL].apply(parse_week)

    # 仅男胎：Y_frac 非空
    male = base[base[Y_FRAC_COL].notna()].copy()

    # 质控和达标
    male["qc_pass"] = male.apply(lambda row: qc_pass_row(row, qc_noise=False), axis=1)
    male["reach"] = ((male[Y_FRAC_COL] >= THRESH_Y_FRAC) & (male["qc_pass"])).astype(int)

    # 清洗必要字段
    male = male.dropna(subset=[MOTHER_ID, BMI_COL, WEEK_COL])
    male = male.sort_values([MOTHER_ID, WEEK_COL]).reset_index(drop=True)

    # earliest 达标孕周（可用于对照/检验）
    def earliest_reach(group):
        hit = group[group["reach"] == 1]
        if hit.empty: return np.nan
        return hit[WEEK_COL].iloc[0]

    earliest = male.groupby(MOTHER_ID).apply(earliest_reach).reset_index()
    earliest.columns = [MOTHER_ID, "earliest_week"]
    bmi_any = male.groupby(MOTHER_ID)[BMI_COL].first().reset_index()
    earliest = earliest.merge(bmi_any, on=MOTHER_ID, how="left")
    earliest = earliest.dropna(subset=[BMI_COL])

    print(
        f"[INFO] 男胎检测条目: {male.shape[0]}, 孕妇人数: {male[MOTHER_ID].nunique()}, 有 earliest: {(~earliest['earliest_week'].isna()).sum()}")
    return male, earliest


# ======================= 检测误差建模功能 =======================

@dataclass
class ErrorModelingResult:
    """存储误差建模结果的数据类"""
    bootstrap_weeks: List[float]
    threshold_sensitivity: Dict[float, float]
    qc_noise_impact: float
    confidence_interval_95: Tuple[float, float]
    confidence_interval_90: Tuple[float, float]
    uncertainty_range: float


def bootstrap_first_detection(sub_df: pd.DataFrame, n_bootstrap: int = N_BOOTSTRAP) -> List[float]:
    """
    Bootstrap重采样，模拟测量噪声对首次达标孕周的影响
    """
    np.random.seed(BOOTSTRAP_SEED)
    bootstrap_weeks = []

    # 获取每位孕妇的测试数据
    mother_groups = sub_df.groupby(MOTHER_ID)

    for boot_iter in range(n_bootstrap):
        boot_first_weeks = []

        for mother_id, group in mother_groups:
            if len(group) < 2:  # 至少需要2次测试才能进行重采样
                continue

            # 对该孕妇的Y浓度加噪声（模拟测量误差）
            y_fracs = group[Y_FRAC_COL].values
            weeks = group[WEEK_COL].values

            # 添加高斯噪声（标准差为原值的5%）
            noise_std = np.std(y_fracs) * 0.05
            noisy_y_fracs = y_fracs + np.random.normal(0, noise_std, len(y_fracs))

            # 重新计算质控和达标状态
            qc_pass_noisy = []
            for _, row in group.iterrows():
                qc_pass_noisy.append(qc_pass_row(row, qc_noise=True))

            reach_noisy = ((noisy_y_fracs >= THRESH_Y_FRAC) & np.array(qc_pass_noisy)).astype(int)

            # 找到首次达标孕周
            first_reach_idx = np.where(reach_noisy == 1)[0]
            if len(first_reach_idx) > 0:
                boot_first_weeks.append(weeks[first_reach_idx[0]])

        if boot_first_weeks:
            bootstrap_weeks.append(np.mean(boot_first_weeks))

    return bootstrap_weeks


def threshold_sensitivity_analysis(sub_df: pd.DataFrame, weeks_grid: np.ndarray,
                                   thresholds: List[float] = THRESH_SENSITIVITY) -> Dict[float, float]:
    """
    阈值敏感性分析：测试不同 y_frac 阈值下的最优检测周
    """
    sensitivity_results = {}

    for thresh in thresholds:
        # 在该阈值下重算达标标签
        reach_thresh = ((sub_df[Y_FRAC_COL] >= thresh) & (sub_df["qc_pass"])).astype(int)
        if reach_thresh.sum() == 0:
            sensitivity_results[thresh] = np.nan
            continue

        weeks = sub_df[WEEK_COL].values
        labels = reach_thresh.values.astype(np.float32)
        mask = ~np.isnan(weeks)
        weeks_clean = weeks[mask]
        labels_clean = labels[mask]

        if len(weeks_clean) < 10:
            sensitivity_results[thresh] = np.nan
            continue

        try:
            nll, k_params, model = fit_logreg_weeks_torch(weeks_clean, labels_clean)
            best_week, _ = choose_week_by_target_prob(model, weeks_grid=weeks_grid, target=0.90)
            sensitivity_results[thresh] = best_week
        except Exception:
            sensitivity_results[thresh] = np.nan

    return sensitivity_results


def estimate_qc_noise_impact(sub_df: pd.DataFrame, n_simulations: int = 500) -> float:
    """
    估计质控波动对检测结果的影响
    """
    original_reach = sub_df["reach"].sum()
    noisy_reaches = []

    for _ in range(n_simulations):
        # 模拟质控波动
        qc_pass_noisy = []
        for _, row in sub_df.iterrows():
            qc_pass_noisy.append(qc_pass_row(row, qc_noise=True))

        reach_noisy = ((sub_df[Y_FRAC_COL] >= THRESH_Y_FRAC) & np.array(qc_pass_noisy)).astype(int)
        noisy_reaches.append(reach_noisy.sum())

    # 计算相对变化
    if original_reach > 0:
        relative_changes = [(nr - original_reach) / original_reach for nr in noisy_reaches]
        return np.std(relative_changes)
    else:
        return 0.0


def comprehensive_error_modeling(sub_df: pd.DataFrame, weeks_grid: np.ndarray) -> ErrorModelingResult:
    """
    综合误差建模分析
    """
    # Bootstrap分析
    bootstrap_weeks = bootstrap_first_detection(sub_df)

    # 阈值敏感性
    threshold_sens = threshold_sensitivity_analysis(sub_df, weeks_grid)

    # 质控噪声影响
    qc_impact = estimate_qc_noise_impact(sub_df)

    # 计算置信区间
    if bootstrap_weeks:
        ci_95 = np.percentile(bootstrap_weeks, [2.5, 97.5])
        ci_90 = np.percentile(bootstrap_weeks, [5, 95])
        uncertainty_range = ci_95[1] - ci_95[0]
    else:
        ci_95 = ci_90 = (np.nan, np.nan)
        uncertainty_range = np.nan

    return ErrorModelingResult(
        bootstrap_weeks=bootstrap_weeks,
        threshold_sensitivity=threshold_sens,
        qc_noise_impact=qc_impact,
        confidence_interval_95=ci_95,
        confidence_interval_90=ci_90,
        uncertainty_range=uncertainty_range
    )


# ======================= PyTorch 逻辑回归（单特征：孕周） =======================

class LogisticWeek(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.sigmoid(self.w * x + self.b)


def fit_logreg_weeks_torch(weeks, labels, weights=None, l2=L2_REG, max_iters=MAX_ITERS_SEG_FIT):
    x = torch.tensor(weeks, dtype=torch.float32, device=DEVICE).view(-1, 1)
    y = torch.tensor(labels, dtype=torch.float32, device=DEVICE).view(-1, 1)
    n = x.shape[0]

    if weights is None:
        wts = torch.ones_like(y)
    else:
        wts = torch.tensor(weights, dtype=torch.float32, device=DEVICE).view(-1, 1)

    # 退化：全 0 或全 1
    y_unique = torch.unique(y)
    if y_unique.numel() == 1:
        p = torch.clamp(torch.mean(y), 1e-6, 1 - 1e-6)
        nll = - (wts * (y * torch.log(p) + (1 - y) * torch.log(1 - p))).sum()
        model = LogisticWeek().to(DEVICE)
        with torch.no_grad():
            model.w[:] = 0.0
            model.b[:] = torch.log(p / (1 - p))
        k_params = 1
        return float(nll.item()), k_params, model

    model = LogisticWeek().to(DEVICE)
    try:
        optimizer = LBFGS(model.parameters(), lr=1.0, max_iter=max_iters, tolerance_grad=1e-7, tolerance_change=1e-9,
                          line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            p = model(x)
            p = torch.clamp(p, 1e-6, 1 - 1e-6)
            nll = - (wts * (y * torch.log(p) + (1 - y) * torch.log(1 - p))).sum()
            reg = 0.0
            for param in model.parameters(): reg = reg + (param ** 2).sum()
            loss = nll + l2 * reg
            loss.backward()
            return loss

        optimizer.step(closure)
    except Exception:
        optimizer = Adam(model.parameters(), lr=0.05, weight_decay=l2)
        for _ in range(max_iters):
            optimizer.zero_grad()
            p = model(x)
            p = torch.clamp(p, 1e-6, 1 - 1e-6)
            nll = - (wts * (y * torch.log(p) + (1 - y) * torch.log(1 - p))).sum()
            reg = 0.0
            for param in model.parameters(): reg = reg + (param ** 2).sum()
            loss = nll + l2 * reg
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        p = model(x)
        p = torch.clamp(p, 1e-6, 1 - 1e-6)
        nll = - (wts * (y * torch.log(p) + (1 - y) * torch.log(1 - p))).sum()

    k_params = 2
    return float(nll.item()), k_params, model


def bic_from_nll(nll, k, n):
    return 2.0 * nll + k * math.log(max(n, 1))


# ======================= DP: BMI 最优分箱（以 BIC 为段代价） =======================

def prepare_mother_level(male_tests: pd.DataFrame) -> pd.DataFrame:
    mothers = (
        male_tests
        .sort_values([MOTHER_ID, WEEK_COL])
        .groupby(MOTHER_ID)
        .agg({BMI_COL: "first"})
        .reset_index()
        .rename(columns={BMI_COL: "bmi"})
    )
    mothers = mothers.dropna(subset=["bmi"])
    mothers = mothers.sort_values("bmi").reset_index(drop=True)
    mothers["mother_idx"] = np.arange(len(mothers), dtype=int)

    id2rows = male_tests.reset_index().groupby(MOTHER_ID)["index"].apply(lambda s: s.values).to_dict()
    mothers["row_idx"] = mothers[MOTHER_ID].map(id2rows)
    return mothers


def select_bmi_knots(mothers: pd.DataFrame, n_max: int = N_BMI_KNOTS_MAX) -> pd.DataFrame:
    m = len(mothers)
    if m <= n_max:
        return mothers
    idx = np.linspace(0, m - 1, n_max).round().astype(int)
    mothers_small = mothers.iloc[idx].copy()
    mothers_small["mother_idx"] = np.arange(len(mothers_small))
    return mothers_small


def segment_costs_bic(male_tests: pd.DataFrame, mothers_frame: pd.DataFrame) -> Tuple[
    np.ndarray, Dict[Tuple[int, int], Tuple[float, int, Tuple[float, float]]]]:
    m = len(mothers_frame)
    cost = np.full((m, m), NLL_BIG, dtype=float)
    meta: Dict[Tuple[int, int], Tuple[float, int, Tuple[float, float]]] = {}

    weeks_all = male_tests[WEEK_COL].values
    labels_all = male_tests["reach"].values.astype(np.float32)
    weights_all = np.ones_like(labels_all, dtype=np.float32)

    rows_by_m = mothers_frame["row_idx"].tolist()

    for i in range(m):
        union_rows = set()
        for j in range(i, m):
            union_rows.update(rows_by_m[j])
            idx = np.array(sorted(union_rows), dtype=int)
            n = len(idx)
            # 段内至少要有一定样本量
            if n < max(50, MIN_MOTHERS_PER_BIN):
                continue

            weeks = weeks_all[idx]
            labels = labels_all[idx]
            weights = weights_all[idx]
            mask = ~np.isnan(weeks)
            weeks = weeks[mask];
            labels = labels[mask];
            weights = weights[mask]
            if len(weeks) < max(50, MIN_MOTHERS_PER_BIN):
                continue

            try:
                nll, k_params, model = fit_logreg_weeks_torch(weeks, labels, weights)
                bic = bic_from_nll(nll, k_params, len(weeks))
                cost[i, j] = bic
                with torch.no_grad():
                    w = float(model.w.item());
                    b = float(model.b.item())
                meta[(i, j)] = (nll, k_params, (w, b))
            except Exception:
                cost[i, j] = NLL_BIG

    return cost, meta


def dp_optimal_binning(cost: np.ndarray, mothers_frame: pd.DataFrame, k_max: int = K_MAX) -> Tuple[
    List[Tuple[int, int]], float]:
    m = cost.shape[0]
    dp = np.full((k_max + 1, m), np.inf, dtype=float)
    prev = [[-1] * m for _ in range(k_max + 1)]

    for j in range(m):
        dp[1, j] = cost[0, j]

    for k in range(2, k_max + 1):
        for j in range(k - 1, m):
            best = np.inf;
            best_i = -1
            for i in range(k - 1, j + 1):
                c = dp[k - 1, i - 1] + cost[i, j]
                if c < best:
                    best = c;
                    best_i = i
            dp[k, j] = best
            prev[k][j] = best_i

    best_total = np.inf;
    best_k = -1
    for k in range(1, k_max + 1):
        if dp[k, m - 1] < best_total:
            best_total = dp[k, m - 1];
            best_k = k

    segs = []
    j = m - 1;
    k = best_k
    while k >= 1:
        i = prev[k][j] if k > 1 else 0
        segs.append((i, j))
        j = i - 1
        k -= 1
    segs.reverse()
    return segs, float(best_total)


# ======================= 风险最小化（最佳检测周） =======================

def stage_weight(week: float) -> float:
    if week <= EARLY_END:
        return RISK_WEIGHT["early"]
    elif week <= MID_END:
        return RISK_WEIGHT["mid"]
    else:
        return RISK_WEIGHT["late"]


def prob_curve_from_model(model: LogisticWeek, weeks: np.ndarray) -> np.ndarray:
    w = torch.tensor(weeks, dtype=torch.float32, device=DEVICE).view(-1, 1)
    with torch.no_grad():
        p = torch.sigmoid(model.w * w + model.b).cpu().numpy().reshape(-1)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    # 单调化（避免数值波动）
    for i in range(1, len(p)):
        if p[i] < p[i - 1]:
            p[i] = p[i - 1]
    return p


def earliest_distribution_from_curve(pw: np.ndarray, weeks: np.ndarray) -> np.ndarray:
    diff = np.diff(np.concatenate([[0.0], pw]))
    diff = np.maximum(diff, 0.0)
    s = diff.sum()
    if s <= 0: return np.zeros_like(diff)
    return diff / s


def expected_cost_for_test_week(t: float, weeks: np.ndarray, p_first: np.ndarray) -> float:
    cost = 0.0
    for wk, pk in zip(weeks, p_first):
        detect_week = wk if wk >= t else t
        cost += stage_weight(detect_week) * pk
    return cost


def choose_best_test_week_from_model(model: LogisticWeek, weeks_grid, test_week_cand) -> Tuple[
    float, float, np.ndarray, np.ndarray, np.ndarray]:
    pw = prob_curve_from_model(model, weeks_grid)
    p_first = earliest_distribution_from_curve(pw, weeks_grid)
    costs = [expected_cost_for_test_week(t, weeks_grid, p_first) for t in test_week_cand]
    idx = int(np.argmin(costs))
    return float(test_week_cand[idx]), float(costs[idx]), costs, pw, p_first


def choose_week_by_target_prob(model: LogisticWeek, weeks_grid, target=0.90):
    """
    选择能达到目标通过率的最早检测周
    """
    pw = prob_curve_from_model(model, weeks_grid)  # 已单调化的概率曲线
    idx = np.where(pw >= target)[0]

    if idx.size == 0:
        # 如果到最晚也达不到目标，返回最晚周和对应概率
        return float(weeks_grid[-1]), float(pw[-1])

    # 返回第一个达到目标的周
    best_week = float(weeks_grid[idx[0]])
    actual_prob = float(pw[idx[0]])
    return best_week, actual_prob


def choose_week_sensitivity_analysis(model: LogisticWeek, weeks_grid, targets=[0.85, 0.90, 0.95]):
    """
    对多个通过率目标进行敏感性分析
    """
    results = {}
    pw = prob_curve_from_model(model, weeks_grid)

    for target in targets:
        idx = np.where(pw >= target)[0]
        if idx.size == 0:
            results[f"target_{target:.0%}"] = {
                "best_week": float(weeks_grid[-1]),
                "actual_prob": float(pw[-1]),
                "achievable": False
            }
        else:
            results[f"target_{target:.0%}"] = {
                "best_week": float(weeks_grid[idx[0]]),
                "actual_prob": float(pw[idx[0]]),
                "achievable": True
            }

    return results


# ======================= 增强可视化（包含误差分析，全中文标签） =======================

def scatter_week_reach(ax, weeks, reach, fp: FontProperties):
    # 将 0/1 加一点抖动，便于可视化
    jitter = (np.random.rand(len(reach)) - 0.5) * 0.06
    ax.scatter(weeks, reach + jitter, s=10, alpha=0.4, edgecolor="none")
    ax.set_xlabel(CHINESE_LABELS["week"], fontproperties=fp)
    ax.set_ylabel(CHINESE_LABELS["reach_status"], fontproperties=fp)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([CHINESE_LABELS["not_reached"], CHINESE_LABELS["reached"]], fontproperties=fp)


def plot_bin_all_with_errors(fig_prefix: str,
                             group_label: str,
                             model: LogisticWeek,
                             best_week: float,
                             best_cost: float,
                             costs: List[float],
                             weeks_grid: np.ndarray,
                             pw: np.ndarray,
                             p_first: np.ndarray,
                             sub_df: pd.DataFrame,
                             error_result: ErrorModelingResult,
                             actual_prob: float = None,
                             sensitivity_results: Dict = None,
                             group_min: float = None,
                             group_max: float = None,
                             group_report: Dict = None):
    """增强版可视化，显示每组独立的真实数据范围，全中文标签"""
    fp = get_chinese_font()

    # 检查边界限制
    boundary_warning = ""
    if group_min is not None and group_max is not None:
        if abs(best_week - group_min) < 0.2:
            boundary_warning += f" {CHINESE_LABELS['boundary_warning_lower']}({group_min:.1f})"
        if abs(best_week - group_max) < 0.2:
            boundary_warning += f" {CHINESE_LABELS['boundary_warning_upper']}({group_max:.1f})"

    # 添加数据质量信息
    quality_info = ""
    if group_report and "analysis_notes" in group_report:
        if group_report["analysis_notes"]:
            quality_info = f"\n{CHINESE_LABELS['quality_reminder']}: {'; '.join(group_report['analysis_notes'][:2])}"

    # 1) 达标概率曲线 + 经验点 + 通过率阈值线 + 置信区间
    fig1, ax1 = plt.subplots(figsize=(8, 5), dpi=140)
    scatter_week_reach(ax1, sub_df[WEEK_COL].values, sub_df["reach"].values, fp)
    ax1.plot(weeks_grid, pw, lw=2, label=CHINESE_LABELS["logistic_prob"], zorder=3)

    label_text = f"{CHINESE_LABELS['optimal_week']} = {best_week:.1f}周{boundary_warning}"
    if actual_prob is not None:
        label_text += f" (90%目标)"
        ax1.axhline(0.90, ls=":", alpha=0.8, lw=1.5, c="orange", label=CHINESE_LABELS["target_90pct"])
    ax1.axvline(best_week, ls="--", c="r", lw=2, label=label_text)

    # 显示该组的分析范围
    if group_min is not None and group_max is not None:
        ax1.axvspan(group_min, group_max, alpha=0.1, color='green',
                    label=f"{CHINESE_LABELS['analysis_range']} [{group_min:.1f}-{group_max:.1f}]")

    # 敏感性分析线
    if sensitivity_results:
        for target_key, result in sensitivity_results.items():
            if result["achievable"]:
                target_pct = float(target_key.split('_')[1].rstrip('%')) / 100
                week = result["best_week"]
                if target_pct == 0.85:
                    label = f"{CHINESE_LABELS['target_85pct']}: {week:.1f}周"
                elif target_pct == 0.95:
                    label = f"{CHINESE_LABELS['target_95pct']}: {week:.1f}周"
                else:
                    continue
                ax1.axvline(week, ls=":", alpha=0.6, lw=1, label=label)
                ax1.axhline(target_pct, ls=":", alpha=0.4, lw=1)

    # 添加置信区间
    if not np.isnan(error_result.confidence_interval_95[0]):
        ax1.axvspan(error_result.confidence_interval_95[0], error_result.confidence_interval_95[1],
                    alpha=0.2, color='red', label=CHINESE_LABELS["confidence_interval_95"])
        ax1.axvspan(error_result.confidence_interval_90[0], error_result.confidence_interval_90[1],
                    alpha=0.3, color='orange', label=CHINESE_LABELS["confidence_interval_90"])

    # 标题包含实际概率和质量信息
    if actual_prob is not None:
        title_text = f"{group_label} - {CHINESE_LABELS['prob_curve_title']}\n{CHINESE_LABELS['actual_achieved']}: {actual_prob:.1%}"
    else:
        title_text = f"{group_label} - {CHINESE_LABELS['prob_curve_title']}"

    if boundary_warning:
        title_text += f"\n{boundary_warning.strip()}"
    if quality_info:
        title_text += quality_info

    ax1.set_title(title_text, fontproperties=fp, fontsize=10)
    ax1.legend(prop=fp, frameon=True, loc='lower right', fontsize=8)
    ax1.set_ylim(-0.05, 1.05)
    fig1.tight_layout()
    fig1.savefig(os.path.join(PLOT_DIR, f"{fig_prefix}_prob_curve_with_targets.png"))
    plt.close(fig1)

    # 2) 该组数据覆盖可视化
    fig2, ax2 = plt.subplots(figsize=(8, 5), dpi=140)

    # 绘制该组的Y浓度分布
    ax2.scatter(sub_df[WEEK_COL].values, sub_df[Y_FRAC_COL].values, s=15, alpha=0.6)

    # 各种阈值线
    ax2.axhline(THRESH_Y_FRAC, ls="--", c="r", lw=2,
                label=f"{CHINESE_LABELS['main_threshold']} {THRESH_Y_FRAC:.2%}")
    for thresh in THRESH_SENSITIVITY:
        if thresh != THRESH_Y_FRAC:
            ax2.axhline(thresh, ls=":", c="gray", alpha=0.7, lw=1,
                        label=f"{CHINESE_LABELS['sensitivity_threshold']} {thresh:.2%}")

    # 最优检测周标记
    ax2.axvline(best_week, ls="--", c="blue", lw=2, alpha=0.8,
                label=f"{CHINESE_LABELS['optimal_week']} {best_week:.1f}")

    # 该组的数据范围和分析范围
    if group_report:
        data_range = group_report.get("data_range", [])
        if len(data_range) == 2:
            ax2.axvspan(data_range[0], data_range[1], alpha=0.15, color='blue',
                        label=f"{CHINESE_LABELS['data_range']} [{data_range[0]:.1f}-{data_range[1]:.1f}]")

        analysis_range = group_report.get("analysis_range", [])
        if len(analysis_range) == 2 and group_min is not None and group_max is not None:
            # 只显示超出数据范围的分析扩展部分
            if group_min < data_range[0]:
                ax2.axvspan(group_min, data_range[0], alpha=0.1, color='green',
                            label=CHINESE_LABELS["analysis_extension"])
            if group_max > data_range[1]:
                ax2.axvspan(data_range[1], group_max, alpha=0.1, color='green')

    ax2.set_xlabel(CHINESE_LABELS["week"], fontproperties=fp)
    ax2.set_ylabel(CHINESE_LABELS["y_frac"], fontproperties=fp)
    ax2.set_title(f"{group_label} - {CHINESE_LABELS['data_coverage_title']}", fontproperties=fp)
    ax2.legend(prop=fp, fontsize=8)
    fig2.tight_layout()
    fig2.savefig(os.path.join(PLOT_DIR, f"{fig_prefix}_data_coverage.png"))
    plt.close(fig2)

    # 3) Bootstrap分布可视化（如果有的话）
    if error_result.bootstrap_weeks:
        fig3, ax3 = plt.subplots(figsize=(8, 5), dpi=140)
        ax3.hist(error_result.bootstrap_weeks, bins=30, alpha=0.7, density=True,
                 label=f'{CHINESE_LABELS["bootstrap_dist"]} (n={len(error_result.bootstrap_weeks)})')
        ax3.axvline(best_week, ls="--", c="r", lw=2,
                    label=f"通过率法最优周 = {best_week:.1f}")
        ax3.axvline(np.mean(error_result.bootstrap_weeks), ls=":", c="blue", lw=2,
                    label=f"{CHINESE_LABELS['bootstrap_mean']} = {np.mean(error_result.bootstrap_weeks):.1f}")
        ax3.set_xlabel(CHINESE_LABELS["detection_week"], fontproperties=fp)
        ax3.set_ylabel(CHINESE_LABELS["probability_density"], fontproperties=fp)
        ax3.set_title(f"{group_label} - {CHINESE_LABELS['bootstrap_title']}", fontproperties=fp)
        ax3.legend(prop=fp)
        fig3.tight_layout()
        fig3.savefig(os.path.join(PLOT_DIR, f"{fig_prefix}_bootstrap_dist.png"))
        plt.close(fig3)


def plot_overview_bins_with_per_group_ranges(mothers_full: pd.DataFrame,
                                             bins_edges: List[Tuple[float, float]],
                                             result_df: pd.DataFrame):
    """总览图，显示每组不同的分析范围，全中文标签"""
    fp = get_chinese_font()

    # BMI 直方图 + 分箱边界
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.hist(mothers_full["bmi"].values, bins=40, alpha=0.7)
    for (lo, hi) in bins_edges:
        ax.axvline(lo, c="r", ls="--", lw=1)
        ax.axvline(hi, c="r", ls="--", lw=1)
    ax.set_xlabel(CHINESE_LABELS["bmi"], fontproperties=fp)
    ax.set_ylabel(CHINESE_LABELS["count"], fontproperties=fp)
    ax.set_title(CHINESE_LABELS["bmi_dist_title"], fontproperties=fp)
    fig.tight_layout()
    fig.savefig(os.path.join(OVERVIEW_DIR, "bmi_bins_hist.png"))
    plt.close(fig)

    # 各组最佳检测周条形图 - 显示不同范围
    fig2, ax2 = plt.subplots(figsize=(14, 8), dpi=150)
    labels = []
    colors = []

    for _, r in result_df.iterrows():
        label = f"[{r['bmi_min']:.1f}, {r['bmi_max']:.1f}]"

        # 根据分析范围的差异设置颜色
        range_width = r.get('analysis_week_max', 30) - r.get('analysis_week_min', 10)
        if range_width < 15:
            colors.append('lightcoral')  # 范围较窄
            label += f" {CHINESE_LABELS['narrow_range']}"
        elif range_width > 25:
            colors.append('lightblue')  # 范围较宽
            label += f" {CHINESE_LABELS['wide_range']}"
        else:
            colors.append('lightgreen')  # 中等范围

        if r.get('boundary_limited', False):
            label += f" {CHINESE_LABELS['boundary_limited']}"
        labels.append(label)

    x_pos = np.arange(len(labels))

    # 主要结果条形图
    bars = ax2.bar(x_pos, result_df["best_week"].values, alpha=0.85,
                   color=colors, capsize=5)

    # 添加每组的分析范围作为误差条
    if 'analysis_week_min' in result_df.columns and 'analysis_week_max' in result_df.columns:
        for i, (bar, _, row) in enumerate(zip(bars, x_pos, result_df.iterrows())):
            min_week = row[1].get('analysis_week_min', np.nan)
            max_week = row[1].get('analysis_week_max', np.nan)
            best_week = row[1]['best_week']

            if not (pd.isna(min_week) or pd.isna(max_week)):
                # 绘制分析范围
                ax2.plot([i, i], [min_week, max_week], 'k-', alpha=0.3, lw=8, zorder=0)
                ax2.plot([i, i], [min_week, max_week], 'gray', alpha=0.8, lw=2, zorder=1)

    # 添加置信区间（如果有）
    if 'ci_95_lower' in result_df.columns and 'ci_95_upper' in result_df.columns:
        ci_lo = result_df["ci_95_lower"].to_numpy(dtype=float)
        ci_hi = result_df["ci_95_upper"].to_numpy(dtype=float)
        y = result_df["best_week"].to_numpy(dtype=float)

        valid = (~np.isnan(ci_lo)) & (~np.isnan(ci_hi)) & (~np.isnan(y)) & (ci_hi >= ci_lo)
        if valid.any():
            lower_err = np.maximum(0.0, y[valid] - ci_lo[valid])
            upper_err = np.maximum(0.0, ci_hi[valid] - y[valid])
            yerr = np.vstack([lower_err, upper_err])

            ax2.errorbar(x_pos[valid], y[valid],
                         yerr=yerr, fmt='none',
                         ecolor='red', elinewidth=2, capsize=4, capthick=2)

    ax2.set_xlabel("BMI组", fontproperties=fp)
    ax2.set_ylabel(CHINESE_LABELS["detection_week"], fontproperties=fp)
    title_text = f"{CHINESE_LABELS['optimal_weeks_title']}\n灰线={CHINESE_LABELS['analysis_range']}, 红线={CHINESE_LABELS['confidence_interval_95']}, {CHINESE_LABELS['boundary_limited']}=边界限制"
    ax2.set_title(title_text, fontproperties=fp)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontproperties=fp, fontsize=9)

    # 添加数值标签
    for i, (v, bar) in enumerate(zip(result_df["best_week"].values, bars)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.3, f"{v:.1f}",
                 ha="center", va="bottom", fontproperties=fp, fontsize=8, weight='bold')

    fig2.tight_layout()
    fig2.savefig(os.path.join(OVERVIEW_DIR, "best_weeks_with_ranges.png"))
    plt.close(fig2)

    # 分析范围对比图
    if 'analysis_week_min' in result_df.columns and 'analysis_week_max' in result_df.columns:
        fig3, ax3 = plt.subplots(figsize=(12, 6), dpi=150)

        # 绘制每组的分析范围
        for i, (_, row) in enumerate(result_df.iterrows()):
            min_week = row.get('analysis_week_min', np.nan)
            max_week = row.get('analysis_week_max', np.nan)
            data_min = row.get('data_week_min', np.nan)
            data_max = row.get('data_week_max', np.nan)

            if not pd.isna(min_week) and not pd.isna(max_week):
                # 分析范围
                ax3.barh(i, max_week - min_week, left=min_week, alpha=0.3,
                         color='blue', label=CHINESE_LABELS["analysis_range"] if i == 0 else "")

                # 数据范围
                if not pd.isna(data_min) and not pd.isna(data_max):
                    ax3.barh(i, data_max - data_min, left=data_min, alpha=0.7,
                             color='green', label=CHINESE_LABELS["data_range"] if i == 0 else "")

                # 最优检测周
                ax3.plot(row['best_week'], i, 'ro', markersize=8,
                         label=CHINESE_LABELS["optimal_detection_week"] if i == 0 else "")

        ax3.set_yticks(range(len(result_df)))
        ax3.set_yticklabels([f"[{r['bmi_min']:.1f}, {r['bmi_max']:.1f}]"
                             for _, r in result_df.iterrows()], fontproperties=fp)
        ax3.set_xlabel(CHINESE_LABELS["week"], fontproperties=fp)
        ax3.set_ylabel("BMI组", fontproperties=fp)
        ax3.set_title(CHINESE_LABELS["analysis_ranges_title"], fontproperties=fp)
        ax3.legend(prop=fp)
        ax3.grid(True, alpha=0.3)

        fig3.tight_layout()
        fig3.savefig(os.path.join(OVERVIEW_DIR, "analysis_ranges_comparison.png"))
        plt.close(fig3)


# ======================= 主流程（每组独立分析版本） =======================

def run_per_group_analysis_pipeline():
    """每组独立分析的完整流程"""

    print("=== NIPT 每组独立真实数据分析 ===\n")

    # 1. 加载数据
    male, earliest = load_data()

    # 显示全局数据概览
    all_weeks = male[WEEK_COL].dropna()
    print(f"[INFO] 全局数据概览:")
    print(f"  总样本: {len(all_weeks)} 次检测")
    print(f"  孕周范围: {all_weeks.min():.1f} - {all_weeks.max():.1f} 周")
    print(f"  平均孕周: {all_weeks.mean():.1f} ± {all_weeks.std():.1f} 周")
    print(f"  总体达标率: {male['reach'].mean():.3f}")

    # 2. BMI分箱准备
    mothers_full = prepare_mother_level(male)
    mothers = select_bmi_knots(mothers_full, n_max=N_BMI_KNOTS_MAX)
    print(f"\n[INFO] BMI分箱: 使用 {len(mothers)} 个节点 (来自 {len(mothers_full)} 位孕妇)")

    # 3. 迭代优化分箱
    best_bins_idx: List[Tuple[int, int]] = []
    best_bic = np.inf

    for outer in range(1, N_OUTER_ITERS + 1):
        print(f"\n[ITER] ===== 外层迭代 {outer} =====")
        cost_mat, meta = segment_costs_bic(male, mothers)
        bins_idx, total_bic = dp_optimal_binning(cost_mat, mothers, k_max=K_MAX)
        print(f"[ITER] 分箱数量: {len(bins_idx)}, 总BIC={total_bic:.2f}")

        if total_bic < best_bic:
            best_bins_idx, best_bic = bins_idx, total_bic

    print(f"\n[RESULT] 最终分箱数量: {len(best_bins_idx)}")

    # 4. 对每个BMI组进行独立分析
    bin_infos = []
    bins_edges_real: List[Tuple[float, float]] = []
    all_bootstrap_results = []
    group_reports = []

    print("\n[INFO] 开始每组独立分析...")

    for k, (i, j) in enumerate(best_bins_idx):
        print(f"\n{'=' * 60}")
        print(f"[GROUP {k + 1}/{len(best_bins_idx)}] 分析BMI组")

        bmi_min = float(mothers.iloc[i]["bmi"])
        bmi_max = float(mothers.iloc[j]["bmi"])
        bins_edges_real.append((bmi_min, bmi_max))

        # 获取该BMI组的所有数据
        in_bin = mothers_full[(mothers_full["bmi"] >= bmi_min) & (mothers_full["bmi"] <= bmi_max)]
        mother_ids = set(in_bin[MOTHER_ID].tolist())
        tests_in_bin = male[male[MOTHER_ID].isin(mother_ids)].copy()

        # **关键修改：为该组独立确定分析范围**
        group_min, group_max, group_report = determine_week_range_per_group(tests_in_bin, bmi_min, bmi_max)
        group_reports.append(group_report)

        print(f"BMI范围: [{bmi_min:.1f}, {bmi_max:.1f}]")
        print(f"该组数据范围: {group_report.get('data_range', ['N/A', 'N/A'])}")
        print(f"该组分析范围: [{group_min:.1f}, {group_max:.1f}] (独立计算)")
        print(f"该组样本量: {group_report.get('sample_count', 0)} 次检测, {in_bin.shape[0]} 位孕妇")

        if "error" in group_report:
            print(f"[WARNING] 跳过该组: {group_report['error']}")
            continue

        # 创建该组的独立分析网格
        weeks_grid_group = np.arange(group_min, group_max + 0.01, 0.1)
        test_cand_group = np.arange(group_min, group_max + 0.01, 0.5)

        print(f"该组网格点数: {len(weeks_grid_group)} (范围: {group_min:.1f}-{group_max:.1f})")

        # 准备该组的训练数据
        weeks = tests_in_bin[WEEK_COL].values
        labels = tests_in_bin["reach"].astype(float).values
        mask = ~np.isnan(weeks)
        weeks_clean = weeks[mask]
        labels_clean = labels[mask]

        if len(weeks_clean) < 15:
            print(f"[WARNING] 该组样本不足({len(weeks_clean)})，跳过分析")
            continue

        print(f"有效样本: {len(weeks_clean)} 次检测")
        print(f"该组达标率: {labels_clean.mean():.3f}")

        # 拟合逻辑回归模型
        try:
            nll, k_params, model = fit_logreg_weeks_torch(weeks_clean, labels_clean)
            print(f"模型拟合完成: NLL={nll:.2f}")
        except Exception as e:
            print(f"[ERROR] 模型拟合失败: {e}")
            continue

        # 使用该组独立范围计算最优检测周
        best_week_group, actual_prob = choose_week_by_target_prob(
            model, weeks_grid=weeks_grid_group, target=0.90)

        # 通过率敏感性分析
        sensitivity_results = choose_week_sensitivity_analysis(
            model, weeks_grid=weeks_grid_group, targets=[0.85, 0.90, 0.95])

        # 综合误差建模
        error_result = comprehensive_error_modeling(tests_in_bin, weeks_grid_group)

        # 边界检查
        boundary_limited = (abs(best_week_group - group_min) < 0.2) or (abs(best_week_group - group_max) < 0.2)

        # 输出该组结果
        print(f"该组结果:")
        print(f"  90%目标最优检测周: {best_week_group:.1f} 周 (实际达到: {actual_prob:.1%})")

        if boundary_limited:
            if abs(best_week_group - group_min) < 0.2:
                print(f"  ⚠️  结果可能被该组下限({group_min:.1f})限制")
            if abs(best_week_group - group_max) < 0.2:
                print(f"  ⚠️  结果可能被该组上限({group_max:.1f})限制")
        else:
            print(f"  ✓ 结果未受该组边界限制")

        # 不确定性信息
        if not np.isnan(error_result.uncertainty_range):
            print(
                f"  95%置信区间: [{error_result.confidence_interval_95[0]:.1f}, {error_result.confidence_interval_95[1]:.1f}] 周")

        # 生成该组的可视化
        group_label = f"BMI [{bmi_min:.1f}, {bmi_max:.1f}]"
        fig_prefix = f"group_{k}_{bmi_min:.1f}_{bmi_max:.1f}".replace(".", "p")

        # 为了兼容，也计算风险最小化结果
        try:
            best_week_risk, best_cost_risk, costs, pw, p_first = choose_best_test_week_from_model(
                model, weeks_grid=weeks_grid_group, test_week_cand=test_cand_group)
        except:
            best_week_risk, best_cost_risk = best_week_group, 0.0
            pw = prob_curve_from_model(model, weeks_grid_group)
            p_first = earliest_distribution_from_curve(pw, weeks_grid_group)
            costs = [0.0] * len(test_cand_group)

        plot_bin_all_with_errors(
            fig_prefix, group_label, model,
            best_week_group, 0.0, costs, weeks_grid_group, pw, p_first,
            tests_in_bin, error_result,
            actual_prob=actual_prob,
            sensitivity_results=sensitivity_results,
            group_min=group_min,
            group_max=group_max,
            group_report=group_report
        )

        # Bootstrap结果记录
        if error_result.bootstrap_weeks:
            bootstrap_result = {
                'bmi_group': f"[{bmi_min:.1f}, {bmi_max:.1f}]",
                'bootstrap_weeks': error_result.bootstrap_weeks,
                'n_bootstrap': len(error_result.bootstrap_weeks),
                'bootstrap_mean': np.mean(error_result.bootstrap_weeks),
                'bootstrap_std': np.std(error_result.bootstrap_weeks)
            }
            all_bootstrap_results.append(bootstrap_result)

        # 获取模型参数
        with torch.no_grad():
            w = float(model.w.item())
            b = float(model.b.item())

        # 构建该组结果记录
        bin_info = {
            "bmi_min": bmi_min,
            "bmi_max": bmi_max,
            "n_mothers": int(in_bin.shape[0]),
            "n_tests": int(len(weeks_clean)),

            # 该组独立的数据和分析范围
            "data_week_min": float(group_report['data_range'][0]) if 'data_range' in group_report else np.nan,
            "data_week_max": float(group_report['data_range'][1]) if 'data_range' in group_report else np.nan,
            "analysis_week_min": group_min,
            "analysis_week_max": group_max,
            "range_width": group_max - group_min,

            # 模型参数
            "w_param": w,
            "b_param": b,

            # 主结果：通过率法
            "best_week": best_week_group,
            "target_prob": 0.90,
            "actual_prob": actual_prob,
            "boundary_limited": boundary_limited,

            # 通过率敏感性
            "week_85pct": sensitivity_results["target_85%"]["best_week"],
            "week_90pct": sensitivity_results["target_90%"]["best_week"],
            "week_95pct": sensitivity_results["target_95%"]["best_week"],
            "prob_85pct": sensitivity_results["target_85%"]["actual_prob"],
            "prob_90pct": sensitivity_results["target_90%"]["actual_prob"],
            "prob_95pct": sensitivity_results["target_95%"]["actual_prob"],

            # 风险最小化结果（作为对比）
            "best_week_risk": best_week_risk,
            "expected_cost": best_cost_risk,

            # 误差分析
            "ci_95_lower": error_result.confidence_interval_95[0],
            "ci_95_upper": error_result.confidence_interval_95[1],
            "ci_90_lower": error_result.confidence_interval_90[0],
            "ci_90_upper": error_result.confidence_interval_90[1],
            "uncertainty_range": error_result.uncertainty_range,
            "qc_noise_impact": error_result.qc_noise_impact,
            "bootstrap_mean": np.mean(error_result.bootstrap_weeks) if error_result.bootstrap_weeks else np.nan,
            "bootstrap_std": np.std(error_result.bootstrap_weeks) if error_result.bootstrap_weeks else np.nan,

            # 数据质量标记
            "data_quality_notes": "; ".join(group_report.get('analysis_notes', [])) if group_report.get(
                'analysis_notes') else ""
        }

        # 添加阈值敏感性结果
        for thresh, sens_week in error_result.threshold_sensitivity.items():
            bin_info[f"thresh_sens_{thresh * 100:.1f}"] = sens_week

        bin_infos.append(bin_info)

    # 5. 保存结果
    if all_bootstrap_results:
        bootstrap_df = pd.DataFrame(all_bootstrap_results)
        bootstrap_df.to_csv(os.path.join(BOOTSTRAP_DIR, "bootstrap_results.csv"),
                            index=False, encoding="utf-8-sig")

    # 保存每组分析报告
    save_group_analysis_reports(group_reports, OUT_DIR)

    # 主结果表
    result_df = pd.DataFrame(bin_infos).sort_values("bmi_min").reset_index(drop=True)

    # 保存详细结果
    out_csv = os.path.join(OUT_DIR, "per_group_optimal_weeks.csv")
    result_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 6. 结果总结
    print("\n" + "=" * 80)
    print("每组独立分析结果总结")
    print("=" * 80)

    # 显示核心结果
    display_cols = ["bmi_min", "bmi_max", "best_week", "actual_prob", "analysis_week_min", "analysis_week_max",
                    "boundary_limited"]
    available_cols = [col for col in display_cols if col in result_df.columns]
    print("\n核心结果:")
    print(result_df[available_cols].to_string(index=False))

    # 分析范围多样性检查
    if 'analysis_week_min' in result_df.columns and 'analysis_week_max' in result_df.columns:
        min_ranges = result_df['analysis_week_min'].values
        max_ranges = result_df['analysis_week_max'].values

        print(f"\n分析范围多样性检查:")
        print(f"  最小分析起点: {min_ranges.min():.1f} 周")
        print(f"  最大分析起点: {min_ranges.max():.1f} 周")
        print(f"  最小分析终点: {max_ranges.min():.1f} 周")
        print(f"  最大分析终点: {max_ranges.max():.1f} 周")
        print(
            f"  范围是否相同: {'否' if (min_ranges.max() - min_ranges.min() > 0.1 or max_ranges.max() - max_ranges.min() > 0.1) else '是'}")

    # 边界限制警告
    boundary_limited_count = result_df['boundary_limited'].sum() if 'boundary_limited' in result_df else 0
    if boundary_limited_count > 0:
        print(f"\n⚠️  警告: {boundary_limited_count}/{len(result_df)} 个BMI组可能受到各自分析边界限制")

    # 最优检测周统计
    print(f"\n最优检测周分布:")
    print(f"  平均: {result_df['best_week'].mean():.1f} ± {result_df['best_week'].std():.1f} 周")
    print(f"  范围: {result_df['best_week'].min():.1f} - {result_df['best_week'].max():.1f} 周")
    print(f"  中位数: {result_df['best_week'].median():.1f} 周")

    # 早期检测可行性
    early_feasible = (result_df['best_week'] < 12).sum()
    if early_feasible > 0:
        print(f"  早期检测(<12周)可行的组数: {early_feasible}/{len(result_df)}")

    print(f"\n[OK] 每组独立分析完成，结果已保存：{out_csv}")

    # 7. 增强版总览图
    plot_overview_bins_with_per_group_ranges(mothers_full, bins_edges_real, result_df)

    # 8. 生成综合报告
    generate_per_group_analysis_report(result_df, group_reports, out_dir=OUT_DIR)

    return result_df


def generate_per_group_analysis_report(result_df: pd.DataFrame, group_reports: List[Dict], out_dir: str):
    """生成每组独立分析的综合报告"""

    # 分析范围多样性统计
    range_diversity = {
        "min_analysis_start": float(result_df['analysis_week_min'].min()) if 'analysis_week_min' in result_df else None,
        "max_analysis_start": float(result_df['analysis_week_min'].max()) if 'analysis_week_min' in result_df else None,
        "min_analysis_end": float(result_df['analysis_week_max'].min()) if 'analysis_week_max' in result_df else None,
        "max_analysis_end": float(result_df['analysis_week_max'].max()) if 'analysis_week_max' in result_df else None,
        "avg_range_width": float(result_df['range_width'].mean()) if 'range_width' in result_df else None,
    }

    # 数据质量统计
    quality_stats = {
        "total_groups": len(result_df),
        "groups_with_warnings": sum(1 for report in group_reports if report.get('analysis_notes')),
        "boundary_limited_groups": int(result_df['boundary_limited'].sum()) if 'boundary_limited' in result_df else 0,
        "avg_samples_per_group": float(result_df['n_tests'].mean()) if 'n_tests' in result_df else None,
    }

    # 结果多样性
    result_diversity = {
        "optimal_week_range": [float(result_df['best_week'].min()), float(result_df['best_week'].max())],
        "early_detection_feasible_groups": int((result_df['best_week'] < 12).sum()),
        "high_confidence_groups": int((result_df['actual_prob'] > 0.92).sum()) if 'actual_prob' in result_df else 0,
    }

    report = {
        "analysis_type": "per_group_independent_analysis",
        "summary": {
            "range_diversity": range_diversity,
            "quality_stats": quality_stats,
            "result_diversity": result_diversity
        },
        "group_details": group_reports,
        "recommendations": []
    }

    # 生成建议
    recommendations = []

    if range_diversity["max_analysis_start"] - range_diversity["min_analysis_start"] > 5:
        recommendations.append("各组分析范围差异较大，说明每组独立分析的必要性得到体现")

    if quality_stats["boundary_limited_groups"] > 0:
        recommendations.append(f"{quality_stats['boundary_limited_groups']} 个组受边界限制，建议针对性扩展数据收集")

    if result_diversity["early_detection_feasible_groups"] > 0:
        recommendations.append(f"{result_diversity['early_detection_feasible_groups']} 个组支持早期检测，具有临床价值")

    if quality_stats["groups_with_warnings"] > len(result_df) * 0.5:
        recommendations.append("超过半数组存在数据质量问题，建议提升数据收集质量")

    report["recommendations"] = recommendations

    # 保存报告
    with open(os.path.join(out_dir, "per_group_analysis_report.json"), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 文本报告
    with open(os.path.join(out_dir, "per_group_analysis_summary.txt"), 'w', encoding='utf-8') as f:
        f.write("NIPT 每组独立分析综合报告\n")
        f.write("=" * 70 + "\n\n")

        f.write("1. 分析方法概述\n")
        f.write("-" * 30 + "\n")
        f.write("本分析为每个BMI组独立确定分析范围，避免全局范围限制\n")
        f.write("每组基于自身数据分布计算最优检测周，提高结果针对性\n\n")

        f.write("2. 分析范围多样性\n")
        f.write("-" * 30 + "\n")
        f.write(
            f"起点范围: {range_diversity['min_analysis_start']:.1f} - {range_diversity['max_analysis_start']:.1f} 周\n")
        f.write(f"终点范围: {range_diversity['min_analysis_end']:.1f} - {range_diversity['max_analysis_end']:.1f} 周\n")
        f.write(f"平均范围宽度: {range_diversity['avg_range_width']:.1f} 周\n\n")

        f.write("3. 结果多样性\n")
        f.write("-" * 30 + "\n")
        f.write(
            f"最优检测周范围: {result_diversity['optimal_week_range'][0]:.1f} - {result_diversity['optimal_week_range'][1]:.1f} 周\n")
        f.write(
            f"支持早期检测的组数: {result_diversity['early_detection_feasible_groups']}/{quality_stats['total_groups']}\n")
        f.write(f"高置信度组数(>92%): {result_diversity['high_confidence_groups']}/{quality_stats['total_groups']}\n\n")

        f.write("4. 数据质量评估\n")
        f.write("-" * 30 + "\n")
        f.write(f"总组数: {quality_stats['total_groups']}\n")
        f.write(f"有质量警告的组数: {quality_stats['groups_with_warnings']}\n")
        f.write(f"受边界限制的组数: {quality_stats['boundary_limited_groups']}\n")
        f.write(f"平均每组样本数: {quality_stats['avg_samples_per_group']:.0f}\n\n")

        f.write("5. 临床建议\n")
        f.write("-" * 30 + "\n")
        for i, rec in enumerate(recommendations, 1):
            f.write(f"{i}. {rec}\n")

    print(f"[OK] 每组独立分析报告已保存至 {out_dir}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result_df = run_per_group_analysis_pipeline()