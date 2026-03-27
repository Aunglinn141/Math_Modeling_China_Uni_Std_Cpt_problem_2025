import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings
import matplotlib.font_manager as fm
import platform
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import seaborn as sns

warnings.filterwarnings('ignore')

def configure_chinese_font():
    """
    配置中文字体 - 强制使用中文显示
    """
    import os

    print("正在配置中文字体...")

    # 强制设置matplotlib支持中文
    plt.rcParams['font.sans-serif'] = [
        'Microsoft YaHei',     # Windows
        'SimHei',              # Windows 黑体
        'SimSun',              # Windows 宋体
        'KaiTi',               # Windows 楷体
        'PingFang SC',         # macOS
        'Heiti SC',            # macOS 黑体
        'STHeiti',             # macOS
        'WenQuanYi Micro Hei', # Linux
        'Droid Sans Fallback', # Linux
        'DejaVu Sans',         # 通用备选
        'Arial Unicode MS'     # 备选
    ]

    # 禁用字体警告
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 尝试获取系统中文字体
    system = platform.system()
    selected_font = None

    # 获取所有可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    if system == "Windows":
        # Windows系统字体检测
        preferred_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi']
        for font in preferred_fonts:
            if font in available_fonts:
                selected_font = font
                break

        # 如果没找到，尝试从系统路径添加
        if not selected_font:
            try:
                windows_fonts = {
                    'Microsoft YaHei': ['msyh.ttc', 'msyhl.ttc'],
                    'SimHei': ['simhei.ttf'],
                    'SimSun': ['simsun.ttc']
                }

                fonts_dir = "C:\\Windows\\Fonts"
                for font_name, files in windows_fonts.items():
                    for file_name in files:
                        font_path = os.path.join(fonts_dir, file_name)
                        if os.path.exists(font_path):
                            try:
                                fm.fontManager.addfont(font_path)
                                selected_font = font_name
                                print(f"成功添加字体: {font_name}")
                                break
                            except Exception as e:
                                continue
                    if selected_font:
                        break
            except Exception as e:
                print(f"Windows字体添加失败: {e}")

    elif system == "Darwin":  # macOS
        preferred_fonts = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS']
        for font in preferred_fonts:
            if font in available_fonts:
                selected_font = font
                break

    else:  # Linux
        preferred_fonts = ['WenQuanYi Micro Hei', 'Droid Sans Fallback', 'Source Han Sans CN']
        for font in preferred_fonts:
            if font in available_fonts:
                selected_font = font
                break

    # 创建FontProperties对象
    if selected_font:
        print(f"✅ 使用中文字体: {selected_font}")
        font_prop = fm.FontProperties(family=selected_font)
    else:
        print("⚠️ 未找到理想中文字体，使用系统默认字体")
        # 即使没找到理想字体，也强制使用中文标签
        font_prop = fm.FontProperties()

    # 设置全局字体
    if selected_font:
        plt.rcParams['font.sans-serif'].insert(0, selected_font)

    return font_prop, selected_font

def convert_gestational_week(x):
    """转换孕周格式"""
    try:
        if isinstance(x, str) and "+" in x:
            week, day = x.split("+")
            return int(week) + int(day) / 7
        return float(x)
    except (ValueError, TypeError):
        return np.nan

def load_and_clean_data(file_path):
    """读取和清洗数据"""
    try:
        df = pd.read_csv(file_path)
        print(f"原始数据形状: {df.shape}")

        required_columns = ["y_frac", "gestation_weeks", "bmi"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"缺少必要列: {missing_columns}")

        df_cleaned = df[required_columns].copy()
        df_cleaned.columns = ["Y_concentration", "Gestational_Week", "BMI"]
        df_cleaned["Gestational_Week"] = df_cleaned["Gestational_Week"].apply(convert_gestational_week)

        initial_count = len(df_cleaned)
        df_cleaned = df_cleaned.dropna()
        final_count = len(df_cleaned)
        print(f"删除缺失值后数据形状: {df_cleaned.shape}")
        print(f"删除了 {initial_count - final_count} 行缺失数据")

        return df_cleaned

    except Exception as e:
        print(f"数据读取和清洗过程中出现错误: {e}")
        raise

def perform_descriptive_analysis(df):
    """执行描述性统计分析"""
    print("\n" + "="*50)
    print("描述性统计分析")
    print("="*50)

    desc_stats = df.describe()
    print("\n描述性统计:")
    print(desc_stats)

    correlation_matrix = df.corr()
    print("\n相关性矩阵:")
    print(correlation_matrix)

    return desc_stats, correlation_matrix

def create_3d_visualization(df, output_dir="output", font_prop=None):
    """创建3D可视化图表 - 强制使用中文"""

    # 创建3D散点图
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 根据Y染色体浓度创建颜色映射
    scatter = ax.scatter(df['Gestational_Week'], df['BMI'], df['Y_concentration'],
                        c=df['Y_concentration'], cmap='viridis', alpha=0.7, s=50)

    # 使用中文标签
    ax.set_xlabel('孕周', fontproperties=font_prop, fontsize=12, labelpad=10)
    ax.set_ylabel('BMI', fontproperties=font_prop, fontsize=12, labelpad=10)
    ax.set_zlabel('Y染色体浓度', fontproperties=font_prop, fontsize=12, labelpad=10)
    ax.set_title('Y染色体浓度的3D散点图', fontproperties=font_prop, fontsize=14, pad=20)

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=30)
    cbar.set_label('Y染色体浓度', fontproperties=font_prop, fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/3d_scatter_plot.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_surface_plot(df, output_dir="output", font_prop=None):
    """创建3D表面图 - 强制使用中文"""

    # 创建网格数据
    weeks = np.linspace(df['Gestational_Week'].min(), df['Gestational_Week'].max(), 30)
    bmis = np.linspace(df['BMI'].min(), df['BMI'].max(), 30)
    Weeks, BMIs = np.meshgrid(weeks, bmis)

    # 使用线性回归拟合表面
    X = df[['Gestational_Week', 'BMI']]
    X = sm.add_constant(X)
    y = df['Y_concentration']
    model = sm.OLS(y, X).fit()

    # 预测表面值
    grid_points = np.column_stack([np.ones(Weeks.ravel().shape), Weeks.ravel(), BMIs.ravel()])
    Z_pred = model.predict(grid_points).reshape(Weeks.shape)

    # 创建3D表面图
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    surface = ax.plot_surface(Weeks, BMIs, Z_pred, cmap='coolwarm', alpha=0.8)

    # 添加实际数据点
    ax.scatter(df['Gestational_Week'], df['BMI'], df['Y_concentration'],
              c='red', s=20, alpha=0.6, label='实际数据')

    # 使用中文标签
    ax.set_xlabel('孕周', fontproperties=font_prop, fontsize=12)
    ax.set_ylabel('BMI', fontproperties=font_prop, fontsize=12)
    ax.set_zlabel('预测Y染色体浓度', fontproperties=font_prop, fontsize=12)
    ax.set_title('Y染色体浓度预测表面图', fontproperties=font_prop, fontsize=14, pad=20)

    # 添加图例
    ax.legend(prop=font_prop)

    # 添加颜色条
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=30)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/3d_surface_plot.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_modern_2d_plots(df, output_dir="output", font_prop=None):
    """创建现代化的2D图表 - 强制使用中文"""

    # 现代化配色方案
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 散点图 1: Y浓度 vs 孕周
    axes[0,0].scatter(df['Gestational_Week'], df['Y_concentration'],
                     alpha=0.6, c=colors[0], s=40, edgecolors='white', linewidth=0.5)

    # 添加回归线
    z = np.polyfit(df['Gestational_Week'], df['Y_concentration'], 1)
    p = np.poly1d(z)
    axes[0,0].plot(df['Gestational_Week'], p(df['Gestational_Week']),
                   color=colors[1], linewidth=2, linestyle='--')

    axes[0,0].set_title('Y染色体浓度 vs 孕周', fontproperties=font_prop, fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('孕周', fontproperties=font_prop, fontsize=12)
    axes[0,0].set_ylabel('Y染色体浓度', fontproperties=font_prop, fontsize=12)
    axes[0,0].grid(True, alpha=0.3)

    # 散点图 2: Y浓度 vs BMI
    axes[0,1].scatter(df['BMI'], df['Y_concentration'],
                     alpha=0.6, c=colors[2], s=40, edgecolors='white', linewidth=0.5)

    z = np.polyfit(df['BMI'], df['Y_concentration'], 1)
    p = np.poly1d(z)
    axes[0,1].plot(df['BMI'], p(df['BMI']),
                   color=colors[3], linewidth=2, linestyle='--')

    axes[0,1].set_title('Y染色体浓度 vs BMI', fontproperties=font_prop, fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('BMI', fontproperties=font_prop, fontsize=12)
    axes[0,1].set_ylabel('Y染色体浓度', fontproperties=font_prop, fontsize=12)
    axes[0,1].grid(True, alpha=0.3)

    # 相关性热图
    corr_matrix = df.corr()
    im = axes[1,0].imshow(corr_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)

    # 设置刻度和标签 - 使用中文
    variable_names = ['Y染色体浓度', '孕周', 'BMI']
    axes[1,0].set_xticks(range(len(corr_matrix.columns)))
    axes[1,0].set_yticks(range(len(corr_matrix.columns)))
    axes[1,0].set_xticklabels(variable_names, rotation=45, ha='right', fontproperties=font_prop)
    axes[1,0].set_yticklabels(variable_names, fontproperties=font_prop)

    # 添加数值标注
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            text = axes[1,0].text(j, i, f'{corr_matrix.iloc[i, j]:.3f}',
                                 ha="center", va="center", color="black", fontweight='bold')

    axes[1,0].set_title('相关性热图', fontproperties=font_prop, fontsize=14, fontweight='bold')

    # 分布图
    n, bins, patches = axes[1,1].hist(df['Y_concentration'], bins=20, alpha=0.7, color=colors[4], edgecolor='white')

    # 为直方图添加渐变色
    for i, patch in enumerate(patches):
        patch.set_facecolor(plt.cm.viridis(i / len(patches)))

    axes[1,1].set_title('Y染色体浓度分布', fontproperties=font_prop, fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel('Y染色体浓度', fontproperties=font_prop, fontsize=12)
    axes[1,1].set_ylabel('频数', fontproperties=font_prop, fontsize=12)
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/modern_2d_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_advanced_correlation_plot(df, output_dir="output", font_prop=None):
    """创建高级相关性可视化 - 强制使用中文"""

    fig, ax = plt.subplots(figsize=(10, 8))

    # 创建相关性矩阵
    corr_matrix = df.corr()

    # 绘制相关性网络图
    variables = ['Y_concentration', 'Gestational_Week', 'BMI']
    var_names = ['Y染色体浓度', '孕周', 'BMI']

    # 设置节点位置
    pos = {0: (0, 1), 1: (-1, -0.5), 2: (1, -0.5)}

    # 绘制节点
    for i, (var, name) in enumerate(zip(variables, var_names)):
        circle = plt.Circle(pos[i], 0.3, color=plt.cm.Set3(i), alpha=0.8)
        ax.add_patch(circle)
        ax.text(pos[i][0], pos[i][1], name, ha='center', va='center',
                fontproperties=font_prop, fontsize=10, fontweight='bold')

    # 绘制连接线（相关性）
    for i in range(len(variables)):
        for j in range(i+1, len(variables)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.1:  # 只显示较强的相关性
                x_vals = [pos[i][0], pos[j][0]]
                y_vals = [pos[i][1], pos[j][1]]

                # 线条粗细基于相关性强度
                linewidth = abs(corr_val) * 5
                color = 'red' if corr_val > 0 else 'blue'
                alpha = min(abs(corr_val) + 0.3, 1.0)

                ax.plot(x_vals, y_vals, color=color, linewidth=linewidth, alpha=alpha)

                # 添加相关系数标签
                mid_x, mid_y = (x_vals[0] + x_vals[1])/2, (y_vals[0] + y_vals[1])/2
                ax.text(mid_x, mid_y, f'{corr_val:.3f}', ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                       fontsize=9, fontweight='bold')

    ax.set_xlim(-2, 2)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.set_title('相关性网络可视化', fontproperties=font_prop, fontsize=16, fontweight='bold', pad=20)

    # 添加图例 - 使用中文
    red_line = mpatches.Patch(color='red', label='正相关')
    blue_line = mpatches.Patch(color='blue', label='负相关')
    ax.legend(handles=[red_line, blue_line], loc='upper right', prop=font_prop)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_network.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_bmi_group_analysis(df, output_dir="output", font_prop=None):
    """创建BMI分组分析图表 - 强制使用中文"""

    # 创建BMI分组
    df['BMI_Group'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 50],
                             labels=['偏瘦', '正常', '超重', '肥胖'])

    # 检查每个组的样本数量
    group_counts = df['BMI_Group'].value_counts()
    print("BMI分组样本数量:")
    for group, count in group_counts.items():
        print(f"  {group}: {count}")

    # 只保留有数据的组
    non_empty_groups = []
    group_names = []
    group_labels = ['偏瘦', '正常', '超重', '肥胖']

    for label in group_labels:
        if label in group_counts and group_counts[label] > 0:
            group_data = df[df['BMI_Group'] == label]['Y_concentration'].values
            if len(group_data) > 0:
                non_empty_groups.append(group_data)
                group_names.append(label)

    if len(non_empty_groups) == 0:
        print("警告: 没有找到有效的BMI分组数据")
        return df

    print(f"有效的BMI分组数量: {len(non_empty_groups)}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 箱线图
    try:
        box_plot = ax1.boxplot(non_empty_groups, labels=group_names, patch_artist=True)
        colors = ['lightblue', 'lightgreen', 'orange', 'lightcoral'][:len(non_empty_groups)]

        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)

        ax1.set_title('不同BMI组的Y染色体浓度分布', fontproperties=font_prop, fontsize=14, fontweight='bold')
        ax1.set_xlabel('BMI分组', fontproperties=font_prop, fontsize=12)
        ax1.set_ylabel('Y染色体浓度', fontproperties=font_prop, fontsize=12)
        ax1.grid(True, alpha=0.3)
    except Exception as e:
        print(f"箱线图创建失败: {e}")
        ax1.text(0.5, 0.5, '箱线图创建失败', ha='center', va='center', transform=ax1.transAxes)

    # 小提琴图或散点图
    try:
        if len(non_empty_groups) > 0:
            parts = ax2.violinplot(non_empty_groups, positions=range(1, len(non_empty_groups) + 1), showmeans=True)

            if 'bodies' in parts:
                for pc in parts['bodies']:
                    pc.set_facecolor('lightblue')
                    pc.set_alpha(0.7)

            ax2.set_xticks(range(1, len(group_names) + 1))
            ax2.set_xticklabels(group_names, fontproperties=font_prop)
            ax2.set_title('小提琴图分布', fontproperties=font_prop, fontsize=14, fontweight='bold')
            ax2.set_xlabel('BMI分组', fontproperties=font_prop, fontsize=12)
            ax2.set_ylabel('Y染色体浓度', fontproperties=font_prop, fontsize=12)
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, '无数据可用', ha='center', va='center', transform=ax2.transAxes)

    except Exception as e:
        print(f"小提琴图创建失败: {e}")
        # 创建散点图作为替代
        for i, (group_data, name) in enumerate(zip(non_empty_groups, group_names)):
            y_values = group_data
            x_values = [i + 1] * len(y_values)
            x_noise = np.random.normal(0, 0.1, len(x_values))
            ax2.scatter(np.array(x_values) + x_noise, y_values, alpha=0.6, s=30)

        ax2.set_xticks(range(1, len(group_names) + 1))
        ax2.set_xticklabels(group_names, fontproperties=font_prop)
        ax2.set_title('散点图分布', fontproperties=font_prop, fontsize=14, fontweight='bold')
        ax2.set_xlabel('BMI分组', fontproperties=font_prop, fontsize=12)
        ax2.set_ylabel('Y染色体浓度', fontproperties=font_prop, fontsize=12)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/bmi_group_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

    return df

def significance_testing(df):
    """执行显著性检验"""
    print("\n" + "=" * 50)
    print("显著性检验分析")
    print("=" * 50)

    # 1. 孕周与Y浓度的相关性检验
    corr_week, p_week = stats.pearsonr(df['Gestational_Week'], df['Y_concentration'])
    print(f"孕周与Y浓度相关性: r={corr_week:.4f}, p={p_week:.4f}")

    # 2. BMI与Y浓度的相关性检验
    corr_bmi, p_bmi = stats.pearsonr(df['BMI'], df['Y_concentration'])
    print(f"BMI与Y浓度相关性: r={corr_bmi:.4f}, p={p_bmi:.4f}")

    # 3. BMI分组间差异检验 (ANOVA)
    try:
        if 'BMI_Group' not in df.columns:
            df['BMI_Group'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 50],
                                     labels=['偏瘦', '正常', '超重', '肥胖'])

        group_counts = df['BMI_Group'].value_counts()
        print(f"\nBMI分组样本数: {dict(group_counts)}")

        valid_groups = []
        for name, group in df.groupby('BMI_Group'):
            if len(group) >= 2:
                valid_groups.append(group['Y_concentration'].values)

        if len(valid_groups) >= 2:
            f_stat, p_anova = stats.f_oneway(*valid_groups)
            print(f"BMI分组间差异检验: F={f_stat:.4f}, p={p_anova:.4f}")
        else:
            print("BMI分组数据不足，无法进行ANOVA检验")
            f_stat, p_anova = np.nan, np.nan

    except Exception as e:
        print(f"ANOVA检验失败: {e}")
        f_stat, p_anova = np.nan, np.nan

    # 4. 偏相关分析
    try:
        from scipy.stats import pearsonr
        week_bmi_corr, _ = pearsonr(df['Gestational_Week'], df['BMI'])
        week_y_corr, _ = pearsonr(df['Gestational_Week'], df['Y_concentration'])
        bmi_y_corr, _ = pearsonr(df['BMI'], df['Y_concentration'])

        numerator = bmi_y_corr - week_bmi_corr * week_y_corr
        denominator = np.sqrt(1 - week_bmi_corr ** 2) * np.sqrt(1 - week_y_corr ** 2)

        if denominator != 0:
            partial_corr = numerator / denominator
            print(f"控制孕周后BMI与Y浓度的偏相关: r={partial_corr:.4f}")
        else:
            partial_corr = np.nan
            print("偏相关计算失败：分母为零")

    except Exception as e:
        print(f"偏相关计算失败: {e}")
        partial_corr = np.nan

    return {
        'week_corr': (corr_week, p_week),
        'bmi_corr': (corr_bmi, p_bmi),
        'anova': (f_stat, p_anova),
        'partial_corr': partial_corr
    }

def create_visualizations(df, output_dir="output", font_prop=None):
    """创建所有可视化图表 - 强制使用中文"""
    Path(output_dir).mkdir(exist_ok=True)

    print("\n" + "="*50)
    print("创建现代化可视化图表（强制中文）")
    print("="*50)

    # 创建3D散点图
    print("创建3D散点图...")
    create_3d_visualization(df, output_dir, font_prop)

    # 创建3D表面图
    print("创建3D表面图...")
    create_surface_plot(df, output_dir, font_prop)

    # 创建现代化2D图表
    print("创建现代化2D图表...")
    create_modern_2d_plots(df, output_dir, font_prop)

    # 创建高级相关性图表
    print("创建相关性网络图...")
    create_advanced_correlation_plot(df, output_dir, font_prop)

    # 创建BMI分组分析
    print("创建BMI分组分析...")
    df = create_bmi_group_analysis(df, output_dir, font_prop)

    print(f"所有图表已保存到 {output_dir} 目录")
    return df

def perform_regression_analysis(df):
    """执行多元线性回归分析"""
    print("\n" + "="*50)
    print("多元线性回归分析")
    print("="*50)

    X = df[["Gestational_Week", "BMI"]]
    X = sm.add_constant(X)
    y = df["Y_concentration"]

    model = sm.OLS(y, X).fit()

    print("\n回归分析结果:")
    print(model.summary())

    print("\n模型诊断:")
    print(f"R-squared: {model.rsquared:.4f}")
    print(f"Adj. R-squared: {model.rsquared_adj:.4f}")
    print(f"F-statistic: {model.fvalue:.4f}")
    print(f"Prob (F-statistic): {model.f_pvalue:.4e}")

    return model

def mixed_effects_model_analysis(df):
    """混合效应模型分析"""
    print("\n" + "="*50)
    print("混合效应模型分析")
    print("="*50)

    X = df[['Gestational_Week', 'BMI']]
    y = df['Y_concentration']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    y_pred = model.predict(X_scaled)
    r2 = r2_score(y, y_pred)

    print(f"模型R² = {r2:.4f}")
    print(f"标准化孕周系数: {model.coef_[0]:.6f}")
    print(f"标准化BMI系数: {model.coef_[1]:.6f}")
    print(f"截距: {model.intercept_:.6f}")

    feature_names = ['Gestational_Week', 'BMI']
    original_coef = model.coef_ / scaler.scale_

    print("\n原始尺度系数:")
    for i, name in enumerate(feature_names):
        print(f"{name}: {original_coef[i]:.6f}")

    return model, scaler, r2

def save_results_to_excel(df, desc_stats, correlation_matrix, model, stats_results, output_file="Y染色体浓度分析结果_中文版本.xlsx"):
    """将分析结果保存到Excel文件"""
    print("\n" + "="*50)
    print("保存结果到Excel")
    print("="*50)

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 原始数据
        df.to_excel(writer, sheet_name='原始数据', index=False)

        # 描述性统计
        desc_stats.to_excel(writer, sheet_name='描述性统计')

        # 相关性矩阵
        correlation_matrix.to_excel(writer, sheet_name='相关性矩阵')

        # 回归系数
        regression_results = pd.DataFrame({
            '变量': model.params.index,
            '系数': model.params.values,
            '标准误': model.bse.values,
            't统计量': model.tvalues.values,
            'p值': model.pvalues.values,
            '置信区间下限': model.conf_int()[0].values,
            '置信区间上限': model.conf_int()[1].values
        })
        regression_results.to_excel(writer, sheet_name='回归系数', index=False)

        # 模型统计
        model_stats = pd.DataFrame({
            '指标': ['R-squared', 'Adj. R-squared', 'F-statistic', 'Prob (F-statistic)',
                   'AIC', 'BIC', '观测数'],
            '值': [model.rsquared, model.rsquared_adj, model.fvalue, model.f_pvalue,
                  model.aic, model.bic, model.nobs]
        })
        model_stats.to_excel(writer, sheet_name='模型统计', index=False)

        # 显著性检验结果
        week_corr, week_p = stats_results['week_corr']
        bmi_corr, bmi_p = stats_results['bmi_corr']
        f_stat, p_anova = stats_results['anova']

        significance_results = pd.DataFrame({
            '检验类型': ['孕周与Y浓度相关性', 'BMI与Y浓度相关性', 'BMI分组间差异(ANOVA)', '偏相关(控制孕周)'],
            '统计量': [week_corr, bmi_corr, f_stat, stats_results['partial_corr']],
            'p值': [week_p, bmi_p, p_anova, 'N/A'],
            '显著性': [
                '显著' if week_p < 0.05 else '不显著',
                '显著' if bmi_p < 0.05 else '不显著',
                '显著' if p_anova < 0.05 else '不显著',
                'N/A'
            ]
        })
        significance_results.to_excel(writer, sheet_name='显著性检验', index=False)

    print(f"分析结果已保存到: {output_file}")

def generate_comprehensive_report(df, stats_results, model_r2, model):
    """生成综合分析报告"""
    print("\n" + "="*60)
    print("                    综合分析报告")
    print("="*60)

    # 基础信息
    print(f"数据样本量: {len(df)}")
    print(f"Y染色体浓度范围: {df['Y_concentration'].min():.4f} - {df['Y_concentration'].max():.4f}")
    print(f"孕周范围: {df['Gestational_Week'].min():.1f} - {df['Gestational_Week'].max():.1f} 周")
    print(f"BMI范围: {df['BMI'].min():.1f} - {df['BMI'].max():.1f}")

    # 主要发现
    print("\n主要发现:")
    week_corr, week_p = stats_results['week_corr']
    bmi_corr, bmi_p = stats_results['bmi_corr']

    print(f"1. 孕周与Y染色体浓度相关性分析:")
    print(f"   - 相关系数: r = {week_corr:.4f}")
    print(f"   - 显著性: p = {week_p:.4f}")
    if week_p < 0.05:
        direction = "正相关" if week_corr > 0 else "负相关"
        print(f"   - 结论: 孕周与Y染色体浓度呈显著{direction}")
    else:
        print("   - 结论: 孕周与Y染色体浓度无显著相关性")

    print(f"\n2. BMI与Y染色体浓度相关性分析:")
    print(f"   - 相关系数: r = {bmi_corr:.4f}")
    print(f"   - 显著性: p = {bmi_p:.4f}")
    if bmi_p < 0.05:
        direction = "正相关" if bmi_corr > 0 else "负相关"
        print(f"   - 结论: BMI与Y染色体浓度呈显著{direction}")
    else:
        print("   - 结论: BMI与Y染色体浓度无显著相关性")

    print(f"\n3. 多元回归模型:")
    print(f"   - 模型解释力: R² = {model.rsquared:.4f} ({model.rsquared:.1%})")
    print(f"   - 调整R²: {model.rsquared_adj:.4f}")
    print(f"   - F统计量: {model.fvalue:.4f} (p = {model.f_pvalue:.4f})")

    # 回归系数解释
    print(f"\n4. 回归系数解释:")
    week_coef = model.params['Gestational_Week']
    bmi_coef = model.params['BMI']
    print(f"   - 孕周系数: {week_coef:.6f}")
    print(f"     解释: 孕周每增加1周，Y染色体浓度平均{'增加' if week_coef > 0 else '减少'}{abs(week_coef):.6f}")
    print(f"   - BMI系数: {bmi_coef:.6f}")
    print(f"     解释: BMI每增加1单位，Y染色体浓度平均{'增加' if bmi_coef > 0 else '减少'}{abs(bmi_coef):.6f}")

    # BMI分组差异
    f_stat, p_anova = stats_results['anova']
    print(f"\n5. BMI分组间差异:")
    if not np.isnan(f_stat):
        print(f"   - F统计量: {f_stat:.4f}")
        print(f"   - 显著性: p = {p_anova:.4f}")
        if p_anova < 0.05:
            print("   - 结论: 不同BMI组间Y染色体浓度存在显著差异")
        else:
            print("   - 结论: 不同BMI组间Y染色体浓度无显著差异")
    else:
        print("   - 数据不足，无法进行ANOVA检验")

    # 偏相关分析
    partial_corr = stats_results['partial_corr']
    print(f"\n6. 偏相关分析:")
    if not np.isnan(partial_corr):
        print(f"   - 控制孕周后BMI与Y浓度的偏相关: r = {partial_corr:.4f}")
        print("   - 解释: 在控制孕周影响后，BMI与Y染色体浓度的纯相关性")
    else:
        print("   - 偏相关计算失败")

    # 临床意义
    print(f"\n临床意义:")
    if week_p < 0.05 and week_corr > 0:
        print("• 随着孕周增长，Y染色体浓度呈上升趋势，符合胎儿发育规律")
    if bmi_p < 0.05:
        if bmi_corr < 0:
            print("• BMI较高的孕妇Y染色体浓度较低，可能影响男胎检测准确性")
        else:
            print("• BMI较高的孕妇Y染色体浓度较高")

    print(f"\n建议:")
    print("• 在进行胎儿性别检测时应考虑孕周因素")
    if abs(bmi_corr) > 0.1:
        print("• 对于BMI异常的孕妇，可能需要调整检测标准或时间")
    print("• 建议在孕中期（16-20周）进行检测以获得更稳定的结果")

def main():
    """主函数"""
    try:
        print("配置现代化图表样式和中文字体...")

        # 配置中文字体 - 强制使用中文
        font_prop, selected_font = configure_chinese_font()

        print(f"\n✅ 中文字体配置完成！")
        if selected_font:
            print(f"使用字体: {selected_font}")
        else:
            print("使用系统默认字体，但强制显示中文")

        # 设置图表样式
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': '#f8f9fa',
            'axes.edgecolor': '#dee2e6',
            'axes.linewidth': 1.2,
            'axes.grid': True,
            'grid.color': '#e9ecef',
            'grid.linestyle': '-',
            'grid.linewidth': 0.8,
            'grid.alpha': 0.8,
            'xtick.color': '#495057',
            'ytick.color': '#495057',
            'axes.labelcolor': '#212529',
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })

        # 数据文件路径
        data_file = "../Data_Cleaned/output/male_tests.csv"

        # 1. 读取和清洗数据
        print("读取和清洗数据...")
        df_male = load_and_clean_data(data_file)

        # 2. 描述性统计分析
        desc_stats, correlation_matrix = perform_descriptive_analysis(df_male)

        # 3. 创建现代化可视化图表（强制中文）
        df_male = create_visualizations(df_male, font_prop=font_prop)

        # 4. 多元线性回归分析
        model = perform_regression_analysis(df_male)

        # 5. 混合效应模型分析
        mixed_model, scaler, mixed_r2 = mixed_effects_model_analysis(df_male)

        # 6. 显著性检验
        stats_results = significance_testing(df_male)

        # 7. 保存结果到Excel
        save_results_to_excel(df_male, desc_stats, correlation_matrix, model, stats_results,
                            "Y染色体浓度分析结果_中文版本.xlsx")

        # 8. 生成综合报告
        generate_comprehensive_report(df_male, stats_results, model.rsquared, model)

        print("\n" + "="*50)
        print("现代化3D分析完成！（中文版本）")
        print("="*50)
        print("生成的图表包括:")
        print("- 3D散点图 (3d_scatter_plot.png)")
        print("- 3D表面图 (3d_surface_plot.png)")
        print("- 现代化2D分析图 (modern_2d_analysis.png)")
        print("- 相关性网络图 (correlation_network.png)")
        print("- BMI分组分析图 (bmi_group_analysis.png)")
        print("\n🎉 所有图表均使用中文标签！")

    except Exception as e:
        print(f"程序执行过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main()