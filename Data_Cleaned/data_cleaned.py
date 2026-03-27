import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path
from typing import Dict, Optional

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NIPTCleaner:
    def __init__(self):
        # 基于用户实际数据的列名映射
        self.column_mapping = {
            "序号": "sample_id",
            "孕妇代码": "mother_id",
            "年龄": "age",
            "身高": "height",
            "体重": "weight",
            "末次月经": "lmp_date",
            "IVF妊娠": "ivf_method",
            "检测日期": "test_time",
            "检测抽血次数": "draw_index",
            "检测孕周": "gestation_str",
            "孕妇BMI": "bmi",
            "原始读段数": "reads_total",
            "在参考基因组上比对的比例": "reads_align_ratio",
            "重复读段的比例": "reads_dup_ratio",
            "唯一比对的读段数": "reads_unique",
            "唯一比对的读段数  ": "reads_unique",  # 有空格的版本
            "GC含量": "gc_total",
            "13号染色体的Z值": "z_13",
            "18号染色体的Z值": "z_18",
            "21号染色体的Z值": "z_21",
            "X染色体的Z值": "z_x",
            "Y染色体的Z值": "z_y",
            "Y染色体浓度": "y_frac",
            "X染色体浓度": "x_frac",
            "13号染色体的GC含量": "gc_13",
            "18号染色体的GC含量": "gc_18",
            "21号染色体的GC含量": "gc_21",
            "被过滤掉读段数的比例": "reads_filtered_ratio",
            "染色体的非整倍体": "aneuploidy_1321",
            "怀孕次数": "gravidity",
            "生产次数": "parity",
            "胎儿是否健康": "fetus_health_after_birth"
        }

        self.output_dir = Path("./output")
        self.output_dir.mkdir(exist_ok=True)

    def load_data(self, file_path: str = "./附件.xlsx") -> pd.DataFrame:
        """加载Excel数据"""
        logger.info(f"加载数据文件: {file_path}")

        try:
            excel_file = pd.ExcelFile(file_path)
            all_frames = []

            for sheet_name in excel_file.sheet_names:
                logger.info(f"处理工作表: {sheet_name}")
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                df['source_sheet'] = sheet_name

                # 显示原始列名用于调试
                logger.debug(f"原始列名: {list(df.columns)}")

                all_frames.append(df)

            combined_df = pd.concat(all_frames, ignore_index=True)
            logger.info(f"成功加载 {len(combined_df)} 行数据")

            return combined_df

        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise

    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化列名"""
        logger.info("开始标准化列名")

        # 创建重命名映射
        rename_dict = {}
        unmapped_columns = []

        for col in df.columns:
            col_clean = str(col).strip()
            if col_clean in self.column_mapping:
                rename_dict[col] = self.column_mapping[col_clean]
            else:
                # 保持原列名，但记录未映射的列
                rename_dict[col] = col
                if col != 'source_sheet':  # 排除我们添加的列
                    unmapped_columns.append(col_clean)

        if unmapped_columns:
            logger.warning(f"未映射的列: {unmapped_columns}")

        # 重命名列
        df_renamed = df.rename(columns=rename_dict)

        # 显示重命名结果
        mapped_count = len([k for k, v in rename_dict.items() if k != v and k != 'source_sheet'])
        logger.info(f"成功重命名 {mapped_count} 个列")
        logger.debug(f"重命名后的列: {list(df_renamed.columns)}")

        return df_renamed

    def clean_datetime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗日期时间列"""
        logger.info("清洗日期时间数据")

        # 处理末次月经日期
        if 'lmp_date' in df.columns:
            df['lmp_date'] = pd.to_datetime(df['lmp_date'], errors='coerce')
            logger.debug(f"末次月经日期有效值: {df['lmp_date'].notna().sum()}")

        # 处理检测日期
        if 'test_time' in df.columns:
            df['test_time'] = pd.to_datetime(df['test_time'], errors='coerce')
            logger.debug(f"检测日期有效值: {df['test_time'].notna().sum()}")

        return df

    def process_gestation_weeks(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理孕周数据"""
        logger.info("处理孕周数据")

        def parse_gestation(gestation_str):
            """解析孕周字符串，如 '11w+6' -> 11.857"""
            if pd.isna(gestation_str):
                return np.nan

            gestation_str = str(gestation_str).strip()

            # 匹配 "数字w+数字" 或 "数字w" 格式
            match = re.match(r'(\d+)w?(?:\+(\d+))?', gestation_str)
            if match:
                weeks = int(match.group(1))
                days = int(match.group(2)) if match.group(2) else 0
                return weeks + days / 7.0

            return np.nan

        # 解析孕周字符串
        if 'gestation_str' in df.columns:
            df['gestation_weeks'] = df['gestation_str'].apply(parse_gestation)
            valid_weeks = df['gestation_weeks'].notna().sum()
            logger.debug(f"成功解析 {valid_weeks} 个孕周值")

        # 基于日期计算孕周（如果需要）
        if 'test_time' in df.columns and 'lmp_date' in df.columns:
            both_available = df['test_time'].notna() & df['lmp_date'].notna()
            if both_available.any():
                calculated_weeks = (df['test_time'] - df['lmp_date']).dt.days / 7.0

                # 只有当解析的孕周为空时才用计算值填充
                if 'gestation_weeks' not in df.columns:
                    df['gestation_weeks'] = np.nan

                missing_mask = df['gestation_weeks'].isna() & both_available
                df.loc[missing_mask, 'gestation_weeks'] = calculated_weeks[missing_mask]

                filled_count = missing_mask.sum()
                if filled_count > 0:
                    logger.debug(f"用计算值填充了 {filled_count} 个孕周")

        return df

    def process_physical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理身体测量数据"""
        logger.info("处理身体测量数据")

        # 标准化身高（转换为米）
        if 'height' in df.columns:
            def normalize_height(height):
                if pd.isna(height):
                    return np.nan
                try:
                    h = float(height)
                    return h / 100 if h > 3 else h  # 假设大于3的是厘米
                except:
                    return np.nan

            df['height_m'] = df['height'].apply(normalize_height)

        # 重新计算BMI（如果需要）
        if 'height' in df.columns and 'weight' in df.columns:
            def calculate_bmi(row):
                if not pd.isna(row.get('bmi')):
                    return row['bmi']  # 如果已有BMI，保留

                height_m = row.get('height_m')
                weight = row.get('weight')

                if pd.isna(height_m) or pd.isna(weight):
                    return np.nan

                try:
                    return float(weight) / (float(height_m) ** 2)
                except:
                    return np.nan

            # 只有当BMI列不存在或有缺失值时才计算
            if 'bmi' not in df.columns:
                df['bmi'] = df.apply(calculate_bmi, axis=1)
            else:
                missing_bmi = df['bmi'].isna()
                if missing_bmi.any():
                    df.loc[missing_bmi, 'bmi'] = df[missing_bmi].apply(calculate_bmi, axis=1)

        return df

    def process_laboratory_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理实验室数据"""
        logger.info("处理实验室数据")

        # 转换百分比格式的列
        percentage_columns = [
            'reads_align_ratio', 'reads_dup_ratio', 'reads_filtered_ratio',
            'gc_total', 'gc_13', 'gc_18', 'gc_21'
        ]

        def convert_percentage(value):
            """转换百分比到小数"""
            if pd.isna(value):
                return np.nan

            if isinstance(value, str) and '%' in value:
                try:
                    return float(value.replace('%', '')) / 100
                except:
                    return np.nan

            try:
                val = float(value)
                # 如果值在1-100之间，认为是百分比
                if 1 < val <= 100:
                    return val / 100
                return val
            except:
                return np.nan

        for col in percentage_columns:
            if col in df.columns:
                df[col] = df[col].apply(convert_percentage)
                logger.debug(f"转换百分比列: {col}")

        # 处理Y染色体浓度
        if 'y_frac' in df.columns:
            df['y_frac'] = pd.to_numeric(df['y_frac'], errors='coerce').clip(0, 1)

        # 推断胎儿性别
        def infer_fetal_sex(row):
            """基于Y染色体数据推断性别"""
            y_frac = row.get('y_frac')
            z_y = row.get('z_y')

            # 如果Y染色体相关指标都为空，推断为女性
            if pd.isna(y_frac) and pd.isna(z_y):
                return 'F'
            else:
                return 'M'

        df['fetal_sex'] = df.apply(infer_fetal_sex, axis=1)

        male_count = (df['fetal_sex'] == 'M').sum()
        female_count = (df['fetal_sex'] == 'F').sum()
        logger.info(f"性别推断 - 男性: {male_count}, 女性: {female_count}")

        return df

    def apply_quality_control(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用质量控制"""
        logger.info("应用质量控制")

        qc_conditions = []

        # GC含量检查
        if 'gc_total' in df.columns:
            gc_ok = df['gc_total'].between(0.40, 0.60) | df['gc_total'].isna()
            qc_conditions.append(gc_ok)
            logger.debug(f"GC含量检查通过: {gc_ok.sum()}/{len(df)}")

        # 过滤读段比例检查
        if 'reads_filtered_ratio' in df.columns:
            filter_ok = (df['reads_filtered_ratio'] <= 0.60) | df['reads_filtered_ratio'].isna()
            qc_conditions.append(filter_ok)
            logger.debug(f"过滤比例检查通过: {filter_ok.sum()}/{len(df)}")

        # 综合质控结果
        if qc_conditions:
            df['qc_pass'] = pd.concat(qc_conditions, axis=1).all(axis=1)
        else:
            df['qc_pass'] = True

        qc_pass_count = df['qc_pass'].sum()
        logger.info(f"质量控制通过: {qc_pass_count}/{len(df)} 个样本")

        return df

    def analyze_male_threshold(self, df: pd.DataFrame) -> pd.DataFrame:
        """分析男胎Y染色体阈值"""
        logger.info("分析男胎Y染色体阈值")

        # 筛选男胎数据
        male_df = df[df['fetal_sex'] == 'M'].copy()

        if len(male_df) == 0:
            logger.warning("没有找到男胎数据")
            return pd.DataFrame()

        # 按孕妇和时间排序
        male_df = male_df.sort_values(['mother_id', 'test_time'])

        # 判断是否达到4%阈值
        if 'y_frac' in male_df.columns:
            male_df['reach_4pct'] = (male_df['y_frac'] >= 0.04) & (male_df['qc_pass'])

            # 找到每个孕妇首次达到阈值的记录
            threshold_reached = male_df[male_df['reach_4pct']]

            if not threshold_reached.empty:
                earliest_threshold = (
                    threshold_reached
                    .groupby('mother_id', as_index=False)
                    .agg({
                        'test_time': 'min',
                        'gestation_weeks': 'min'
                    })
                    .rename(columns={
                        'test_time': 'first_reach_time',
                        'gestation_weeks': 'first_reach_gw'
                    })
                )

                logger.info(f"达到4%阈值的孕妇数: {len(earliest_threshold)}")
                return earliest_threshold

        return pd.DataFrame()

    def save_results(self, df: pd.DataFrame, earliest_threshold: pd.DataFrame):
        """保存处理结果"""
        logger.info("保存处理结果")

        # 保存完整清洗数据
        output_file = self.output_dir / "nipt_cleaned.csv"
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"保存完整数据: {output_file} ({len(df)} 行)")

        # 保存男胎数据
        male_df = df[df['fetal_sex'] == 'M']
        if not male_df.empty:
            male_file = self.output_dir / "male_tests.csv"
            male_df.to_csv(male_file, index=False, encoding='utf-8-sig')
            logger.info(f"保存男胎数据: {male_file} ({len(male_df)} 行)")

        # 保存女胎数据
        female_df = df[df['fetal_sex'] == 'F']
        if not female_df.empty:
            female_file = self.output_dir / "female_tests.csv"
            female_df.to_csv(female_file, index=False, encoding='utf-8-sig')
            logger.info(f"保存女胎数据: {female_file} ({len(female_df)} 行)")

        # 保存阈值分析结果
        if not earliest_threshold.empty:
            threshold_file = self.output_dir / "earliest_male_threshold.csv"
            earliest_threshold.to_csv(threshold_file, index=False, encoding='utf-8-sig')
            logger.info(f"保存阈值分析: {threshold_file} ({len(earliest_threshold)} 行)")

    def process_all(self, file_path: str = "./附件.xlsx") -> Dict:
        """完整处理流程"""
        logger.info("开始NIPT数据完整处理流程")

        try:
            # 1. 加载数据
            df = self.load_data(file_path)

            # 2. 标准化列名
            df = self.standardize_columns(df)

            # 3. 清洗日期时间
            df = self.clean_datetime_columns(df)

            # 4. 处理孕周数据
            df = self.process_gestation_weeks(df)

            # 5. 处理身体测量数据
            df = self.process_physical_data(df)

            # 6. 处理实验室数据
            df = self.process_laboratory_data(df)

            # 7. 应用质量控制
            df = self.apply_quality_control(df)

            # 8. 分析男胎阈值
            earliest_threshold = self.analyze_male_threshold(df)

            # 9. 保存结果
            self.save_results(df, earliest_threshold)

            # 统计信息
            stats = {
                'total_samples': len(df),
                'male_samples': (df['fetal_sex'] == 'M').sum(),
                'female_samples': (df['fetal_sex'] == 'F').sum(),
                'qc_pass_samples': df['qc_pass'].sum(),
                'mothers_reach_threshold': len(earliest_threshold)
            }

            logger.info("数据处理完成")
            return stats

        except Exception as e:
            logger.error(f"处理失败: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """主函数"""
    cleaner = NIPTCleaner()

    try:
        stats = cleaner.process_all()

        print("\n" + "=" * 60)
        print("NIPT数据处理完成！")
        print("=" * 60)
        print(f"总样本数: {stats['total_samples']}")
        print(f"男胎样本数: {stats['male_samples']}")
        print(f"女胎样本数: {stats['female_samples']}")
        print(f"质控通过样本数: {stats['qc_pass_samples']}")
        print(f"达到Y染色体阈值的孕妇数: {stats['mothers_reach_threshold']}")
        print(f"结果已保存到: ./output/")

    except Exception as e:
        print(f"处理失败: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())