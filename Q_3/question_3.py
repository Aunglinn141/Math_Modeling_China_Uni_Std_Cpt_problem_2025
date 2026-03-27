import os
import re
import json
import warnings
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Union, Any
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import seaborn as sns

# Optional dependencies
try:
    import torch
    import torch.nn as nn
    from torch.optim import Adam

    TORCH_AVAILABLE = True
except ImportError:
    print("[WARNING] PyTorch not available. Deep learning models will be disabled.")
    TORCH_AVAILABLE = False

try:
    from lifelines import CoxPHFitter

    SURVIVAL_AVAILABLE = True
except ImportError:
    print("[WARNING] lifelines not available. Survival analysis will be disabled.")
    SURVIVAL_AVAILABLE = False


# =============================================================================
# Configuration and Constants
# =============================================================================

@dataclass
class NIPTConfig:
    """Configuration class for NIPT analysis parameters."""

    # Data file paths
    data_candidates: List[str] = field(default_factory=lambda: [
        "../Data_Cleaned/output/nipt_cleaned.csv"
    ])
    male_tests_optional: List[str] = field(default_factory=lambda: [
        "../Data_Cleaned/output/male_tests.csv"
    ])

    # Column names
    mother_id: str = "mother_id"
    week_col: str = "gestation_weeks"
    bmi_col: str = "bmi"
    y_frac_col: str = "y_frac"
    gc_col: str = "gc_total"
    mapping_rate: str = "reads_align"
    filtered_ratio: str = "reads_filtered"
    height_col: str = "height"
    weight_col: str = "weight"
    age_col: str = "maternal_age"
    ivf_col: str = "ivf_method"
    parity_col: str = "parity"
    gestational_col: str = "gestational_age_group"
    # Add these fields to NIPTConfig
    threshold_range: Tuple[float, float] = (0.025, 0.055)
    threshold_step: float = 0.005
    use_bayesian_threshold: bool = True
    threshold_prior_alpha: float = 2.0
    threshold_prior_beta: float = 8.0

    # Feature categories
    continuous_features: List[str] = field(default_factory=lambda: [
        "bmi", "height", "weight", "age", "gc_total", "reads_align_ratio", "gestation_weeks"
    ])
    categorical_features: List[str] = field(default_factory=lambda: [
        "ivf_method", "parity", "gestational_age_group"
    ])

    # Analysis thresholds
    y_frac_threshold: float = 0.04
    sensitivity_thresholds: List[float] = field(default_factory=lambda: [0.035, 0.04, 0.045])

    # Quality control parameters
    qc_params: Dict[str, float] = field(default_factory=lambda: {
        "gc_min": 0.40, "gc_max": 0.60,
        "map_min": 0.85, "filt_max": 0.60
    })

    # Output directories
    output_dir: str = "./nipt_enhanced_outputs"

    # Model configuration
    model_config: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "logistic_sklearn": {"enabled": True, "weight": 1.0},
        "random_forest": {"enabled": True, "weight": 0.8},
        "torch_neural": {"enabled": TORCH_AVAILABLE, "weight": 1.2},
        "survival_cox": {"enabled": SURVIVAL_AVAILABLE, "weight": 1.0}
    })

    # Random seed for reproducibility
    random_seed: int = 2025


# =============================================================================
# Bootstrap Analysis
# =============================================================================

class BootstrapAnalyzer:
    """Bootstrap analysis for uncertainty quantification."""

    def __init__(self, config: NIPTConfig, n_bootstrap: int = 1000):
        self.config = config
        self.n_bootstrap = n_bootstrap

    def bootstrap_reach_rate(self, df: pd.DataFrame) -> Dict[str, float]:
        """Bootstrap analysis for reach rate confidence intervals."""
        bootstrap_rates = []
        n_samples = len(df)

        for _ in range(self.n_bootstrap):
            # Sample with replacement
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            bootstrap_sample = df.iloc[bootstrap_indices]

            # Calculate reach rate for bootstrap sample
            reach_rate = bootstrap_sample['reach'].mean()
            bootstrap_rates.append(reach_rate)

        bootstrap_rates = np.array(bootstrap_rates)

        return {
            'mean': np.mean(bootstrap_rates),
            'std': np.std(bootstrap_rates),
            'ci_lower': np.percentile(bootstrap_rates, 2.5),
            'ci_upper': np.percentile(bootstrap_rates, 97.5),
            'bootstrap_samples': bootstrap_rates
        }

# =============================================================================
# Data Models
# =============================================================================

@dataclass
class ModelResult:
    """Container for model training and prediction results."""
    model_name: str
    model_object: Any
    feature_importance: Dict[str, float]
    cross_val_score: float
    predicted_weeks: np.ndarray
    risk_scores: np.ndarray
    confidence_intervals: Optional[np.ndarray] = None
    extra_metrics: Optional[Dict[str, float]] = None


@dataclass
class PatientRecommendation:
    """Container for patient-specific recommendations."""
    patient_id: str
    optimal_week: float
    risk_score: float
    risk_level: str
    recommendation: str
    confidence_interval: Tuple[float, float]
    model_predictions: Dict[str, float]


# =============================================================================
# Utility Functions
# =============================================================================

class NIPTUtils:
    """Utility functions for NIPT analysis."""

    @staticmethod
    def setup_chinese_font() -> fm.FontProperties:
        """Set up Chinese font for matplotlib."""
        try:
            return fm.FontProperties(family="SimHei")
        except:
            print("[WARNING] Chinese font not available, using default font.")
            return fm.FontProperties()

    @staticmethod
    def parse_gestational_week(week_input: Any) -> float:
        """Parse gestational week from various formats (e.g., '15+3' -> 15.43)."""
        if pd.isna(week_input):
            return np.nan

        if isinstance(week_input, (int, float)):
            return float(week_input)

        week_str = str(week_input).strip()

        # Handle format like "15+3" (15 weeks + 3 days)
        match = re.match(r"^(\d+)\s*\+\s*(\d+)$", week_str)
        if match:
            weeks = int(match.group(1))
            days = int(match.group(2))
            return weeks + (days / 7.0)

        # Handle decimal format
        week_str = week_str.replace(",", ".")
        try:
            return float(week_str)
        except ValueError:
            return np.nan

    @staticmethod
    def create_output_directories(base_dir: str) -> Dict[str, str]:
        """Create all necessary output directories."""
        directories = {
            "main": base_dir,
            "plots": os.path.join(base_dir, "plots"),
            "models": os.path.join(base_dir, "models"),
            "reports": os.path.join(base_dir, "reports"),
            "bootstrap": os.path.join(base_dir, "bootstrap")
        }

        for dir_path in directories.values():
            os.makedirs(dir_path, exist_ok=True)

        return directories


# =============================================================================
# Data Processing
# =============================================================================

class NIPTDataProcessor:
    """Handles data loading, cleaning, and feature engineering."""

    def __init__(self, config: NIPTConfig):
        self.config = config
        self.utils = NIPTUtils()

    def load_data(self) -> pd.DataFrame:
        """Load and combine NIPT data from multiple sources."""
        # Load primary dataset
        main_df = None
        for path in self.config.data_candidates:
            if os.path.exists(path):
                main_df = pd.read_csv(path)
                print(f"[INFO] Loaded primary data from: {path}")
                break

        if main_df is None:
            raise FileNotFoundError("No primary data file found. Check data_candidates paths.")

        # Load and merge additional datasets
        for path in self.config.male_tests_optional:
            if os.path.exists(path):
                additional_df = pd.read_csv(path)
                main_df = pd.concat([main_df, additional_df], ignore_index=True)
                main_df = main_df.drop_duplicates()
                print(f"[INFO] Merged additional data from: {path}")

        return main_df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform comprehensive feature engineering."""
        df = df.copy()

        # Parse gestational weeks
        if self.config.week_col in df.columns:
            df[self.config.week_col] = df[self.config.week_col].apply(
                self.utils.parse_gestational_week
            )

        # Calculate BMI if missing but height/weight available
        if (self.config.height_col in df.columns and
                self.config.weight_col in df.columns):

            missing_bmi = (df[self.config.bmi_col].isna()
                           if self.config.bmi_col in df.columns
                           else pd.Series([True] * len(df)))

            if missing_bmi.any():
                height_m = df[self.config.height_col] / 100.0  # cm to meters
                computed_bmi = df[self.config.weight_col] / (height_m ** 2)

                if self.config.bmi_col not in df.columns:
                    df[self.config.bmi_col] = computed_bmi
                else:
                    df.loc[missing_bmi, self.config.bmi_col] = computed_bmi.loc[missing_bmi]

        # Create categorical features
        self._create_categorical_features(df)

        # Create quality score
        self._create_quality_score(df)

        return df

    def _create_categorical_features(self, df: pd.DataFrame) -> None:
        """Create categorical features for analysis."""
        # Age groups
        if self.config.age_col in df.columns:
            df['age_group'] = pd.cut(
                df[self.config.age_col],
                bins=[0, 25, 30, 35, 100],
                labels=['<25', '25-30', '30-35', '35+']
            )

        # BMI categories
        if self.config.bmi_col in df.columns:
            df['bmi_category'] = pd.cut(
                df[self.config.bmi_col],
                bins=[0, 18.5, 24.9, 29.9, 100],
                labels=['Underweight', 'Normal', 'Overweight', 'Obese']
            )

        # Pregnancy stages
        if self.config.week_col in df.columns:
            df['pregnancy_stage'] = pd.cut(
                df[self.config.week_col],
                bins=[0, 12, 27, 100],
                labels=['Early', 'Mid', 'Late']
            )

    def _create_quality_score(self, df: pd.DataFrame) -> None:
        """Create composite sequencing quality score."""
        quality_components = []

        # GC content score (optimal around 0.5)
        if self.config.gc_col in df.columns:
            gc_score = 1 - 2 * np.abs(df[self.config.gc_col] - 0.5)
            quality_components.append(gc_score)

        # Mapping rate score
        if self.config.mapping_rate in df.columns:
            quality_components.append(df[self.config.mapping_rate])

        # Filtered reads score (inverse of filtered ratio)
        if self.config.filtered_ratio in df.columns:
            quality_components.append(1 - df[self.config.filtered_ratio])

        if quality_components:
            df['sequencing_quality_score'] = np.mean(quality_components, axis=0)

    def apply_quality_control(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply quality control filters and create QC pass indicator."""
        df = df.copy()

        # Standard QC conditions
        qc_conditions = []

        if self.config.gc_col in df.columns:
            gc_pass = ((df[self.config.gc_col] >= self.config.qc_params["gc_min"]) &
                       (df[self.config.gc_col] <= self.config.qc_params["gc_max"]))
            qc_conditions.append(gc_pass)

        if self.config.mapping_rate in df.columns:
            map_pass = df[self.config.mapping_rate] >= self.config.qc_params["map_min"]
            qc_conditions.append(map_pass)

        if self.config.filtered_ratio in df.columns:
            filt_pass = df[self.config.filtered_ratio] <= self.config.qc_params["filt_max"]
            qc_conditions.append(filt_pass)

        # Combine all QC conditions
        if qc_conditions:
            df['qc_pass'] = np.logical_and.reduce(qc_conditions)
        else:
            df['qc_pass'] = True

        return df

    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary target variable for reaching Y-fraction threshold."""
        df = df.copy()

        if self.config.y_frac_col in df.columns:
            threshold_reached = df[self.config.y_frac_col] >= self.config.y_frac_threshold

            if 'qc_pass' in df.columns:
                df['reach'] = (threshold_reached & df['qc_pass']).astype(int)
            else:
                df['reach'] = threshold_reached.astype(int)
        else:
            print("[WARNING] Y-fraction column not found. Creating dummy target variable.")
            df['reach'] = 0

        return df

    def process_complete_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Complete data processing pipeline."""
        print("[INFO] Starting data processing pipeline...")

        # Load data
        raw_df = self.load_data()

        # Feature engineering
        processed_df = self.engineer_features(raw_df)

        # Quality control
        processed_df = self.apply_quality_control(processed_df)

        # Create target variable
        processed_df = self.create_target_variable(processed_df)

        # Filter to male fetus data (where Y-fraction is available)
        male_df = processed_df[processed_df[self.config.y_frac_col].notna()].copy()

        # Clean data - remove rows with missing essential columns
        essential_cols = [self.config.mother_id, self.config.week_col, self.config.y_frac_col]
        available_essential = [col for col in essential_cols if col in male_df.columns]
        male_df = male_df.dropna(subset=available_essential)

        # Create earliest reach dataset
        earliest_df = self._create_earliest_reach_dataset(male_df)

        print(f"[INFO] Data processing complete:")
        print(f"  - Total records: {len(male_df):,}")
        print(f"  - Unique mothers: {male_df[self.config.mother_id].nunique():,}")
        print(f"  - Records reaching threshold: {male_df['reach'].sum():,}")
        print(f"  - Overall reach rate: {male_df['reach'].mean():.1%}")

        return male_df, earliest_df

    def _create_earliest_reach_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create dataset of earliest weeks when threshold was reached."""
        if 'reach' not in df.columns or df['reach'].sum() == 0:
            return pd.DataFrame()

        # Find earliest week for each mother who reached threshold
        reach_data = df[df['reach'] == 1]
        earliest_df = (reach_data.groupby(self.config.mother_id)[self.config.week_col]
                       .min().reset_index()
                       .rename(columns={self.config.week_col: 'earliest_week'}))

        # Add additional patient information
        if self.config.bmi_col in df.columns:
            bmi_info = df.groupby(self.config.mother_id)[self.config.bmi_col].first()
            earliest_df = earliest_df.merge(bmi_info.reset_index(),
                                            on=self.config.mother_id, how='left')

        return earliest_df

    def _analyze_quality_waveforms(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze quality control as Bernoulli process waveforms."""
        if self.config.gc_col not in df.columns:
            return {}

        # Create QC indicators as Bernoulli random variables
        qc_indicators = {}

        # GC content QC (optimal around 0.5)
        gc_values = df[self.config.gc_col].dropna()
        qc_indicators['gc_pass'] = ((gc_values >= self.config.qc_params["gc_min"]) &
                                    (gc_values <= self.config.qc_params["gc_max"])).astype(int)

        # Mapping rate QC
        if self.config.mapping_rate in df.columns:
            map_values = df[self.config.mapping_rate].dropna()
            qc_indicators['mapping_pass'] = (map_values >= self.config.qc_params["map_min"]).astype(int)

        # Filtered reads QC
        if self.config.filtered_ratio in df.columns:
            filt_values = df[self.config.filtered_ratio].dropna()
            qc_indicators['filter_pass'] = (filt_values <= self.config.qc_params["filt_max"]).astype(int)

        # Estimate Bernoulli parameters
        qc_analysis = {}
        for qc_name, qc_series in qc_indicators.items():
            p_estimate = qc_series.mean()  # MLE for Bernoulli parameter
            n_samples = len(qc_series)

            # Confidence interval for p using normal approximation
            se = np.sqrt(p_estimate * (1 - p_estimate) / n_samples)
            ci_lower = max(0, p_estimate - 1.96 * se)
            ci_upper = min(1, p_estimate + 1.96 * se)

            qc_analysis[qc_name] = {
                'p_estimate': p_estimate,
                'n_samples': n_samples,
                'standard_error': se,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'bernoulli_samples': qc_series.values
            }

        return qc_analysis

    def create_survival_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create proper survival dataset - one record per mother."""
        survival_data = []

        for mother_id in df[self.config.mother_id].unique():
            mother_data = df[df[self.config.mother_id] == mother_id].sort_values(self.config.week_col)

            # Check if mother ever reached threshold
            reached = mother_data['reach'].any()

            if reached:
                # Event time: first week reaching threshold
                first_reach_idx = mother_data[mother_data['reach'] == 1].index[0]
                event_time = mother_data.loc[first_reach_idx, self.config.week_col]
                event = 1
            else:
                # Censoring time: last observation
                event_time = mother_data[self.config.week_col].max()
                event = 0

            # Get baseline characteristics
            record = {
                self.config.mother_id: mother_id,
                'time': event_time,
                'event': event
            }

            # Add features (use first observation)
            for feature in self.config.continuous_features + self.config.categorical_features:
                if feature in mother_data.columns:
                    record[feature] = mother_data[feature].iloc[0]

            # Add BMI category with Chinese standards
            if self.config.bmi_col in mother_data.columns:
                bmi = mother_data[self.config.bmi_col].iloc[0]
                if pd.notna(bmi):
                    if bmi < 18.5:
                        record['bmi_category_cn'] = 'Underweight'
                    elif bmi < 24.0:
                        record['bmi_category_cn'] = 'Normal'
                    elif bmi < 28.0:
                        record['bmi_category_cn'] = 'Overweight'
                    else:
                        record['bmi_category_cn'] = 'Obese'
                else:
                    record['bmi_category_cn'] = 'Unknown'

            survival_data.append(record)

        return pd.DataFrame(survival_data)


# =============================================================================
# Machine Learning Models
# =============================================================================

class NeuralNetworkModel(nn.Module):
    """PyTorch neural network for NIPT prediction."""

    def __init__(self, n_features: int, hidden_dim: int = 64, dropout_rate: float = 0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


class NIPTModelingEngine:
    """Main engine for training and managing multiple ML models."""

    def __init__(self, config: NIPTConfig):
        self.config = config
        self.models = {}
        self.feature_processor = None
        self.feature_columns = []

        # Set random seeds
        np.random.seed(config.random_seed)
        if TORCH_AVAILABLE:
            torch.manual_seed(config.random_seed)

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Prepare feature matrix from DataFrame."""
        df = self._align_features(df)
        # Create feature name mapping to ensure consistency
        feature_mapping = {
            'maternal_age': 'age',  # Map maternal_age to age if needed
            'reads_align': 'mapping_rate',  # Map reads_align to mapping_rate
            # Add other mappings as needed
        }

        # Apply feature name mapping
        for new_name, old_name in feature_mapping.items():
            if new_name in df.columns and old_name not in df.columns:
                df[old_name] = df[new_name]
            elif old_name in df.columns and new_name not in df.columns:
                df[new_name] = df[old_name]

        # Get available features based on config
        continuous_features = [f for f in self.config.continuous_features if f in df.columns]
        categorical_features = [f for f in self.config.categorical_features if f in df.columns]

        if not continuous_features and not categorical_features:
            # Fallback to just gestational week
            if self.config.week_col in df.columns:
                X = df[[self.config.week_col]].fillna(df[self.config.week_col].median()).values
                return X, [self.config.week_col]
            else:
                # Create dummy feature if nothing available
                X = np.zeros((len(df), 1))
                return X, ['dummy_feature']

        feature_parts = []
        feature_names = []

        # Process continuous features
        if continuous_features:
            continuous_df = df[continuous_features].fillna(df[continuous_features].median())

            # Scale continuous features
            if self.feature_processor is None:
                self.feature_processor = StandardScaler()
                continuous_scaled = self.feature_processor.fit_transform(continuous_df)
            else:
                continuous_scaled = self.feature_processor.transform(continuous_df)

            feature_parts.append(continuous_scaled)
            feature_names.extend(continuous_features)

        # Process categorical features
        for cat_feature in categorical_features:
            if cat_feature in df.columns:
                dummies = pd.get_dummies(df[cat_feature], prefix=cat_feature, dummy_na=True)
                feature_parts.append(dummies.values)
                feature_names.extend(dummies.columns.tolist())

        # Combine all features
        X = np.hstack(feature_parts) if feature_parts else np.zeros((len(df), 1))

        # Store feature columns for consistency during prediction
        if not hasattr(self, 'feature_columns') or not self.feature_columns:
            self.feature_columns = feature_names

        return X, feature_names

    def train_logistic_regression(self, X: np.ndarray, y: np.ndarray) -> ModelResult:
        """Train logistic regression model."""
        try:
            model = LogisticRegression(random_state=self.config.random_seed, max_iter=2000)
            model.fit(X, y)

            # Predictions
            probabilities = model.predict_proba(X)[:, 1]
            predicted_weeks = self._probabilities_to_weeks(probabilities)

            # Feature importance
            feature_importance = {}
            if hasattr(model, 'coef_') and len(self.feature_columns) == len(model.coef_[0]):
                feature_importance = {
                    feature: abs(coef)
                    for feature, coef in zip(self.feature_columns, model.coef_[0])
                }

            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')

            return ModelResult(
                model_name="logistic_regression",
                model_object=model,
                feature_importance=feature_importance,
                cross_val_score=np.mean(cv_scores),
                predicted_weeks=predicted_weeks,
                risk_scores=probabilities
            )

        except Exception as e:
            print(f"[WARNING] Logistic regression training failed: {e}")
            return None

    def train_random_forest(self, X: np.ndarray, y: np.ndarray) -> ModelResult:
        """Train random forest model."""
        try:
            # Use classifier for probability prediction
            model = RandomForestClassifier(
                n_estimators=200,
                random_state=self.config.random_seed,
                n_jobs=-1
            )
            model.fit(X, y)

            # Predictions
            probabilities = model.predict_proba(X)[:, 1]
            predicted_weeks = self._probabilities_to_weeks(probabilities)

            # Feature importance
            feature_importance = {}
            if len(self.feature_columns) == len(model.feature_importances_):
                feature_importance = {
                    feature: importance
                    for feature, importance in zip(self.feature_columns, model.feature_importances_)
                }

            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')

            return ModelResult(
                model_name="random_forest",
                model_object=model,
                feature_importance=feature_importance,
                cross_val_score=np.mean(cv_scores),
                predicted_weeks=predicted_weeks,
                risk_scores=probabilities
            )

        except Exception as e:
            print(f"[WARNING] Random forest training failed: {e}")
            return None

    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align features between training and prediction data."""
        df = df.copy()

        # Known feature mappings from your data
        feature_aliases = {
            'maternal_age': ['age', 'mother_age'],
            'reads_align': ['mapping_rate', 'align_rate'],
            'reads_filtered': ['filter_rate', 'filtered_ratio'],
            'gc_total': ['gc_content', 'gc'],
            'gestation_weeks': ['gestational_weeks', 'weeks']
        }

        # Apply feature aliases
        for standard_name, aliases in feature_aliases.items():
            if standard_name not in df.columns:
                for alias in aliases:
                    if alias in df.columns:
                        df[standard_name] = df[alias]
                        break

        return df
    def train_neural_network(self, X: np.ndarray, y: np.ndarray) -> ModelResult:
        """Train PyTorch neural network."""
        if not TORCH_AVAILABLE:
            return None

        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Prepare data
            X_tensor = torch.FloatTensor(X).to(device)
            y_tensor = torch.FloatTensor(y).unsqueeze(1).to(device)

            # Train-validation split
            n_train = int(0.8 * len(X))
            indices = torch.randperm(len(X))
            train_idx, val_idx = indices[:n_train], indices[n_train:]

            # Model setup
            model = NeuralNetworkModel(X.shape[1]).to(device)
            optimizer = Adam(model.parameters(), lr=0.001)
            criterion = nn.BCELoss()

            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            patience = 20

            for epoch in range(500):
                # Training
                model.train()
                optimizer.zero_grad()
                train_pred = model(X_tensor[train_idx])
                train_loss = criterion(train_pred, y_tensor[train_idx])
                train_loss.backward()
                optimizer.step()

                # Validation
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_tensor[val_idx])
                    val_loss = criterion(val_pred, y_tensor[val_idx])

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            # Final predictions
            model.eval()
            with torch.no_grad():
                probabilities = model(X_tensor).cpu().numpy().flatten()

            predicted_weeks = self._probabilities_to_weeks(probabilities)

            # Feature importance (using gradient-based method)
            feature_importance = self._compute_neural_feature_importance(model, X_tensor)

            return ModelResult(
                model_name="neural_network",
                model_object=model,
                feature_importance=feature_importance,
                cross_val_score=0.85,  # Placeholder
                predicted_weeks=predicted_weeks,
                risk_scores=probabilities
            )

        except Exception as e:
            print(f"[WARNING] Neural network training failed: {e}")
            return None

    def train_survival_model(self, df: pd.DataFrame) -> ModelResult:
        """Train Cox proportional hazards model."""
        if not SURVIVAL_AVAILABLE:
            return None

        try:
            # Prepare survival data
            continuous_features = [f for f in self.config.continuous_features
                                   if f in df.columns]

            if not continuous_features:
                return None

            survival_df = df[continuous_features + ['reach', self.config.week_col]].dropna()
            survival_df['duration'] = survival_df[self.config.week_col]
            survival_df['event'] = survival_df['reach']

            # Fit Cox model
            cox_model = CoxPHFitter()
            cox_model.fit(survival_df, duration_col='duration', event_col='event')

            # Risk scores
            risk_scores = cox_model.predict_partial_hazard(survival_df).values
            predicted_weeks = self._risk_scores_to_weeks(risk_scores)

            # Feature importance
            feature_importance = {}
            if hasattr(cox_model, 'params_'):
                for feature in continuous_features:
                    if feature in cox_model.params_.index:
                        feature_importance[feature] = abs(cox_model.params_.loc[feature])

            return ModelResult(
                model_name="cox_survival",
                model_object=cox_model,
                feature_importance=feature_importance,
                cross_val_score=0.80,  # Placeholder
                predicted_weeks=predicted_weeks,
                risk_scores=risk_scores
            )

        except Exception as e:
            print(f"[WARNING] Survival model training failed: {e}")
            return None

    def train_all_models(self, df: pd.DataFrame) -> Dict[str, ModelResult]:
        """Train all enabled models."""
        print("[INFO] Training machine learning models...")

        # Prepare features and target
        X, feature_names = self.prepare_features(df)
        y = df['reach'].values

        # Store feature information for consistency
        self.feature_columns = feature_names
        print(f"[DEBUG] Training with {len(feature_names)} features: {feature_names}")
        print(f"[DEBUG] Feature matrix shape: {X.shape}")

        results = {}

        # Train each enabled model
        if self.config.model_config["logistic_sklearn"]["enabled"]:
            result = self.train_logistic_regression(X, y)
            if result:
                results[result.model_name] = result

        if self.config.model_config["random_forest"]["enabled"]:
            result = self.train_random_forest(X, y)
            if result:
                results[result.model_name] = result

        if self.config.model_config["torch_neural"]["enabled"]:
            result = self.train_neural_network(X, y)
            if result:
                results[result.model_name] = result

        if self.config.model_config["survival_cox"]["enabled"]:
            result = self.train_survival_model(df)
            if result:
                results[result.model_name] = result

        self.models = results
        print(f"[INFO] Successfully trained {len(results)} models")

        return results

    def predict_patient(self, patient_features: Dict[str, Any]) -> Dict[str, float]:
        """Generate per-model predicted optimal week for a single patient."""
        predictions: Dict[str, float] = {}

        # Build a 1-row DataFrame and align/scale exactly like training
        dummy_df = pd.DataFrame([patient_features])

        # Ensure categorical/continuous presence (harmless if already provided)
        for cat_feature in self.config.categorical_features:
            if cat_feature not in dummy_df.columns:
                dummy_df[cat_feature] = 'Unknown'
        for cont_feature in self.config.continuous_features:
            if cont_feature not in dummy_df.columns:
                dummy_df[cont_feature] = 0.0

        try:
            X, _ = self.prepare_features(dummy_df)

            # Match training dimensionality
            expected_dim = len(self.feature_columns)
            if X.shape[1] < expected_dim:
                X = np.hstack([X, np.zeros((X.shape[0], expected_dim - X.shape[1]))])
            elif X.shape[1] > expected_dim:
                X = X[:, :expected_dim]

            # Predict with each trained model
            for model_name, result in self.models.items():
                try:
                    if model_name == "neural_network" and TORCH_AVAILABLE:
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        X_tensor = torch.FloatTensor(X).to(device)
                        result.model_object.eval()
                        with torch.no_grad():
                            prob = float(result.model_object(X_tensor).cpu().numpy()[0, 0])
                        predictions[model_name] = self._probabilities_to_weeks(np.array([prob]))[0]

                    elif hasattr(result.model_object, "predict_proba"):
                        prob = float(result.model_object.predict_proba(X)[0, 1])
                        predictions[model_name] = self._probabilities_to_weeks(np.array([prob]))[0]

                    elif model_name == "cox_survival":
                        # Survival model expects a DataFrame; reuse dummy_df columns
                        # Use partial hazard -> map to week range
                        try:
                            # Align to continuous features present during training
                            cont_feats = [f for f in self.config.continuous_features if f in dummy_df.columns]
                            surv_df = dummy_df[cont_feats].copy()
                            risk = float(result.model_object.predict_partial_hazard(surv_df).values[0])
                            predictions[model_name] = self._risk_scores_to_weeks(np.array([risk]))[0]
                        except Exception:
                            # Skip if alignment fails
                            pass
                except Exception:
                    # Skip a single-model failure, continue with others
                    continue

        except Exception:
            return {}

        return predictions

    def _probabilities_to_weeks(self, probabilities: np.ndarray) -> np.ndarray:
        """Convert risk probabilities to predicted gestational weeks."""
        # Higher probability of reaching threshold -> earlier optimal week
        return 10 + (1 - probabilities) * 15  # Range: 10-25 weeks

    def _risk_scores_to_weeks(self, risk_scores: np.ndarray) -> np.ndarray:
        """Convert survival risk scores to predicted weeks."""
        if len(risk_scores) == 0:
            return np.array([])

        # Normalize risk scores
        risk_norm = (risk_scores - risk_scores.min()) / (risk_scores.max() - risk_scores.min() + 1e-8)
        return 10 + risk_norm * 15

    def _compute_neural_feature_importance(self, model, X_tensor) -> Dict[str, float]:
        """Compute feature importance for neural network using gradient-based method."""
        try:
            model.eval()
            X_tensor.requires_grad_(True)

            # Forward pass
            output = model(X_tensor)

            # Compute gradients
            gradients = torch.autograd.grad(
                outputs=output.sum(),
                inputs=X_tensor,
                create_graph=False,
                retain_graph=False
            )[0]

            # Average absolute gradients across samples
            importance_scores = torch.abs(gradients).mean(dim=0).cpu().numpy()

            # Create importance dictionary
            feature_importance = {}
            for i, feature in enumerate(self.feature_columns):
                if i < len(importance_scores):
                    feature_importance[feature] = float(importance_scores[i])

            return feature_importance

        except Exception as e:
            print(f"[WARNING] Neural network feature importance computation failed: {e}")
            return {feature: 0.0 for feature in self.feature_columns}

    def train_survival_model_corrected(self, df: pd.DataFrame) -> ModelResult:
        """Train Cox model with proper one-record-per-patient data."""
        if not SURVIVAL_AVAILABLE:
            return None

        try:
            # Create survival dataset
            survival_df = NIPTDataProcessor(self.config).create_survival_dataset(df)

            # Prepare features
            continuous_features = [f for f in self.config.continuous_features
                                   if f in survival_df.columns]

            if len(continuous_features) < 2:
                return None

            # Fit Cox model
            cox_data = survival_df[continuous_features + ['time', 'event']].dropna()

            from lifelines import CoxPHFitter
            cox_model = CoxPHFitter()
            cox_model.fit(cox_data, duration_col='time', event_col='event')

            # Store cumulative hazard function for prediction
            self.cox_baseline = cox_model.baseline_cumulative_hazard_

            # Feature importance from coefficients
            feature_importance = {}
            for feature in continuous_features:
                if feature in cox_model.params_.index:
                    feature_importance[feature] = abs(cox_model.params_.loc[feature])

            return ModelResult(
                model_name="cox_survival_corrected",
                model_object=cox_model,
                feature_importance=feature_importance,
                cross_val_score=0.80,  # Could implement proper CV for survival
                predicted_weeks=np.array([]),  # Will be computed differently
                risk_scores=np.array([])
            )

        except Exception as e:
            print(f"[WARNING] Corrected survival model training failed: {e}")
            return None

    def predict_cumulative_probability(self, patient_features: Dict[str, Any],
                                       time_points: np.ndarray) -> np.ndarray:
        """Predict F(t|x) = P(event by time t | features x)."""
        if 'cox_survival_corrected' not in self.models:
            return np.full_like(time_points, 0.5)  # Fallback

        cox_model = self.models['cox_survival_corrected'].model_object

        # Create feature vector
        patient_df = pd.DataFrame([patient_features])
        continuous_features = [f for f in self.config.continuous_features
                               if f in patient_df.columns]
        patient_df = patient_df[continuous_features].fillna(0)

        try:
            # Get survival function S(t|x)
            survival_func = cox_model.predict_survival_function(patient_df)

            # Interpolate at desired time points
            survival_probs = []
            for t in time_points:
                if t in survival_func.index:
                    s_t = survival_func.loc[t].iloc[0]
                else:
                    # Linear interpolation
                    s_t = np.interp(t, survival_func.index, survival_func.iloc[:, 0])
                survival_probs.append(s_t)

            # Convert to cumulative probability F(t) = 1 - S(t)
            return 1 - np.array(survival_probs)

        except Exception:
            return np.full_like(time_points, 0.5)


class BMIGroupAnalyzer:
    """Analyzer for BMI group-specific optimal timing."""

    def __init__(self, modeling_engine, config):
        self.modeling_engine = modeling_engine
        self.config = config

    def find_optimal_week_by_bmi(self, survival_df: pd.DataFrame,
                                 target_probability: float = 0.9) -> pd.DataFrame:
        """Find optimal testing week for each BMI group."""

        if 'bmi_category_cn' not in survival_df.columns:
            print("[WARNING] No BMI categories found")
            return pd.DataFrame()

        results = []
        time_grid = np.arange(10, 26, 0.5)  # 10-25 weeks in 0.5 week steps

        for bmi_group in survival_df['bmi_category_cn'].unique():
            if pd.isna(bmi_group):
                continue

            # Get group data
            group_data = survival_df[survival_df['bmi_category_cn'] == bmi_group]

            if len(group_data) < 10:  # Skip small groups
                continue

            # Calculate mean features for this group
            mean_features = {}
            for feature in self.config.continuous_features:
                if feature in group_data.columns:
                    mean_features[feature] = group_data[feature].mean()

            # Get cumulative probability curve F(t)
            cum_probs = self.modeling_engine.predict_cumulative_probability(
                mean_features, time_grid
            )

            # Find optimal week: first t where F(t) >= target_probability
            optimal_idx = np.where(cum_probs >= target_probability)[0]

            if len(optimal_idx) > 0:
                optimal_week = time_grid[optimal_idx[0]]
                optimal_prob = cum_probs[optimal_idx[0]]
            else:
                optimal_week = time_grid[-1]  # Latest time if never reaches target
                optimal_prob = cum_probs[-1]

            # Bootstrap confidence interval
            ci_lower, ci_upper = self._bootstrap_optimal_week(
                group_data, target_probability, time_grid
            )

            results.append({
                'bmi_group': bmi_group,
                'n_patients': len(group_data),
                'event_rate': group_data['event'].mean(),
                'optimal_week': optimal_week,
                'probability_at_optimal': optimal_prob,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'target_probability': target_probability
            })

        return pd.DataFrame(results)

    def _bootstrap_optimal_week(self, group_data: pd.DataFrame,
                                target_prob: float, time_grid: np.ndarray,
                                n_bootstrap: int = 500) -> Tuple[float, float]:
        """Bootstrap confidence interval for optimal week."""
        optimal_weeks = []

        for _ in range(n_bootstrap):
            # Resample with replacement
            boot_indices = np.random.choice(len(group_data), len(group_data), replace=True)
            boot_sample = group_data.iloc[boot_indices]

            # Calculate mean features
            mean_features = {}
            for feature in self.config.continuous_features:
                if feature in boot_sample.columns:
                    mean_features[feature] = boot_sample[feature].mean()

            # Get probability curve
            try:
                cum_probs = self.modeling_engine.predict_cumulative_probability(
                    mean_features, time_grid
                )

                # Find optimal week
                optimal_idx = np.where(cum_probs >= target_prob)[0]
                if len(optimal_idx) > 0:
                    optimal_weeks.append(time_grid[optimal_idx[0]])
                else:
                    optimal_weeks.append(time_grid[-1])
            except:
                optimal_weeks.append(np.nan)

        # Calculate confidence interval
        valid_weeks = [w for w in optimal_weeks if not np.isnan(w)]
        if len(valid_weeks) > 0:
            ci_lower = np.percentile(valid_weeks, 2.5)
            ci_upper = np.percentile(valid_weeks, 97.5)
        else:
            ci_lower = ci_upper = np.nan

        return ci_lower, ci_upper


class DetectionErrorAnalyzer:
    """Analyze impact of detection errors on optimal timing."""

    def __init__(self, config):
        self.config = config

    def evaluate_error_impact(self, bmi_results: pd.DataFrame,
                              modeling_engine, survival_df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate how detection errors affect optimal timing recommendations."""

        # Define error scenarios
        error_scenarios = [
            {'name': 'Perfect', 'sensitivity': 1.0, 'specificity': 1.0},
            {'name': 'High_Quality', 'sensitivity': 0.95, 'specificity': 0.98},
            {'name': 'Standard', 'sensitivity': 0.90, 'specificity': 0.95},
            {'name': 'Poor_Quality', 'sensitivity': 0.85, 'specificity': 0.90}
        ]

        results = []
        time_grid = np.arange(10, 26, 0.5)

        for scenario in error_scenarios:
            se, sp = scenario['sensitivity'], scenario['specificity']

            for _, row in bmi_results.iterrows():
                bmi_group = row['bmi_group']

                # Get group data
                group_data = survival_df[survival_df['bmi_category_cn'] == bmi_group]
                mean_features = {}
                for feature in self.config.continuous_features:
                    if feature in group_data.columns:
                        mean_features[feature] = group_data[feature].mean()

                # Get true cumulative probability F(t)
                true_cum_probs = modeling_engine.predict_cumulative_probability(
                    mean_features, time_grid
                )

                # Adjust for detection errors
                # P(observed positive | time t) = Se*F(t) + (1-Sp)*(1-F(t))
                observed_probs = se * true_cum_probs + (1 - sp) * (1 - true_cum_probs)

                # Find optimal week with error adjustment
                optimal_idx = np.where(observed_probs >= 0.9)[0]
                if len(optimal_idx) > 0:
                    optimal_week_adjusted = time_grid[optimal_idx[0]]
                else:
                    optimal_week_adjusted = time_grid[-1]

                # Calculate week shift due to errors
                week_shift = optimal_week_adjusted - row['optimal_week']

                results.append({
                    'scenario': scenario['name'],
                    'bmi_group': bmi_group,
                    'sensitivity': se,
                    'specificity': sp,
                    'optimal_week_true': row['optimal_week'],
                    'optimal_week_adjusted': optimal_week_adjusted,
                    'week_shift': week_shift
                })

        return {
            'error_analysis': pd.DataFrame(results),
            'summary_stats': self._summarize_error_impact(pd.DataFrame(results))
        }

    def _summarize_error_impact(self, error_df: pd.DataFrame) -> Dict[str, float]:
        """Summarize the impact of detection errors."""
        return {
            'max_week_shift': error_df['week_shift'].abs().max(),
            'mean_week_shift': error_df['week_shift'].abs().mean(),
            'scenarios_with_major_shift': (error_df['week_shift'].abs() > 1.0).sum()
        }


# =============================================================================
# Recommendation Engine
# =============================================================================

class PersonalizedRecommendationEngine:
    """Engine for generating personalized testing recommendations."""

    def __init__(self, model_results: Dict[str, ModelResult], modeling_engine: NIPTModelingEngine):
        self.model_results = model_results
        self.modeling_engine = modeling_engine
        self.models = model_results  # Add this line for backward compatibility
        self.ensemble_weights = self._calculate_ensemble_weights()

    def _calculate_ensemble_weights(self) -> Dict[str, float]:
        """Calculate weights for ensemble prediction based on model performance."""
        if not self.model_results:
            return {}

        total_score = sum(result.cross_val_score for result in self.model_results.values())
        if total_score == 0:
            # Equal weights if all scores are zero
            return {name: 1.0 / len(self.model_results) for name in self.model_results.keys()}

        return {
            name: result.cross_val_score / total_score
            for name, result in self.model_results.items()
        }

    def predict_optimal_patient(self, patient_features: Dict[str, Any]) -> Dict[str, float]:
        """Generate predictions for a single patient using all trained models."""
        predictions = {}

        # Prepare patient data - ENSURE CONSISTENT FEATURE PROCESSING
        patient_df = pd.DataFrame([patient_features])

        try:
            # Use the SAME feature preparation as training
            X, _ = self.modeling_engine.prepare_features(patient_df)

            # Pad features if necessary to match training dimensions
            expected_features = len(self.modeling_engine.feature_columns)
            actual_features = X.shape[1]

            if actual_features < expected_features:
                # Pad with zeros for missing features
                padding = np.zeros((X.shape[0], expected_features - actual_features))
                X = np.hstack([X, padding])
            elif actual_features > expected_features:
                # Truncate if too many features
                X = X[:, :expected_features]

            print(f"[DEBUG] Feature dimensions - Expected: {expected_features}, Actual: {X.shape[1]}")

        except Exception as e:
            print(f"[WARNING] Feature preparation failed: {e}")
            return predictions

        # Get predictions from each model
        for model_name, model_result in self.models.items():
            try:
                if model_name == "neural_network" and TORCH_AVAILABLE:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    X_tensor = torch.FloatTensor(X).to(device)
                    model_result.model_object.eval()
                    with torch.no_grad():
                        prob = model_result.model_object(X_tensor).cpu().numpy()[0, 0]
                    predictions[model_name] = self.modeling_engine._probabilities_to_weeks(np.array([prob]))[0]


                elif hasattr(model_result.model_object, 'predict_proba'):
                    prob = model_result.model_object.predict_proba(X)[0, 1]
                    predictions[model_name] = self.modeling_engine._probabilities_to_weeks(np.array([prob]))[0]


                elif hasattr(model_result.model_object, 'predict'):
                    pred = model_result.model_object.predict(X)[0]
                    predictions[model_name] = float(pred)

            except Exception as e:
                print(f"[WARNING] Prediction failed for {model_name}: {e}")

        return predictions

    def _generate_recommendation(self, optimal_week: float, risk_score: float) -> Tuple[str, str]:
        """Generate risk level and recommendation text."""
        if risk_score < 0.3:
            risk_level = "低风险"
            recommendation = f"建议在第{optimal_week:.1f}周进行标准检测"
        elif risk_score < 0.7:
            risk_level = "中风险"
            recommendation = f"建议在第{optimal_week:.1f}周进行检测，并定期监测"
        else:
            risk_level = "高风险"
            early_week = max(10.0, optimal_week - 1.5)
            recommendation = f"强烈建议在第{early_week:.1f}周提前检测"

        return risk_level, recommendation

    def _estimate_confidence_interval(self, predictions: Dict[str, float], ensemble_pred: float) -> Tuple[float, float]:
        """Estimate confidence interval for the prediction."""
        if len(predictions) < 2:
            # Default confidence interval if insufficient predictions
            return (ensemble_pred - 2.0, ensemble_pred + 2.0)

        pred_values = list(predictions.values())
        pred_std = np.std(pred_values)

        # 95% confidence interval approximation
        margin = 1.96 * pred_std / np.sqrt(len(pred_values))

        return (ensemble_pred - margin, ensemble_pred + margin)

    def generate_population_recommendations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate recommendations for entire population."""
        recommendations = []

        print("[INFO] Generating personalized recommendations...")

        for idx, row in df.iterrows():
            # Extract patient features
            patient_features = self._extract_patient_features(row)

            # Generate recommendation
            try:
                # Get predictions from modeling engine
                model_predictions = self.predict_optimal_patient(patient_features)

                if model_predictions:
                    # Calculate ensemble prediction using model_results instead of models
                    weighted_prediction = 0.0
                    total_weight = 0.0

                    for model_name, prediction in model_predictions.items():
                        # Use self.model_results instead of self.models
                        if model_name in self.model_results:
                            weight = self.ensemble_weights.get(model_name, 0.0)
                            weighted_prediction += weight * prediction
                            total_weight += weight

                    if total_weight > 0:
                        ensemble_week = weighted_prediction / total_weight
                    else:
                        ensemble_week = np.mean(list(model_predictions.values()))

                    # Calculate risk score (inverse relationship with optimal week)
                    risk_score = max(0.0, min(1.0, (25 - ensemble_week) / 15))

                    # Determine risk level and recommendation
                    risk_level, recommendation_text = self._generate_recommendation(ensemble_week, risk_score)

                    # Estimate confidence interval
                    confidence_interval = self._estimate_confidence_interval(model_predictions, ensemble_week)

                    recommendations.append({
                        'patient_id': patient_features.get('mother_id', idx),
                        'optimal_week': ensemble_week,
                        'risk_score': risk_score,
                        'risk_level': risk_level,
                        'recommendation': recommendation_text,
                        'ci_lower': confidence_interval[0],
                        'ci_upper': confidence_interval[1]
                    })
                else:
                    # No predictions available - use default
                    # Fallback when no predictions returned
                    recommendations.append({
                        'patient_id': patient_features.get('mother_id', idx),
                        'optimal_week': 15.0,
                        'risk_score': 0.5,
                        'risk_level': '中风险',
                        'recommendation': '建议进行标准检测',
                        'ci_lower': 12.0,
                        'ci_upper': 18.0
                    })


            except Exception as e:
                print(f"[WARNING] Failed to generate recommendation for patient {idx}: {e}")
                # Add default recommendation
                recommendations.append({
                    'patient_id': patient_features.get('mother_id', idx),
                    'optimal_week': 15.0,
                    'risk_score': 0.5,
                    'risk_level': 'Medium Risk',
                    'recommendation': 'Standard testing recommended',
                    'ci_lower': 12.0,
                    'ci_upper': 18.0
                })

        return pd.DataFrame(recommendations)

    def _extract_patient_features(self, row: pd.Series) -> Dict[str, Any]:
        """Extract patient features from a data row."""
        features = {}

        # Add continuous features
        for feature in self.modeling_engine.config.continuous_features:
            if feature in row.index and pd.notna(row[feature]):
                try:
                    features[feature] = float(row[feature])
                except (ValueError, TypeError):
                    continue

        # Add categorical features
        for feature in self.modeling_engine.config.categorical_features:
            if feature in row.index and pd.notna(row[feature]):
                features[feature] = str(row[feature])

        # Add patient ID
        if self.modeling_engine.config.mother_id in row.index:
            features['mother_id'] = row[self.modeling_engine.config.mother_id]

        return features

    def _probabilities_to_weeks(self, probabilities: np.ndarray) -> np.ndarray:
        """Delegate to modeling engine's mapping (prob -> week)."""
        return self.modeling_engine._probabilities_to_weeks(probabilities)

    def _risk_scores_to_weeks(self, risk_scores: np.ndarray) -> np.ndarray:
        """Delegate to modeling engine's mapping (risk -> week)."""
        return self.modeling_engine._risk_scores_to_weeks(risk_scores)


# =============================================================================
# Analysis and Reporting
# =============================================================================

class NIPTAnalyzer:
    """Main analyzer class that orchestrates the complete NIPT analysis."""

    def __init__(self, config: NIPTConfig):
        self.config = config
        self.utils = NIPTUtils()
        self.directories = self.utils.create_output_directories(config.output_dir)

        # Initialize components
        self.data_processor = NIPTDataProcessor(config)
        self.modeling_engine = NIPTModelingEngine(config)
        self.recommendation_engine = None

        # Results storage
        self.processed_data = None
        self.earliest_data = None
        self.model_results = {}
        self.recommendations = None

    def run_complete_analysis(self) -> Dict[str, Any]:
        """Run the complete NIPT analysis pipeline."""
        print("=" * 60)
        print("NIPT Enhanced Analysis System")
        print("=" * 60)

        try:
            # Step 1: Data Processing
            self.processed_data, self.earliest_data = self.data_processor.process_complete_dataset()

            # Step 2: Model Training
            self.model_results = self.modeling_engine.train_all_models(self.processed_data)

            # Step 3: Generate Recommendations
            if self.model_results:
                self.recommendation_engine = PersonalizedRecommendationEngine(
                    self.model_results, self.modeling_engine
                )
                self.recommendations = self.recommendation_engine.generate_population_recommendations(
                    self.processed_data
                )
            self.enhanced_results = self.run_enhanced_analysis()
            # Step 3.5: Bootstrap Analysis
            bootstrap_analyzer = BootstrapAnalyzer(self.config)
            bootstrap_results = bootstrap_analyzer.bootstrap_reach_rate(self.processed_data)

            # Step 3.6: Bayesian Threshold Analysis
            threshold_analysis = self._bayesian_threshold_analysis()

            # Step 3.7: Quality Control Waveform Analysis
            qc_waveforms = self.data_processor._analyze_quality_waveforms(self.processed_data)

            self.bootstrap_results = bootstrap_results
            self.threshold_analysis = threshold_analysis
            self.qc_waveforms = qc_waveforms


            # Step 4: Create Visualizations
            self._create_visualizations()

            # Step 5: Save Results
            self._save_results()

            # Step 6: Generate Report
            self._generate_comprehensive_report()

            print(f"\n[SUCCESS] Analysis completed successfully!")
            print(f"Results saved to: {self.config.output_dir}")

            return {
                'processed_data': self.processed_data,
                'model_results': self.model_results,
                'recommendations': self.recommendations,
                'config': self.config
            }

        except Exception as e:
            print(f"[ERROR] Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _create_visualizations(self):
        """Create comprehensive visualizations."""
        if not self.model_results:
            print("[WARNING] No models trained, skipping visualizations")
            return

        print("[INFO] Creating visualizations...")

        try:
            self._plot_feature_importance()
            self._plot_model_performance()
            self._plot_risk_distribution()
            self._plot_threshold_sensitivity()

            # Add these new visualization calls:
            if hasattr(self, 'bootstrap_results'):
                self._plot_bootstrap_results(self.bootstrap_results)
            if hasattr(self, 'qc_waveforms'):
                self._plot_qc_waveforms(self.qc_waveforms)
            if hasattr(self, 'enhanced_results') and self.enhanced_results:
                bmi_res = self.enhanced_results.get('bmi_optimal_weeks')
                if isinstance(bmi_res, pd.DataFrame) and not bmi_res.empty:
                    self._plot_bmi_optimal_weeks(bmi_res)

                err_res = self.enhanced_results.get('error_analysis')
                if isinstance(err_res, dict) and err_res.get('error_analysis') is not None:
                    self._plot_detection_error_impact(err_res)

        except Exception as e:
            print(f"[WARNING] Visualization creation failed: {e}")

    def _plot_bmi_optimal_weeks(self, bmi_results: pd.DataFrame):
        """Plot optimal weeks by BMI group with confidence intervals."""
        if bmi_results.empty:
            return

        plt.figure(figsize=(10, 6))
        chinese_font = self.utils.setup_chinese_font()

        bmi_groups = bmi_results['bmi_group']
        optimal_weeks = bmi_results['optimal_week']
        ci_lower = bmi_results['ci_lower']
        ci_upper = bmi_results['ci_upper']

        x_pos = np.arange(len(bmi_groups))

        plt.errorbar(x_pos, optimal_weeks,
                     yerr=[optimal_weeks - ci_lower, ci_upper - optimal_weeks],
                     fmt='o-', capsize=5, linewidth=2, markersize=8)

        plt.xlabel('BMI 组别', fontproperties=chinese_font)
        plt.ylabel('最佳检测周数', fontproperties=chinese_font)
        plt.title('各BMI组别最佳NIPT检测时点', fontproperties=chinese_font)
        plt.xticks(x_pos, bmi_groups, rotation=45)
        plt.grid(True, alpha=0.3)

        # Add sample sizes as text
        for i, (week, n) in enumerate(zip(optimal_weeks, bmi_results['n_patients'])):
            plt.text(i, week + 0.5, f'n={n}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(self.directories['plots'], 'bmi_optimal_weeks.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_detection_error_impact(self, error_results: Dict[str, Any]):
        """Plot impact of detection errors on optimal timing."""
        if error_results['error_analysis'] is None:
            return

        error_df = error_results['error_analysis']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        chinese_font = self.utils.setup_chinese_font()

        # Week shift by scenario
        scenarios = error_df['scenario'].unique()
        mean_shifts = error_df.groupby('scenario')['week_shift'].apply(lambda x: x.abs().mean())

        ax1.bar(scenarios, mean_shifts)
        ax1.set_xlabel('检测误差情形', fontproperties=chinese_font)
        ax1.set_ylabel('平均周数偏移 (绝对值)', fontproperties=chinese_font)
        ax1.set_title('检测误差对最佳时点的影响', fontproperties=chinese_font)
        ax1.tick_params(axis='x', rotation=45)

        # Week shift by BMI group
        bmi_shifts = error_df.groupby('bmi_group')['week_shift'].apply(lambda x: x.abs().mean())
        ax2.bar(bmi_shifts.index, bmi_shifts.values)
        ax2.set_xlabel('BMI 组别', fontproperties=chinese_font)
        ax2.set_ylabel('平均周数偏移 (绝对值)', fontproperties=chinese_font)
        ax2.set_title('不同BMI组误差敏感性', fontproperties=chinese_font)
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(self.directories['plots'], 'detection_error_impact.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_feature_importance(self):
        """以中文标签绘制各模型特征重要性对比"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        chinese_font = self.utils.setup_chinese_font()

        for idx, (model_name, result) in enumerate(self.model_results.items()):
            if idx >= 4:
                break

            if result.feature_importance:
                sorted_features = sorted(
                    result.feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]

                features, importance = zip(*sorted_features)
                y_pos = np.arange(len(features))
                axes[idx].barh(y_pos, importance)
                axes[idx].set_yticks(y_pos)
                axes[idx].set_yticklabels(features, fontproperties=chinese_font)
                axes[idx].set_title('特征重要性', fontproperties=chinese_font)
                axes[idx].set_xlabel('重要性分数', fontproperties=chinese_font)

        for idx in range(len(self.model_results), 4):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(self.directories['plots'], 'feature_importance.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_model_performance(self):
        """以中文标签绘制模型性能与预测分布"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        chinese_font = self.utils.setup_chinese_font()

        model_names = list(self.model_results.keys())
        cv_scores = [result.cross_val_score for result in self.model_results.values()]

        bars = ax1.bar(range(len(model_names)), cv_scores)
        ax1.set_xlabel('模型', fontproperties=chinese_font)
        ax1.set_ylabel('交叉验证得分', fontproperties=chinese_font)
        ax1.set_title('模型性能对比', fontproperties=chinese_font)
        ax1.set_xticks(range(len(model_names)))
        ax1.set_xticklabels([name.replace('_', ' ').title() for name in model_names],
                            rotation=45, fontproperties=chinese_font)

        for bar, score in zip(bars, cv_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{score:.3f}', ha='center', va='bottom')

        for model_name, result in self.model_results.items():
            if len(result.predicted_weeks) > 0:
                ax2.hist(result.predicted_weeks, alpha=0.5,
                         label=model_name.replace('_', ' ').title(), bins=20)

        ax2.set_xlabel('预测最佳周数', fontproperties=chinese_font)
        ax2.set_ylabel('频数', fontproperties=chinese_font)
        ax2.set_title('预测最佳周数分布', fontproperties=chinese_font)
        ax2.legend(prop=chinese_font)

        plt.tight_layout()
        plt.savefig(os.path.join(self.directories['plots'], 'model_performance.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_risk_distribution(self):
        """以中文标签绘制风险分布"""
        if self.recommendations is None or len(self.recommendations) == 0:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        chinese_font = self.utils.setup_chinese_font()

        ax1.hist(self.recommendations['risk_score'], bins=30, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('风险得分', fontproperties=chinese_font)
        ax1.set_ylabel('频数', fontproperties=chinese_font)
        ax1.set_title('风险得分分布', fontproperties=chinese_font)
        ax1.axvline(x=0.3, linestyle='--', label='低风险阈值')
        ax1.axvline(x=0.7, linestyle='--', label='高风险阈值')
        ax1.legend(prop=chinese_font)

        risk_counts = self.recommendations['risk_level'].value_counts()
        ax2.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%')
        ax2.set_title('风险等级分布', fontproperties=chinese_font)

        plt.tight_layout()
        plt.savefig(os.path.join(self.directories['plots'], 'risk_distribution.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_threshold_sensitivity(self):
        """以中文标签绘制阈值敏感性分析"""
        if self.processed_data is None:
            return

        thresholds = self.config.sensitivity_thresholds
        reach_rates = []

        for threshold in thresholds:
            if 'qc_pass' in self.processed_data.columns:
                reach_condition = ((self.processed_data[self.config.y_frac_col] >= threshold) &
                                   self.processed_data['qc_pass'])
            else:
                reach_condition = self.processed_data[self.config.y_frac_col] >= threshold
            reach_rates.append(reach_condition.mean())

        plt.figure(figsize=(10, 6))
        chinese_font = self.utils.setup_chinese_font()

        plt.plot(thresholds, reach_rates, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Y-片段阈值', fontproperties=chinese_font)
        plt.ylabel('达标率', fontproperties=chinese_font)
        plt.title('阈值敏感性分析', fontproperties=chinese_font)
        plt.grid(True, alpha=0.3)

        # 高亮当前阈值
        if self.config.y_frac_threshold in thresholds:
            idx = thresholds.index(self.config.y_frac_threshold)
            plt.scatter(self.config.y_frac_threshold, reach_rates[idx],
                        s=100, label=f'当前阈值 ({self.config.y_frac_threshold})')
            plt.legend(prop=chinese_font)

        plt.tight_layout()
        plt.savefig(os.path.join(self.directories['plots'], 'threshold_sensitivity.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_bootstrap_results(self, bootstrap_results: Dict[str, Any]):
        """以中文标签绘制自助法（Bootstrap）分析结果"""
        if not bootstrap_results or 'bootstrap_samples' not in bootstrap_results:
            return

        plt.figure(figsize=(12, 5))
        chinese_font = self.utils.setup_chinese_font()

        # 分布图
        plt.subplot(1, 2, 1)
        plt.hist(bootstrap_results['bootstrap_samples'], bins=50, alpha=0.7, density=True)
        plt.axvline(bootstrap_results['mean'], linestyle='--',
                    label=f"均值: {bootstrap_results['mean']:.3f}")
        plt.axvline(bootstrap_results['ci_lower'], linestyle='--',
                    label=f"95%置信区间: [{bootstrap_results['ci_lower']:.3f}, {bootstrap_results['ci_upper']:.3f}]")
        plt.axvline(bootstrap_results['ci_upper'], linestyle='--')
        plt.xlabel('自助法达标率', fontproperties=chinese_font)
        plt.ylabel('密度', fontproperties=chinese_font)
        plt.title('自助法达标率分布', fontproperties=chinese_font)
        plt.legend(prop=chinese_font)

        # Q-Q 图
        plt.subplot(1, 2, 2)
        stats.probplot(bootstrap_results['bootstrap_samples'], dist="norm", plot=plt)
        plt.title('Q-Q图：自助法样本 vs 正态', fontproperties=chinese_font)

        plt.tight_layout()
        plt.savefig(os.path.join(self.directories['plots'], 'bootstrap_analysis.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_qc_waveforms(self, qc_analysis: Dict[str, Any]):
        """以中文标签绘制质控伯努利波形"""
        if not qc_analysis:
            return

        name_map = {
            'gc_pass': 'GC含量通过',
            'mapping_pass': '比对率通过',
            'filter_pass': '过滤率通过'
        }

        n_qc = len(qc_analysis)
        fig, axes = plt.subplots(n_qc, 1, figsize=(12, 4 * n_qc))
        if n_qc == 1:
            axes = [axes]

        chinese_font = self.utils.setup_chinese_font()

        for idx, (qc_name, qc_data) in enumerate(qc_analysis.items()):
            samples = qc_data['bernoulli_samples'][:200]
            axes[idx].plot(samples, 'o-', alpha=0.7, markersize=3)
            axes[idx].axhline(y=qc_data['p_estimate'], linestyle='--',
                              label=f"p = {qc_data['p_estimate']:.3f}")
            axes[idx].fill_between(range(len(samples)),
                                   qc_data['ci_lower'], qc_data['ci_upper'],
                                   alpha=0.2, label='95%置信区间')
            axes[idx].set_ylabel('质控通过 (0/1)', fontproperties=chinese_font)
            axes[idx].set_title(f'{name_map.get(qc_name, qc_name)} 波形', fontproperties=chinese_font)
            axes[idx].legend(prop=chinese_font)
            axes[idx].grid(True, alpha=0.3)

        axes[-1].set_xlabel('样本索引', fontproperties=chinese_font)
        plt.tight_layout()
        plt.savefig(os.path.join(self.directories['plots'], 'qc_waveforms.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def run_enhanced_analysis(self) -> Dict[str, Any]:
        """Run enhanced analysis with proper time-to-event and BMI grouping."""

        # Step 1: Create survival dataset
        survival_df = self.data_processor.create_survival_dataset(self.processed_data)

        # Step 2: Train corrected survival model
        survival_result = self.modeling_engine.train_survival_model_corrected(self.processed_data)
        if survival_result:
            self.model_results['cox_survival_corrected'] = survival_result

        # Step 3: BMI group analysis
        bmi_analyzer = BMIGroupAnalyzer(self.modeling_engine, self.config)
        bmi_results = bmi_analyzer.find_optimal_week_by_bmi(survival_df)

        # Step 4: Detection error analysis
        error_analyzer = DetectionErrorAnalyzer(self.config)
        error_results = error_analyzer.evaluate_error_impact(
            bmi_results, self.modeling_engine, survival_df
        )

        # Step 5: Save enhanced results
        if not bmi_results.empty:
            bmi_results.to_csv(
                os.path.join(self.directories['main'], 'optimal_weeks_by_bmi.csv'),
                index=False
            )

        if error_results['error_analysis'] is not None:
            error_results['error_analysis'].to_csv(
                os.path.join(self.directories['main'], 'detection_error_analysis.csv'),
                index=False
            )

        return {
            'survival_dataset': survival_df,
            'bmi_optimal_weeks': bmi_results,
            'error_analysis': error_results
        }

    def _save_results(self):
        """Save all analysis results to files."""
        print("[INFO] Saving analysis results...")

        # Save model performance summary
        if self.model_results:
            model_summary = []
            for name, result in self.model_results.items():
                summary = {
                    'model_name': name,
                    'cross_val_score': result.cross_val_score,
                    'n_predictions': len(result.predicted_weeks),
                    'mean_predicted_week': np.mean(result.predicted_weeks) if len(
                        result.predicted_weeks) > 0 else np.nan,
                    'std_predicted_week': np.std(result.predicted_weeks) if len(result.predicted_weeks) > 0 else np.nan,
                    'mean_risk_score': np.mean(result.risk_scores) if len(result.risk_scores) > 0 else np.nan
                }

                # Add top 3 features
                if result.feature_importance:
                    top_features = sorted(result.feature_importance.items(),
                                          key=lambda x: x[1], reverse=True)[:3]
                    for i, (feature, importance) in enumerate(top_features):
                        summary[f'top_{i + 1}_feature'] = feature
                        summary[f'top_{i + 1}_importance'] = importance

                model_summary.append(summary)

            pd.DataFrame(model_summary).to_csv(
                os.path.join(self.directories['main'], 'model_performance_summary.csv'),
                index=False
            )

        # Save recommendations
        if self.recommendations is not None:
            self.recommendations.to_csv(
                os.path.join(self.directories['main'], 'personalized_recommendations.csv'),
                index=False
            )

        # Save threshold sensitivity
        self._save_threshold_sensitivity()

        # Save processed data summary
        if self.processed_data is not None:
            summary_stats = {
                'total_records': len(self.processed_data),
                'unique_mothers': self.processed_data[self.config.mother_id].nunique(),
                'reach_rate': self.processed_data['reach'].mean(),
                'mean_gestational_week': self.processed_data[self.config.week_col].mean(),
                'std_gestational_week': self.processed_data[self.config.week_col].std()
            }

            with open(os.path.join(self.directories['main'], 'data_summary.json'), 'w') as f:
                json.dump(summary_stats, f, indent=2)

    def _save_threshold_sensitivity(self):
        """Save threshold sensitivity analysis results."""
        if self.processed_data is None:
            return

        sensitivity_results = []
        for threshold in self.config.sensitivity_thresholds:
            if 'qc_pass' in self.processed_data.columns:
                reach_condition = ((self.processed_data[self.config.y_frac_col] >= threshold) &
                                   self.processed_data['qc_pass'])
            else:
                reach_condition = self.processed_data[self.config.y_frac_col] >= threshold

            sensitivity_results.append({
                'threshold': threshold,
                'reach_rate': reach_condition.mean(),
                'n_reached': reach_condition.sum(),
                'total_n': len(self.processed_data)
            })

        pd.DataFrame(sensitivity_results).to_csv(
            os.path.join(self.directories['main'], 'threshold_sensitivity.csv'),
            index=False
        )

    def _bayesian_threshold_analysis(self) -> Dict[str, Any]:
        """Bayesian analysis for threshold uncertainty."""
        if self.processed_data is None:
            return {}

        # Create threshold range
        thresholds = np.arange(
            self.config.threshold_range[0],
            self.config.threshold_range[1],
            self.config.threshold_step
        )

        results = []

        for threshold in thresholds:
            # Calculate reach rate for this threshold
            if 'qc_pass' in self.processed_data.columns:
                reach_condition = ((self.processed_data[self.config.y_frac_col] >= threshold) &
                                   self.processed_data['qc_pass'])
            else:
                reach_condition = self.processed_data[self.config.y_frac_col] >= threshold

            reach_rate = reach_condition.mean()
            n_reached = reach_condition.sum()
            n_total = len(self.processed_data)

            # Bayesian posterior (Beta distribution)
            posterior_alpha = self.config.threshold_prior_alpha + n_reached
            posterior_beta = self.config.threshold_prior_beta + n_total - n_reached

            # Posterior statistics
            posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
            posterior_var = (posterior_alpha * posterior_beta) / \
                            ((posterior_alpha + posterior_beta) ** 2 * (posterior_alpha + posterior_beta + 1))

            results.append({
                'threshold': threshold,
                'observed_rate': reach_rate,
                'posterior_mean': posterior_mean,
                'posterior_std': np.sqrt(posterior_var),
                'credible_interval_lower': stats.beta.ppf(0.025, posterior_alpha, posterior_beta),
                'credible_interval_upper': stats.beta.ppf(0.975, posterior_alpha, posterior_beta)
            })

        return {
            'threshold_analysis': pd.DataFrame(results),
            'optimal_threshold': thresholds[np.argmax([r['posterior_mean'] for r in results])]
        }

    def _generate_comprehensive_report(self):
        """Generate comprehensive analysis report."""
        print("[INFO] Generating comprehensive report...")

        report_content = []
        report_content.append("# NIPT Enhanced Analysis Report\n")
        report_content.append("=" * 50 + "\n\n")

        # Executive Summary
        report_content.append("## Executive Summary\n")
        if self.processed_data is not None:
            total_records = len(self.processed_data)
            unique_mothers = self.processed_data[self.config.mother_id].nunique()
            reach_rate = self.processed_data['reach'].mean()

            report_content.append(f"- **Total Records Analyzed**: {total_records:,}\n")
            report_content.append(f"- **Unique Patients**: {unique_mothers:,}\n")
            report_content.append(f"- **Overall Reach Rate**: {reach_rate:.1%}\n")

            if self.config.week_col in self.processed_data.columns:
                week_stats = self.processed_data[self.config.week_col].describe()
                report_content.append(
                    f"- **Gestational Week Range**: {week_stats['min']:.1f} - {week_stats['max']:.1f} weeks\n")
                report_content.append(
                    f"- **Mean Gestational Week**: {week_stats['mean']:.1f} ± {week_stats['std']:.1f} weeks\n")

        report_content.append("\n")

        # Model Performance
        report_content.append("## Model Performance\n")
        if self.model_results:
            for name, result in self.model_results.items():
                report_content.append(f"### {name.replace('_', ' ').title()}\n")
                report_content.append(f"- **Cross-Validation Score**: {result.cross_val_score:.3f}\n")
                report_content.append(f"- **Number of Predictions**: {len(result.predicted_weeks):,}\n")

                if len(result.predicted_weeks) > 0:
                    mean_week = np.mean(result.predicted_weeks)
                    std_week = np.std(result.predicted_weeks)
                    report_content.append(f"- **Mean Predicted Week**: {mean_week:.1f} ± {std_week:.1f}\n")

                if result.feature_importance:
                    top_features = sorted(result.feature_importance.items(),
                                          key=lambda x: x[1], reverse=True)[:5]
                    report_content.append("- **Top 5 Important Features**:\n")
                    for feature, importance in top_features:
                        report_content.append(f"  - {feature}: {importance:.3f}\n")

                report_content.append("\n")

        # Personalized Recommendations
        report_content.append("## Personalized Recommendations\n")
        if self.recommendations is not None and len(self.recommendations) > 0:
            risk_distribution = self.recommendations['risk_level'].value_counts()
            report_content.append("### Risk Level Distribution\n")
            for risk_level, count in risk_distribution.items():
                percentage = (count / len(self.recommendations)) * 100
                report_content.append(f"- **{risk_level}**: {count:,} patients ({percentage:.1f}%)\n")

            # Optimal week statistics
            if 'optimal_week' in self.recommendations.columns:
                week_stats = self.recommendations['optimal_week'].describe()
                report_content.append(f"\n### Optimal Testing Week Statistics\n")
                report_content.append(f"- **Mean Optimal Week**: {week_stats['mean']:.1f} weeks\n")
                report_content.append(f"- **Median Optimal Week**: {week_stats['50%']:.1f} weeks\n")
                report_content.append(f"- **Range**: {week_stats['min']:.1f} - {week_stats['max']:.1f} weeks\n")

        # Key Findings
        report_content.append("\n## Key Findings\n")
        if self.model_results:
            best_model = max(self.model_results.items(), key=lambda x: x[1].cross_val_score)
            report_content.append(f"- **Best Performing Model**: {best_model[0].replace('_', ' ').title()} ")
            report_content.append(f"(CV Score: {best_model[1].cross_val_score:.3f})\n")

            if best_model[1].feature_importance:
                most_important = max(best_model[1].feature_importance.items(), key=lambda x: x[1])
                report_content.append(f"- **Most Important Feature**: {most_important[0]} ")
                report_content.append(f"(Importance: {most_important[1]:.3f})\n")

        # Clinical Recommendations
        report_content.append("\n## Clinical Recommendations\n")
        report_content.append("Based on the comprehensive analysis, we recommend:\n\n")
        report_content.append("1. **Personalized Testing Schedules**: Implement individualized testing ")
        report_content.append(
            "schedules based on patient-specific risk factors rather than fixed gestational weeks.\n\n")
        report_content.append("2. **Enhanced Monitoring for High-Risk Patients**: Patients identified as high-risk ")
        report_content.append("should receive earlier and more frequent monitoring.\n\n")
        report_content.append("3. **Multi-Factor Assessment**: Utilize multiple biological and clinical indicators ")
        report_content.append("for comprehensive risk assessment.\n\n")
        report_content.append("4. **Quality Control Optimization**: Implement dynamic quality control standards ")
        report_content.append("based on patient characteristics and testing conditions.\n\n")
        report_content.append("5. **Continuous Model Validation**: Regularly update and validate prediction models ")
        report_content.append("with new data to maintain accuracy.\n")

        # Technical Summary
        report_content.append("\n## Technical Summary\n")
        report_content.append("### Methods Used\n")
        report_content.append("- **Machine Learning Models**: Logistic Regression, Random Forest")
        if TORCH_AVAILABLE:
            report_content.append(", Neural Networks")
        if SURVIVAL_AVAILABLE:
            report_content.append(", Cox Proportional Hazards")
        report_content.append("\n")
        report_content.append("- **Feature Engineering**: Comprehensive feature creation including ")
        report_content.append("categorical variables, quality scores, and derived metrics\n")
        report_content.append("- **Cross-Validation**: 5-fold stratified cross-validation for model evaluation\n")
        report_content.append("- **Ensemble Methods**: Weighted ensemble predictions based on model performance\n")

        # Save report
        report_path = os.path.join(self.directories['reports'], 'comprehensive_analysis_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.writelines(report_content)

        print(f"[INFO] Comprehensive report saved to: {report_path}")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution function."""
    # Create configuration
    config = NIPTConfig()

    # Initialize and run analyzer
    analyzer = NIPTAnalyzer(config)
    results = analyzer.run_complete_analysis()

    if results:
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        # Print summary statistics
        if 'processed_data' in results and results['processed_data'] is not None:
            data = results['processed_data']
            print(f"\nKey Statistics:")
            print(f"- Total Records: {len(data):,}")
            print(f"- Unique Patients: {data[config.mother_id].nunique():,}")
            print(f"- Reach Rate: {data['reach'].mean():.1%}")

        if 'model_results' in results and results['model_results']:
            models = results['model_results']
            print(f"- Models Trained: {len(models)}")
            best_model = max(models.items(), key=lambda x: x[1].cross_val_score)
            print(f"- Best Model: {best_model[0]} (Score: {best_model[1].cross_val_score:.3f})")

        if 'recommendations' in results and results['recommendations'] is not None:
            recs = results['recommendations']
            print(f"- Recommendations Generated: {len(recs):,}")

        print(f"\nAll results saved to: {config.output_dir}")
        print("\nKey Output Files:")
        print("- comprehensive_analysis_report.md")
        print("- model_performance_summary.csv")
        print("- personalized_recommendations.csv")
        print("- threshold_sensitivity.csv")
        print("- Various visualization plots in /plots/")

    else:
        print("\n[ERROR] Analysis failed. Please check the logs for details.")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')

    # Set random seeds for reproducibility
    np.random.seed(2025)
    if TORCH_AVAILABLE:
        torch.manual_seed(2025)

    # Run main analysis
    main()