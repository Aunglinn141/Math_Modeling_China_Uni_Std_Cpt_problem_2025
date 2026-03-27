# NIPT Enhanced Analysis Report
==================================================

## Executive Summary
- **Total Records Analyzed**: 1,082
- **Unique Patients**: 267
- **Overall Reach Rate**: 50.6%
- **Gestational Week Range**: 11.0 - 29.0 weeks
- **Mean Gestational Week**: 16.8 ± 4.1 weeks

## Model Performance
### Logistic Regression
- **Cross-Validation Score**: 0.917
- **Number of Predictions**: 1,082
- **Mean Predicted Week**: 17.4 ± 5.2
- **Top 5 Important Features**:
  - gc_total: 2.869
  - ivf_method_IVF（试管婴儿）: 0.622
  - weight: 0.452
  - ivf_method_自然受孕: 0.395
  - parity_3.0: 0.339

### Random Forest
- **Cross-Validation Score**: 0.941
- **Number of Predictions**: 1,082
- **Mean Predicted Week**: 17.4 ± 6.8
- **Top 5 Important Features**:
  - gc_total: 0.701
  - reads_align_ratio: 0.056
  - bmi: 0.053
  - weight: 0.052
  - gestation_weeks: 0.049

### Neural Network
- **Cross-Validation Score**: 0.850
- **Number of Predictions**: 1,082
- **Mean Predicted Week**: 17.4 ± 6.4
- **Top 5 Important Features**:
  - gc_total: 0.416
  - ivf_method_IVF（试管婴儿）: 0.138
  - parity_3.0: 0.114
  - ivf_method_IUI（人工授精）: 0.085
  - parity_2.0: 0.077

### Cox Survival Corrected
- **Cross-Validation Score**: 0.800
- **Number of Predictions**: 0
- **Top 5 Important Features**:
  - gc_total: 108.937
  - reads_align_ratio: 6.882
  - gestation_weeks: 0.444
  - bmi: 0.210
  - weight: 0.103

## Personalized Recommendations
### Risk Level Distribution
- **高风险**: 534 patients (49.4%)
- **低风险**: 445 patients (41.1%)
- **中风险**: 103 patients (9.5%)

### Optimal Testing Week Statistics
- **Mean Optimal Week**: 17.1 weeks
- **Median Optimal Week**: 14.6 weeks
- **Range**: 10.1 - 25.0 weeks

## Key Findings
- **Best Performing Model**: Random Forest (CV Score: 0.941)
- **Most Important Feature**: gc_total (Importance: 0.701)

## Clinical Recommendations
Based on the comprehensive analysis, we recommend:

1. **Personalized Testing Schedules**: Implement individualized testing schedules based on patient-specific risk factors rather than fixed gestational weeks.

2. **Enhanced Monitoring for High-Risk Patients**: Patients identified as high-risk should receive earlier and more frequent monitoring.

3. **Multi-Factor Assessment**: Utilize multiple biological and clinical indicators for comprehensive risk assessment.

4. **Quality Control Optimization**: Implement dynamic quality control standards based on patient characteristics and testing conditions.

5. **Continuous Model Validation**: Regularly update and validate prediction models with new data to maintain accuracy.

## Technical Summary
### Methods Used
- **Machine Learning Models**: Logistic Regression, Random Forest, Neural Networks, Cox Proportional Hazards
- **Feature Engineering**: Comprehensive feature creation including categorical variables, quality scores, and derived metrics
- **Cross-Validation**: 5-fold stratified cross-validation for model evaluation
- **Ensemble Methods**: Weighted ensemble predictions based on model performance
