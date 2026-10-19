"""Feature engineering and statistical feature-analysis mechanisms."""

from mlzoomcamp.core.features.analysis import (
    binary_target_group_summary,
    categorical_mutual_information,
    correlation_matrix,
    find_correlated_feature_pairs,
    numeric_target_correlations,
)
from mlzoomcamp.core.features.engineering import add_log1p_feature, years_to_age

__all__ = [
    "add_log1p_feature",
    "binary_target_group_summary",
    "categorical_mutual_information",
    "correlation_matrix",
    "find_correlated_feature_pairs",
    "numeric_target_correlations",
    "years_to_age",
]
