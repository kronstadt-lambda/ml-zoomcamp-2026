import pandas as pd


def years_to_age(
    df: pd.DataFrame,
    year_column: str,
    reference_year: int,
    output_column: str,
    *,
    drop_year_column: bool = False,
) -> pd.DataFrame:
    """Create an age feature from a year-based column."""

    if year_column not in df.columns:
        raise ValueError(
            f"Year column '{year_column}' not found"
        )

    transformed = df.copy()

    transformed[output_column] = (
        reference_year - transformed[year_column]
    )

    if drop_year_column:
        transformed = transformed.drop(columns=year_column)

    return transformed
