from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def get_dtype(df: DataFrame, column: str) -> str:
    """
    Returns the Spark data type of a column.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
    column : str

    Returns
    -------
    str
        Data type of the column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in DataFrame.")

    return dict(df.dtypes)[column]


def is_numeric(dtype: str) -> bool:
    """
    Checks if a dtype corresponds to a numeric type.
    """
    return (
        dtype in [
            "int", "bigint", "double", "float",
            "long", "smallint", "tinyint"
        ]
        or dtype.startswith("decimal")
    )


def is_date(dtype: str) -> bool:
    """
    Checks if a dtype corresponds to a date or timestamp.
    """
    return dtype in ["date", "timestamp"]


def is_boolean(dtype: str) -> bool:
    """
    Checks if a dtype corresponds to a boolean type.
    """
    return dtype == "boolean"


def infer_semantic_type(df: DataFrame, column: str) -> str:
    """
    Infers the semantic type of a column.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
    column : str

    Returns
    -------
    str
        One of: 'numeric', 'date', 'categorical'
    """
    dtype = get_dtype(df, column)

    if is_numeric(dtype):
        return "numeric"

    if is_date(dtype):
        return "date"

    if is_boolean(dtype):
        return "categorical"

    return "categorical"




class EquivalenceGroups:
    """
    Base class providing common utilities for building and validating
    equivalence groups based on quasi-identifiers in a dataset.
    """
    def _validate_columns(self, df: DataFrame, columns: list):
        """
        Validates that the provided columns exist in the DataFrame.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input dataset.
        columns : list of str
            List of column names to validate.

        Raises
        ------
        ValueError
            If the list of columns is empty or if any column does not exist
            in the DataFrame.
        """
        if not columns:
            raise ValueError("At least one column must be provided.")

        missing_cols = [c for c in columns if c not in df.columns]
        if missing_cols:
            raise ValueError(f"The following columns do not exist in the DataFrame: {missing_cols}")

    def _build_equivalence_groups(self,df: DataFrame,quasi_identifiers: list,extra_aggs: list = None) -> DataFrame:
        """
        Builds equivalence groups based on the provided quasi-identifiers.

        Each group represents a unique combination of quasi-identifier values,
        and includes the size of the group. Additional aggregations can be
        optionally applied (e.g., diversity of a sensitive attribute).

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input dataset.
        quasi_identifiers : list of str
            Columns used to define equivalence groups.
        extra_aggs : list of pyspark.sql.Column, optional
            Additional aggregation expressions to apply within each group.
            For example:
            - F.countDistinct("sensitive_col").alias("l_diversity")

        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame containing one row per equivalence group, with:
            - quasi-identifier columns
            - 'group_size' (number of records in the group)
            - any additional aggregation columns provided

        Notes
        -----
        This method is used as a common building block for privacy metrics
        such as k-anonymity, l-diversity and t-closeness.
        """
        aggs = [F.count("*").alias("group_size")]

        if extra_aggs is not None:
            aggs.extend(extra_aggs)

        return df.groupBy(*quasi_identifiers).agg(*aggs)





