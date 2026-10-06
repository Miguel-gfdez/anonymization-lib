from pyspark.sql import DataFrame
from pyspark.sql import functions as F


class Suppression:
    """
    Applies suppression techniques to one or multiple columns.

    Suppression removes or hides information from selected attributes in
    order to reduce the amount of identifying information available in
    the dataset.

    Two suppression modes are supported:

    - Null suppression:
        Replaces every value in the selected column with NULL while
        preserving the original Spark data type.

    - Drop suppression:
        Removes the selected column completely from the dataset.

    Parameters
    ----------
    columns_modes : dict
        Dictionary where keys are column names and values specify the
        suppression mode to apply.

        Supported modes are:

        - ``"null"``:
            Replaces all values in the column with NULL.

        - ``"drop"``:
            Removes the column from the dataset.

        Example::

            {
                "age": "null",
                "name": "drop"
            }

    Notes
    -----
    The transformation is implemented entirely using native Spark
    DataFrame operations.

    No records are collected to the driver and no Python UDF is used,
    allowing suppression to be executed in a distributed manner.

    Columns suppressed using ``"null"`` preserve their original Spark
    data type.
    """

    VALID_MODES = {
        "null",
        "drop"
    }

    def __init__(
        self,
        columns_modes: dict
    ):
        """
        Initializes the suppression transformation.

        Parameters
        ----------
        columns_modes : dict
            Dictionary where keys are column names and values are
            suppression modes.

            Supported modes are:

            - ``"null"``:
                Replaces all values in the column with NULL while
                preserving the original column data type.

            - ``"drop"``:
                Removes the column from the dataset.

            Example::

                {
                    "age": "null",
                    "name": "drop"
                }

        Raises
        ------
        ValueError
            If ``columns_modes`` is not a dictionary, is empty, contains
            invalid or empty column names, or contains unsupported
            suppression modes.
        """
        if not isinstance(columns_modes, dict):
            raise ValueError(
                "'columns_modes' must be a dictionary."
            )

        if not columns_modes:
            raise ValueError(
                "'columns_modes' must not be empty."
            )

        if not all(
            isinstance(column, str) and column.strip()
            for column in columns_modes
        ):
            raise ValueError(
                "All column names must be non-empty strings."
            )

        if not all(
            isinstance(mode, str)
            for mode in columns_modes.values()
        ):
            raise ValueError(
                "All suppression modes must be strings."
            )

        for column, mode in columns_modes.items():
            if mode not in self.VALID_MODES:
                raise ValueError(
                    f"Invalid mode '{mode}' for column '{column}'. "
                    f"Use {sorted(self.VALID_MODES)}."
                )

        self.columns_modes = columns_modes.copy()

    def transform(
        self,
        df: DataFrame
    ) -> DataFrame:
        """
        Applies the configured suppression operations to the input dataset.

        Columns configured with ``"drop"`` are removed from the resulting
        DataFrame.

        Columns configured with ``"null"`` are preserved but every value
        is replaced with NULL. The original Spark data type of each column
        is maintained.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input dataset containing the columns to suppress.

        Returns
        -------
        pyspark.sql.DataFrame
            New DataFrame with the configured suppression operations
            applied.

        Raises
        ------
        ValueError
            If the input DataFrame is None, is not Spark-compatible, or
            does not contain one or more of the configured columns.

        Notes
        -----
        All configured columns are validated before any transformation is
        constructed.

        Drop suppression is applied to all selected columns in a single
        Spark ``drop`` operation.

        Null suppression is applied through a single projection, avoiding
        the construction of multiple consecutive ``withColumn``
        transformations.

        The transformation is lazy and uses only native Spark expressions.
        Dataset records are not collected to the driver.
        """
        if df is None:
            raise ValueError(
                "Input DataFrame cannot be None."
            )

        # Compatible with standard Spark and Spark Connect.
        if not hasattr(df, "select"):
            raise ValueError(
                "Input must be a Spark-compatible DataFrame."
            )

        # ------------------------------------------------------
        # VALIDATE COLUMNS
        # ------------------------------------------------------

        missing_columns = [
            column
            for column in self.columns_modes
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                "Columns not found in DataFrame: "
                + ", ".join(missing_columns)
            )

        # ------------------------------------------------------
        # SEPARATE SUPPRESSION MODES
        # ------------------------------------------------------

        drop_columns = [
            column
            for column, mode in self.columns_modes.items()
            if mode == "drop"
        ]

        null_columns = {
            column
            for column, mode in self.columns_modes.items()
            if mode == "null"
        }

        # ------------------------------------------------------
        # NULL SUPPRESSION
        # ------------------------------------------------------

        if null_columns:
            schema_types = {
                field.name: field.dataType
                for field in df.schema.fields
                if field.name in null_columns
            }

            result_df = df.select(
                *[
                    (
                        F.lit(None)
                        .cast(schema_types[column])
                        .alias(column)
                        if column in null_columns
                        else F.col(column)
                    )
                    for column in df.columns
                ]
            )
        else:
            result_df = df

        # ------------------------------------------------------
        # DROP SUPPRESSION
        # ------------------------------------------------------

        if drop_columns:
            result_df = result_df.drop(
                *drop_columns
            )

        return result_df