from pyspark.sql import DataFrame
from pyspark.sql import functions as F


class Substitution:
    """
    Applies masking to a column by replacing its values fully or partially
    with a specified character.

    The transformation supports two substitution modes:

    - Full substitution:
        Replaces every character of the original value while preserving
        its original length.

    - Partial substitution:
        Replaces a specific portion of the original value, preserving the
        characters outside the configured substitution range.

    Parameters
    ----------
    column : str
        Name of the column to be masked.

    replacement_char : str, default="*"
        Single character used to replace characters from the original
        values.

    mode : str, default="full"
        Substitution mode.

        Supported values are:

        - ``"full"``:
            Replaces the complete value.

        - ``"partial"``:
            Replaces only a configured portion of the value.

    start : int, optional
        Starting position, using a 0-based index, for partial substitution.

        Required when ``mode="partial"``.

    length : int, optional
        Maximum number of characters to replace starting from ``start``.

        Required when ``mode="partial"``.

    Notes
    -----
    The transformation is implemented entirely using Spark SQL
    expressions. No records are collected to the driver and no Python UDF
    is used, allowing the operation to be executed in a distributed manner.

    Null values are preserved.

    Values are converted to strings before masking. Therefore, non-null
    values in the transformed column are returned as strings.
    """

    VALID_MODES = {
        "full",
        "partial"
    }

    def __init__(
        self,
        column: str,
        replacement_char: str = "*",
        mode: str = "full",
        start: int = None,
        length: int = None
    ):
        """
        Initializes the substitution transformation.

        Parameters
        ----------
        column : str
            Name of the column to be masked.

        replacement_char : str, default="*"
            Single character used to replace the original values.

        mode : str, default="full"
            Substitution mode.

            Supported values are:

            - ``"full"``:
                Replaces the entire value.

            - ``"partial"``:
                Replaces only a portion of the value.

        start : int, optional
            Starting position, using a 0-based index, for partial
            substitution.

            Required when ``mode="partial"``.

        length : int, optional
            Number of characters to replace starting from ``start``.

            Required when ``mode="partial"``.

        Raises
        ------
        ValueError
            If the column name is invalid, the replacement character does
            not contain exactly one character, the mode is unsupported, or
            the parameters required by the selected mode are invalid.
        """
        if not isinstance(column, str) or not column.strip():
            raise ValueError(
                "'column' must be a non-empty string."
            )

        if (
            not isinstance(replacement_char, str)
            or len(replacement_char) != 1
        ):
            raise ValueError(
                "'replacement_char' must be a single character."
            )

        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Unsupported mode '{mode}'. "
                f"Use {sorted(self.VALID_MODES)}."
            )

        self._validate_mode_parameters(
            mode,
            start,
            length
        )

        self.column = column
        self.replacement_char = replacement_char
        self.mode = mode
        self.start = start
        self.length = length

    @staticmethod
    def _validate_mode_parameters(
        mode: str,
        start: int | None,
        length: int | None
    ) -> None:
        """
        Validates parameters specific to the selected substitution mode.

        Parameters
        ----------
        mode : str
            Substitution mode. Expected values are ``"full"`` or
            ``"partial"``.

        start : int or None
            Starting position for partial substitution.

        length : int or None
            Number of characters to replace in partial substitution.

        Returns
        -------
        None
            The method only validates the provided parameters.

        Raises
        ------
        ValueError
            If ``start`` or ``length`` are provided in full mode, if they
            are missing in partial mode, if they are not integers, or if
            their values are outside the valid range.
        """
        if mode == "full":
            if start is not None or length is not None:
                raise ValueError(
                    "'start' and 'length' must not be provided "
                    "in 'full' mode."
                )

            return

        if start is None or length is None:
            raise ValueError(
                "In 'partial' mode, 'start' and 'length' "
                "must be provided."
            )

        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(length, int)
            or isinstance(length, bool)
        ):
            raise ValueError(
                "'start' and 'length' must be integers."
            )

        if start < 0:
            raise ValueError(
                "'start' must be greater than or equal to 0."
            )

        if length <= 0:
            raise ValueError(
                "'length' must be greater than 0."
            )

    def transform(
        self,
        df: DataFrame
    ) -> DataFrame:
        """
        Applies the configured substitution to the selected column.

        In full mode, every character of each non-null value is replaced
        with ``replacement_char`` while preserving the original value
        length.

        In partial mode, substitution starts at the configured 0-based
        position and replaces at most ``length`` characters. Characters
        before and after the selected region are preserved.

        If the requested substitution range extends beyond the end of the
        value, only the available characters are replaced.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input dataset containing the column to be masked.

        Returns
        -------
        pyspark.sql.DataFrame
            New DataFrame with the selected column masked.

        Raises
        ------
        ValueError
            If the input DataFrame is None, is not Spark-compatible, or
            does not contain the configured column.

        Notes
        -----
        Null values are preserved.

        Non-null values are converted to strings before substitution, so
        the resulting transformed values are strings.

        The transformation uses only native Spark SQL expressions and is
        therefore evaluated in a distributed manner without collecting
        dataset records to the driver.
        """
        if df is None:
            raise ValueError(
                "Input DataFrame cannot be None."
            )

        # Compatible with standard Spark and Spark Connect.
        if not hasattr(df, "withColumn"):
            raise ValueError(
                "Input must be a Spark-compatible DataFrame."
            )

        if self.column not in df.columns:
            raise ValueError(
                f"The column '{self.column}' does not exist "
                f"in the DataFrame."
            )

        source_col = F.col(self.column)
        string_col = source_col.cast("string")
        value_length = F.length(string_col)

        # ------------------------------------------------------
        # FULL SUBSTITUTION
        # ------------------------------------------------------

        if self.mode == "full":
            masked_value = F.repeat(
                F.lit(self.replacement_char),
                value_length
            )

            return df.withColumn(
                self.column,
                F.when(
                    source_col.isNull(),
                    F.lit(None).cast("string")
                ).otherwise(
                    masked_value
                )
            )

        # ------------------------------------------------------
        # PARTIAL SUBSTITUTION
        # ------------------------------------------------------

        replacement_length = F.greatest(
            F.lit(0),
            F.least(
                F.lit(self.length),
                value_length - F.lit(self.start)
            )
        )

        prefix = F.substring(
            string_col,
            1,
            self.start
        )

        masked_part = F.repeat(
            F.lit(self.replacement_char),
            replacement_length
        )

        suffix = F.substring(
            string_col,
            self.start + self.length + 1,
            value_length
        )

        masked_value = F.concat(
            prefix,
            masked_part,
            suffix
        )

        return df.withColumn(
            self.column,
            F.when(
                source_col.isNull(),
                F.lit(None).cast("string")
            ).otherwise(
                masked_value
            )
        )