from pyspark.sql import DataFrame
from pyspark.sql import functions as F


class Substitution:
    """
    Applies masking to a column by replacing its values fully or partially
    with a specified character.
    """
    def __init__(self, column: str, replacement_char: str = "*", mode: str = "full", start: int = None, length: int = None):
        """
        Initializes the substitution transformation.

        Parameters
        ----------
        column : str
            Name of the column to be masked.

        replacement_char : str, default="*"
            Single character used to replace the original values.

        mode : str, default="full"
            Substitution mode:
            - 'full': replaces the entire value.
            - 'partial': replaces only a portion of the value.

        start : int, optional
            Starting position (0-based index) for partial substitution.
            Required when mode is 'partial'.

        length : int, optional
            Number of characters to replace starting from 'start'.
            Required when mode is 'partial'.

        """
        if not isinstance(column, str) or not column.strip():
            raise ValueError("'column' must be a non-empty string.")

        if not isinstance(replacement_char, str) or len(replacement_char) != 1:
            raise ValueError("The replacement character must be a single character.")

        if mode not in ("full", "partial"):
            raise ValueError("Mode must be 'full' or 'partial'.")

        if mode == "partial":
            if start is None or length is None:
                raise ValueError("In 'partial' mode, 'start' and 'length' must be provided.")
            
            if not isinstance(start, int) or not isinstance(length, int):
                raise ValueError("'start' and 'length' must be integers.")

            if start < 0 or length <= 0:
                raise ValueError("'start' must be >= 0 and 'length' must be > 0.")

        if mode == "full" and (start is not None or length is not None):
            raise ValueError("'start' and 'length' must not be provided in 'full' mode.")

        self.column = column
        self.replacement_char = replacement_char
        self.mode = mode
        self.start = start
        self.length = length

    def transform(self, df: DataFrame) -> DataFrame:
        """
        Applies the substitution to the specified column and returns a new DataFrame.
        """
        if df is None:
            raise ValueError("Input DataFrame cannot be None.")

        if not isinstance(df, DataFrame):
            raise ValueError("Input must be a pyspark.sql.DataFrame.")

        if self.column not in df.columns:
            raise ValueError(f"The column '{self.column}' does not exist in the DataFrame.")

        if self.mode == "full":
            return df.withColumn(
                self.column,
                F.when(
                    F.col(self.column).isNull(),
                    F.col(self.column)
                ).otherwise(
                    F.repeat(
                        F.lit(self.replacement_char),
                        F.length(F.col(self.column).cast("string"))
                    )
                )
            )

        return df.withColumn(
            self.column,
            F.when(
                F.col(self.column).isNull(),
                F.col(self.column)
            ).otherwise(
                F.concat(
                    F.substring(F.col(self.column).cast("string"), 1, self.start),
                    F.repeat(
                        F.lit(self.replacement_char),
                        F.when(
                            F.length(F.col(self.column).cast("string")) - self.start < self.length,
                            F.length(F.col(self.column).cast("string")) - self.start
                        ).otherwise(self.length)
                    ),
                    F.substring(
                        F.col(self.column).cast("string"),
                        self.start + self.length + 1,
                        F.length(F.col(self.column).cast("string"))
                    )
                )
            )
        )