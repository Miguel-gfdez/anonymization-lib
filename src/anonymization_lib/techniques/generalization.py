from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from ..utils import infer_semantic_type
import warnings


class Generalization:
    """
    Applies generalization to a column in a Spark DataFrame.

    The transformation supports:
    - Categorical mapping using external rules (value -> generalized value)
    - Numerical generalization using interval rules (start;end -> generalized value)
    - Temporal aggregation (year, month, quarter, semester), optionally preserving the year

    """

    def __init__(self,column: str,rules_path: str = None,mode: str = None,default_value: str = None,include_year: bool = True,output_column: str=None):
        """
        Initializes the generalization transformation.

        Parameters
        ----------
        column : str
            Name of the column to be generalized.

        rules_path : str
            Path to a text file containing generalization rules. 
            Required for categorical and numerical generalization. 
            
            The expected formats are:
            - Categorical: value_original;value_generalized
            - Numerical: start;end;value_generalized

        mode : str
            Temporal generalization mode. Used only for date/time columns.
            Required only when the target column is of type date or timestamp.

            Supported values are:
            - 'year'
            - 'month'
            - 'quarter'
            - 'semester'

        default_value : str, default=Null
            Value assigned when no rule applies or when a value falls outside
            the defined intervals.

        include_year : bool, default=True
            If True (default), temporal generalization preserves the year
            (e.g., 2024-05, 2024-Q2, 2024-S1). If False, values are grouped
            only by the temporal period (e.g., 05, Q2, S1).
        
        output_column : str, optional
            Name of the column after generalization. 
            If not provided, the original column is overwritten. 
            If provided, the original column is replaced and renamed to the specified name.
        """
        if not isinstance(column, str) or not column.strip():
            raise ValueError("'column' must be a non-empty string.")
        
        if output_column is not None:
            if not isinstance(output_column, str) or not output_column.strip():
                raise ValueError("'output_column' must be a non-empty string.")


        self.column = column
        self.rules_path = rules_path
        self.mode = mode
        self.default_value = default_value
        self.include_year = include_year
        self.output_column = output_column or column

    def _rename_output_column(self, df: DataFrame) -> DataFrame:
        if self.output_column is not None and self.output_column != self.column:
            return df.withColumnRenamed(self.column, self.output_column)
        return df

    def _apply_categorical(self, df: DataFrame) -> DataFrame:
        """
        Applies categorical generalization using mapping rules.

        Returns
        -------
        pyspark.sql.DataFrame
        """

        mapping = {}

        with open(self.rules_path, "r") as f:
            for line in f:
                parts = line.strip().split(";")

                if len(parts) != 2:
                    warnings.warn(f"Invalid line ignored: {line}")
                    continue

                original, general = parts
                mapping[original] = general

        mapping_expr = F.create_map([F.lit(x) for pair in mapping.items() for x in pair])

        new_col = mapping_expr[F.col(self.column)]

        if self.default_value is not None:
            new_col = F.coalesce(new_col, F.lit(self.default_value))

        return self._rename_output_column(df.withColumn(self.column, new_col))

    def _apply_numeric(self, df: DataFrame) -> DataFrame:
        """
        Applies numeric generalization using interval rules.

        Returns
        -------
        pyspark.sql.DataFrame
        """

        expr = None

        with open(self.rules_path, "r") as f:
            for line in f:
                parts = line.strip().split(";")

                if len(parts) != 3:
                    warnings.warn(f"Invalid line ignored: {line}")
                    continue

                start, end, label = parts

                try:
                    start = float(start)
                    end = float(end)
                except:
                    warnings.warn(f"Invalid numeric values: {line}")
                    continue

                if start > end:
                    warnings.warn(f"Invalid interval: {line}")
                    continue

                condition = (F.col(self.column) >= start) & (F.col(self.column) <= end)

                if expr is None:
                    expr = F.when(condition, F.lit(label))
                else:
                    expr = expr.when(condition, F.lit(label))

        if expr is None:
            raise ValueError("No valid rules found.")

        if self.default_value is not None:
            expr = expr.otherwise(F.lit(str(self.default_value)))
        else:
            expr = expr.otherwise(F.col(self.column).cast("string"))

        return self._rename_output_column(df.withColumn(self.column, expr.cast("string")))
    

    def _apply_temporal_generalization(self, df: DataFrame) -> DataFrame:
        """
        Applies temporal generalization based on the selected aggregation level.

        By default, include_year is set to True, meaning that the generalized
        values preserve the year (e.g., 2024-05, 2024-Q2, 2024-S1). If set to False,
        values are grouped only by the temporal period (e.g., 05, Q2, S1).
        """

        col = F.col(self.column)

        if self.mode == "year":
            new_col = F.year(col).cast("string")

        elif self.mode == "month":
            if self.include_year:
                new_col = F.date_format(col, "yyyy-MM")
            else:
                new_col = F.date_format(col, "MMMM")

        elif self.mode == "quarter":
            quarter = F.concat(F.lit("Q"), F.quarter(col))
            if self.include_year:
                new_col = F.concat(F.year(col), F.lit("-"), quarter)
            else:
                new_col = quarter

        elif self.mode == "semester":
            semester = F.concat(
                F.lit("S"),
                F.when(F.month(col) <= 6, 1).otherwise(2)
            )
            if self.include_year:
                new_col = F.concat(F.year(col), F.lit("-"), semester)
            else:
                new_col = semester

        else:
            raise ValueError(f"Unsupported mode '{self.mode}'.")

        return self._rename_output_column(df.withColumn(self.column, new_col.cast("string")))
    
    
    def transform(self, df: DataFrame) -> DataFrame:
        """
        Applies generalization to the specified column.

        Depending on the column type and configuration, applies:
        - Categorical mapping using external rules.
        - Numerical interval generalization using external rules.
        - Temporal aggregation using a selected mode.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input dataset.

        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame with the selected column generalized.
        """

        if df is None:
            raise ValueError("Input DataFrame cannot be None.")

        if not isinstance(df, DataFrame):
            raise ValueError("Input must be a pyspark.sql.DataFrame.")

        if self.column not in df.columns:
            raise ValueError(f"Column '{self.column}' does not exist in the DataFrame.")

        semantic_type = infer_semantic_type(df, self.column)

        if semantic_type == "date":
            if self.mode is None:
                raise ValueError("'mode' must be provided for temporal generalization.")
            return self._apply_temporal_generalization(df)

        if semantic_type == "numeric":
            if self.rules_path is None:
                raise ValueError("'rules_path' must be provided for numeric generalization.")
            return self._apply_numeric(df)

        if semantic_type == "categorical":
            if self.rules_path is None:
                raise ValueError("'rules_path' must be provided for categorical generalization.")
            return self._apply_categorical(df)


