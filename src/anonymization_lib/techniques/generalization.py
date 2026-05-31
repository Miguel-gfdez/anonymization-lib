from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from ..utils import infer_semantic_type
import warnings
import json



class Generalization:
    """
    Applies generalization to a column in a Spark DataFrame.

    The transformation supports:
    - Categorical mapping using external rules
    - Numerical generalization using interval rules
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

    def _get_rules_type(self) -> str:
        with open(self.rules_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        return config.get("type")
    
    def _apply_temporal_with_mode(self, df: DataFrame) -> DataFrame:
        if self.mode is None:
            raise ValueError("'mode' must be provided for temporal generalization.")

        return self._apply_temporal_generalization(df)

    def _apply_categorical(self, df: DataFrame) -> DataFrame:
        """
        Applies categorical generalization using mapping rules from a JSON file.

        Returns
        -------
        pyspark.sql.DataFrame
        """

        mapping = {}

        with open(self.rules_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        rules = config.get("rules", [])

        if not rules:
            raise ValueError("No rules found in JSON file.")

        for rule in rules:
            try:
                original = str(rule["from"])
                general = str(rule["to"])
            except KeyError as e:
                warnings.warn(f"Invalid rule ignored. Missing key {e}: {rule}")
                continue
            except (TypeError, ValueError):
                warnings.warn(f"Invalid categorical rule ignored: {rule}")
                continue

            mapping[original] = general

        if not mapping:
            raise ValueError("No valid rules found.")

        mapping_expr = F.create_map(
            [F.lit(x) for pair in mapping.items() for x in pair]
        )

        new_col = mapping_expr[F.col(self.column).cast("string")]

        if self.default_value is not None:
            new_col = F.coalesce(new_col, F.lit(str(self.default_value)))
        else:
            new_col = F.coalesce(new_col, F.col(self.column).cast("string"))

        return self._rename_output_column(df.withColumn(self.column, new_col.cast("string")))

    def _apply_numeric(self, df: DataFrame) -> DataFrame:
        """
        Applies numeric generalization using interval rules from a JSON file.

        Returns
        -------
        pyspark.sql.DataFrame
        """

        expr = None

        with open(self.rules_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        rules = config.get("rules", [])

        if not rules:
            raise ValueError("No rules found in JSON file.")

        for rule in rules:
            try:
                start = float(rule["min"])
                end = float(rule["max"])
                label = str(rule["value"])
            except KeyError as e:
                warnings.warn(f"Invalid rule ignored. Missing key {e}: {rule}")
                continue
            except (TypeError, ValueError):
                warnings.warn(f"Invalid numeric values in rule: {rule}")
                continue

            if start > end:
                warnings.warn(f"Invalid interval ignored: {rule}")
                continue

            condition = (
                (F.col(self.column).cast("double") >= start) &
                (F.col(self.column).cast("double") <= end)
            )

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

        if self.rules_path is not None:
            json_type = self._get_rules_type()

            if json_type == "numeric":
                return self._apply_numeric(df)

            if json_type == "categorical":
                return self._apply_categorical(df)

            if json_type == "date":
                return self._apply_temporal_with_mode(df)

            raise ValueError(f"Unsupported generalization type in JSON: {json_type}")

        semantic_type = infer_semantic_type(df, self.column)

        if semantic_type == "date":
            return self._apply_temporal_with_mode(df)

        raise ValueError(
            "For numeric or categorical generalization, 'rules_path' must be provided."
        )


