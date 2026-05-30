from pyspark.sql import DataFrame
from pyspark.sql import functions as F
import os


class ParameterSuggestion:
    """
    Suggests anonymization parameters for a selected column based on
    its distribution and semantic type.

    Numeric columns are analyzed using equal-frequency intervals,
    while categorical columns are analyzed according to the frequency
    distribution of their categories.
    """

    NUMERIC_TYPES = {
        "int", "bigint", "double", "float", "long",
        "smallint", "tinyint", "decimal"
    }

    def __init__(
        self,
        column: str,
        rare_threshold: float = 0.05,
        max_categories: int = 20,
        num_bins: int = 5
    ):
        if not isinstance(column, str) or not column.strip():
            raise ValueError("'column' must be a non-empty string.")

        if not isinstance(rare_threshold, (int, float)):
            raise ValueError("'rare_threshold' must be numeric.")

        rare_threshold = float(rare_threshold)

        if rare_threshold <= 0.0 or rare_threshold >= 1.0:
            raise ValueError("'rare_threshold' must be between 0 and 1.")

        if not isinstance(max_categories, int) or max_categories <= 1:
            raise ValueError("'max_categories' must be an integer greater than 1.")

        if not isinstance(num_bins, int) or num_bins <= 1:
            raise ValueError("'num_bins' must be an integer greater than 1.")

        self.column = column
        self.rare_threshold = rare_threshold
        self.max_categories = max_categories
        self.num_bins = num_bins

    def suggest(self, df: DataFrame) -> "ParameterSuggestionResult":
        """
        Analyzes the selected column and generates anonymization suggestions.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input dataset.

        Returns
        -------
        ParameterSuggestionResult
            Object containing summary information and suggestedanonymization configurations.
        """
        if not isinstance(df, DataFrame):
            raise ValueError("'df' must be a PySpark DataFrame.")

        if self.column not in df.columns:
            raise ValueError(f"Column '{self.column}' does not exist in the DataFrame.")

        total_records = df.count()

        if total_records == 0:
            raise ValueError("The DataFrame is empty.")

        spark_type = dict(df.dtypes)[self.column]

        if any(t in spark_type for t in self.NUMERIC_TYPES):
            return self._suggest_numeric(df, total_records, spark_type)

        return self._suggest_categorical(df, total_records, spark_type)

    def _suggest_numeric(self,df: DataFrame,total_records: int,spark_type: str) -> "ParameterSuggestionResult":
        """
        Generates anonymization suggestions for numeric columns.

        Numeric values are grouped into equal-frequency intervals
        using approximate quantiles.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input dataset.

        total_records : int
            Total number of rows in the dataset.

        spark_type : str
            Spark data type of the analyzed column.

        Returns
        -------
        ParameterSuggestionResult
            Object containing interval suggestions and summary statistics.
        """
        
        stats_df = df.select(
            F.min(self.column).alias("min_value"),
            F.max(self.column).alias("max_value"),
            F.mean(self.column).alias("mean_value"),
            F.stddev(self.column).alias("stddev_value"),
            F.count(self.column).alias("non_null_records")
        )

        stats = stats_df.collect()[0]

        min_value = stats["min_value"]
        max_value = stats["max_value"]

        summary_df = (
            stats_df
            .withColumns({
                "column": F.lit(self.column),
                "inferred_type": F.lit("numeric"),
                "spark_type": F.lit(spark_type),
                "total_records": F.lit(total_records),
                "suggested_technique": F.lit("generalization"),
                "binning_strategy": F.lit("equal_frequency")
            })
            .select(
                "column",
                "inferred_type",
                "spark_type",
                "total_records",
                "non_null_records",
                "min_value",
                "max_value",
                "mean_value",
                "stddev_value",
                "suggested_technique",
                "binning_strategy"
            )
        )

        empty_schema = """
            start_value DOUBLE,
            end_value DOUBLE,
            count LONG,
            frequency DOUBLE
        """
        integer_types = {"int", "bigint", "long", "smallint", "tinyint"}
        is_integer_column = any(t in spark_type for t in integer_types)

        if min_value is None or max_value is None or min_value == max_value:
            suggestions_df = df.sparkSession.createDataFrame([], empty_schema)
            return ParameterSuggestionResult(summary_df, suggestions_df, suggestion_type='numeric')

        quantile_probs = [i / self.num_bins for i in range(self.num_bins + 1)]

        quantiles = df.approxQuantile(
            self.column,
            quantile_probs,
            0.01
        )

        quantiles = sorted(set(quantiles))

        if len(quantiles) <= 1:
            suggestions_df = df.sparkSession.createDataFrame([], empty_schema)
            return ParameterSuggestionResult(summary_df, suggestions_df, suggestion_type='numeric')

        bin_expr = None

        for i in range(len(quantiles) - 1):
            start = quantiles[i]
            end = quantiles[i + 1]

            if i == len(quantiles) - 2:
                condition = (
                    (F.col(self.column) >= F.lit(start)) &
                    (F.col(self.column) <= F.lit(end))
                )
            else:
                condition = (
                    (F.col(self.column) >= F.lit(start)) &
                    (F.col(self.column) < F.lit(end))
                )

            if bin_expr is None:
                bin_expr = F.when(condition, F.lit(i))
            else:
                bin_expr = bin_expr.when(condition, F.lit(i))

        binned_df = (
            df
            .where(F.col(self.column).isNotNull())
            .withColumn("bin_index", bin_expr)
            .where(F.col("bin_index").isNotNull())
        )

        intervals_df = df.sparkSession.createDataFrame(
            [
                (i, float(quantiles[i]), float(quantiles[i + 1]))
                for i in range(len(quantiles) - 1)
            ],
            ["bin_index", "start_value", "end_value"]
        )

        suggestions_df = (
            binned_df
            .groupBy("bin_index")
            .agg(F.count("*").alias("count"))
            .join(intervals_df, on="bin_index", how="left")
            .withColumn("frequency", F.round(F.col("count") / F.lit(total_records), 4))
        )

        if is_integer_column:
            suggestions_df = (
                suggestions_df
                .withColumn("start_value", F.col("start_value").cast("int"))
                .withColumn("end_value", F.col("end_value").cast("int"))
            )
        else:
            suggestions_df = (
                suggestions_df
                .withColumn("start_value", F.round(F.col("start_value"), 2))
                .withColumn("end_value", F.round(F.col("end_value"), 2))
            )

        suggestions_df = (
            suggestions_df
            .select(
                "start_value",
                "end_value",
                "count",
                "frequency"
            )
            .orderBy("start_value")
        )

        return ParameterSuggestionResult(summary_df, suggestions_df, suggestion_type='numeric')

    def _suggest_categorical(self,df: DataFrame,total_records: int,spark_type: str) -> "ParameterSuggestionResult":
        """
        Generates anonymization suggestions for categorical columns.

        Categories whose frequency is below the configured threshold
        are considered infrequent values and are suggested to be
        generalized using the default replacement value.

        Categories above the threshold are considered sufficiently
        representative and are preserved as explicit mappings in the
        generated rules file.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input dataset.

        total_records : int
            Total number of rows in the dataset.

        spark_type : str
            Spark data type of the analyzed column.

        Returns
        -------
        ParameterSuggestionResult
            Object containing:
            - summary statistics
            - aggregated anonymization suggestions
            - representative categories to preserve explicitly
        """

        freq_df = (
            df.groupBy(self.column)
            .agg(F.count("*").alias("count"))
            .withColumn(
                "frequency",
                F.col("count") / F.lit(total_records)
            )
        )

        rare_df = freq_df.filter(
            F.col("frequency") < F.lit(self.rare_threshold)
        )

        common_df = freq_df.filter(
            F.col("frequency") >= F.lit(self.rare_threshold)
        )

        suggestions_df = (
            rare_df
            .agg(
                F.count("*").alias("rare_categories_count"),
                F.sum("count").alias("affected_rows"),
                F.round(
                    F.sum("frequency"),
                    4
                ).alias("affected_frequency")
            )
        )

        detail_df = (
            common_df
            .orderBy(F.desc("count"))
            .withColumn(
                "category_value",
                F.col(self.column).cast("string")
            )
            .select(
                "category_value",
                "count",
                F.round(
                    F.col("frequency"),
                    4
                ).alias("frequency")
            )
        )

        total_categories = freq_df.count()
        rare_categories = rare_df.count()
        common_categories = common_df.count()

        summary_df = df.sparkSession.createDataFrame(
            [(
                self.column,
                "categorical",
                spark_type,
                total_records,
                total_categories,
                rare_categories,
                common_categories,
                self.rare_threshold,
                "generalization",
                "default_other"
            )],
            [
                "column",
                "inferred_type",
                "spark_type",
                "total_records",
                "total_categories",
                "rare_categories",
                "common_categories",
                "rare_threshold",
                "suggested_technique",
                "default_generalization"
            ]
        )

        return ParameterSuggestionResult(summary_df=summary_df,suggestions_df=suggestions_df,preserved_categories_df=detail_df,suggestion_type="categorical")


class ParameterSuggestionResult:
    """
    Stores the results generated by ParameterSuggestion.

    The result may contain:
    - summary_df:
        General metadata and anonymization recommendations.

    - suggestions_df:
        Aggregated suggestion statistics.

    - preserved_categories_df:
        Categories that should be explicitly preserved in
        categorical generalization rules.
    """

    def __init__(self,summary_df: DataFrame,suggestions_df: DataFrame,preserved_categories_df: DataFrame = None, suggestion_type: str=None):
        self.summary_df = summary_df
        self.suggestions_df = suggestions_df
        self.preserved_categories_df = preserved_categories_df
        self.suggestion_type=suggestion_type

    def show_summary(self, truncate=False):
        """
        Displays the summary DataFrame..
        """
        self.summary_df.show(truncate=truncate)

    def show_suggestions(self, truncate=False):
        """
        Displays the suggestions DataFrame.
        """
        self.suggestions_df.show(truncate=truncate)

    def show_preserved_categories(self, truncate=False):
        """
        Displays the preserved categories DataFrame.
        """
        if self.preserved_categories_df is not None:
            self.preserved_categories_df.show(truncate=truncate)
    
    def save(self, path: str):
        """
        Saves the generated anonymization suggestions to a text file.

        Numeric suggestions are exported as interval-based rules:

            start;end;generalized_value

        Categorical suggestions are exported as grouping rules:

            original_value;generalized_value

        Parameters
        ----------
        path : str
            Output file path.

        """

        if not isinstance(path, str) or not path.strip():
            raise ValueError("'path' must be a non-empty string.")

        path = path.strip()

        if not path.endswith(".txt"):
            path += ".txt"

        directory = os.path.dirname(path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:

            if self.suggestion_type == "numeric":
                rows = self.suggestions_df.collect()

                for row in rows:

                    start = row["start_value"]
                    end = row["end_value"]

                    generalized = f"{start}-{end}"

                    f.write(f"{start};{end};{generalized}\n")

            elif self.suggestion_type == "categorical":

                detail_rows = self.preserved_categories_df.collect()

                for row in detail_rows:

                    value = row["category_value"]

                    f.write(f"{value};{value}\n")


########################################


class AnonymizationAdvisor:
    """
    Suggests anonymization improvements based on k-anonymity.
    """

    def __init__(self, quasi_identifiers: list, k: int):
        if not isinstance(quasi_identifiers, list) or not quasi_identifiers:
            raise ValueError("'quasi_identifiers' must be a non-empty list.")

        if not all(isinstance(col, str) and col.strip() for col in quasi_identifiers):
            raise ValueError("All quasi-identifiers must be non-empty strings.")

        if len(quasi_identifiers) != len(set(quasi_identifiers)):
            raise ValueError("'quasi_identifiers' must not contain duplicates.")

        if not isinstance(k, int) or k <= 1:
            raise ValueError("'k' must be an integer greater than 1.")

        self.quasi_identifiers = quasi_identifiers
        self.k = k

    def suggest(self, df: DataFrame) -> "AnonymizationAdvisorResult":
        """
        Analyzes the dataset and generates recommendations to improve
        k-anonymity by identifying quasi-identifiers that contribute the most
        to the creation of equivalence groups.

        Parameters
        ----------
        df : DataFrame
            Input Spark DataFrame.

        Returns
        -------
        AnonymizationAdvisorResult
            Object containing a summary of the current equivalence groups
            and column-level anonymization suggestions.
        """
        if df is None:
            raise ValueError("Input DataFrame cannot be None.")

        if not isinstance(df, DataFrame):
            raise ValueError("Input must be a pyspark.sql.DataFrame.")

        missing_columns = [
            column for column in self.quasi_identifiers
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(f"Columns not found in DataFrame: {missing_columns}")

        equivalence_groups = (
            self
            ._build_equivalence_groups(df, self.quasi_identifiers)
            .cache()
        )

        equivalence_groups.count()

        risky_groups = equivalence_groups.filter(F.col("group_size") < self.k)

        summary_df = self._build_summary(df, equivalence_groups, risky_groups)
        suggestions_df = self._build_suggestions(df, equivalence_groups)

        return AnonymizationAdvisorResult(
            summary_df=summary_df,
            suggestions_df=suggestions_df
        )

    def _build_equivalence_groups(self, df: DataFrame, columns: list) -> DataFrame:
        return (
            df
            .groupBy(*columns)
            .agg(F.count("*").alias("group_size"))
        )

    def _get_k_value(self, equivalence_groups: DataFrame) -> int:
        return (
            equivalence_groups
            .agg(F.min("group_size").alias("k"))
            .collect()[0]["k"]
        )

    def _get_group_count(self, equivalence_groups: DataFrame) -> int:
        return equivalence_groups.count()

    def _get_column_cardinality(self, df: DataFrame, column: str) -> int:
        return df.select(column).distinct().count()

    def _build_summary(self,df: DataFrame,equivalence_groups: DataFrame,risky_groups: DataFrame) -> DataFrame:
        total_records = df.count()
        total_groups = equivalence_groups.count()
        risky_groups_count = risky_groups.count()

        risky_records = (
            risky_groups
            .agg(F.coalesce(F.sum("group_size"), F.lit(0)).alias("risky_records"))
            .collect()[0]["risky_records"]
        )

        current_k = self._get_k_value(equivalence_groups)

        risky_records_frequency = (
            round(risky_records / total_records, 4)
            if total_records > 0
            else 0.0
        )

        return df.sparkSession.createDataFrame(
            [(
                self.k,
                current_k,
                total_records,
                total_groups,
                risky_groups_count,
                risky_records,
                risky_records_frequency
            )],
            [
                "target_k",
                "current_k",
                "total_records",
                "total_equivalence_groups",
                "risky_groups",
                "risky_records",
                "risky_records_frequency"
            ]
        )

    def _build_suggestions(
        self,
        df: DataFrame,
        equivalence_groups: DataFrame
    ) -> DataFrame:
        current_groups = self._get_group_count(equivalence_groups)

        rows = []

        for column in self.quasi_identifiers:
            remaining_columns = [
                col for col in self.quasi_identifiers
                if col != column
            ]

            if not remaining_columns:
                continue

            reduced_groups = (
                equivalence_groups
                .groupBy(*remaining_columns)
                .agg(F.sum("group_size").alias("group_size"))
            )

            groups_without_column = self._get_group_count(reduced_groups)
            cardinality = self._get_column_cardinality(df, column)

            group_reduction = current_groups - groups_without_column

            group_reduction_frequency = (
                round(group_reduction / current_groups, 4)
                if current_groups > 0
                else 0.0
            )

            rows.append((
                column,
                cardinality,
                current_groups,
                groups_without_column,
                group_reduction,
                group_reduction_frequency
            ))

        return (
            df.sparkSession.createDataFrame(
                rows,
                [
                    "column",
                    "cardinality",
                    "current_equivalence_groups",
                    "groups_without_column",
                    "group_reduction",
                    "group_reduction_frequency"
                ]
            )
            .withColumn(
                "suggested_action",
                F.when(
                    F.col("group_reduction_frequency") >= 0.50,
                    F.lit("Strong candidate for generalization")
                ).when(
                    F.col("group_reduction_frequency") >= 0.20,
                    F.lit("Consider moderate generalization")
                ).otherwise(
                    F.lit("Low impact on equivalence groups")
                )
            )
            .orderBy(
                F.desc("group_reduction_frequency"),
                F.desc("cardinality")
            )
        )


class AnonymizationAdvisorResult:
    """
    Stores the results generated by AnonymizationAdvisor.

    Attributes
    ----------
    summary_df : DataFrame
        Global statistics describing the current equivalence groups.

    suggestions_df : DataFrame
        Column-level anonymization recommendations.
    """

    def __init__(self, summary_df: DataFrame, suggestions_df: DataFrame):
        self.summary_df = summary_df
        self.suggestions_df = suggestions_df

    def get_summary_df(self) -> DataFrame:
        """
        Returns the summary DataFrame.

        Returns
        -------
        DataFrame
            Global equivalence group statistics.
        """
        return self.summary_df

    def get_suggestions_df(self) -> DataFrame:
        """
        Returns the suggestions DataFrame.

        Returns
        -------
        DataFrame
            Column-level anonymization recommendations.
        """
        return self.suggestions_df

    def show_summary(self, truncate=False):
        """
        Displays the summary DataFrame.

        Parameters
        ----------
        truncate : bool, default=False
            Whether to truncate long values.
        """
        self.summary_df.show(truncate=truncate)

    def show_suggestions(self, truncate=False):
        """
        Displays the suggestions DataFrame.

        Parameters
        ----------
        truncate : bool, default=False
            Whether to truncate long values.
        """
        self.suggestions_df.show(truncate=truncate)













