from pyspark.sql import DataFrame
from pyspark.sql import functions as F
import os


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

    def _build_suggestions(self,df: DataFrame,equivalence_groups: DataFrame) -> DataFrame:
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
                    F.lit("High impact on equivalence groups")
                ).when(
                    F.col("group_reduction_frequency") >= 0.20,
                    F.lit("Medium impact on equivalence groups")
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
        Displays a global privacy assessment of the dataset.

        The summary includes:

        - target_k:
            Desired k-anonymity level defined by the user.

        - current_k:
            Current k-anonymity level of the dataset, calculated as the
            size of the smallest equivalence group.

        - total_records:
            Total number of records in the dataset.

        - total_equivalence_groups:
            Number of distinct equivalence groups generated from the
            selected quasi-identifiers.

        - risky_groups:
            Number of equivalence groups whose size is smaller than
            the target k value.

        - risky_records:
            Total number of records belonging to risky equivalence groups.

        - risky_records_frequency:
            Proportion of records affected by risky groups with respect
            to the total dataset size.

        Interpretation
        --------------
        A dataset satisfies k-anonymity when:

            current_k >= target_k

        and therefore:

            risky_groups = 0
            risky_records = 0

        """
        self.summary_df.show(truncate=truncate)

    def show_suggestions(self, truncate=False):
        """
        Displays anonymization recommendations for each quasi-identifier.

        The analysis estimates the impact of removing each quasi-identifier
        from the equivalence group definition.

        The output includes:

        - column:
            Evaluated quasi-identifier.

        - cardinality:
            Number of distinct values in the column.

        - current_equivalence_groups:
            Number of equivalence groups generated using all
            selected quasi-identifiers.

        - groups_without_column:
            Number of equivalence groups that would remain if the
            column were excluded from the quasi-identifier set.

        - group_reduction:
            Absolute reduction in equivalence groups obtained by
            removing the column.

        - group_reduction_frequency:
            Relative reduction in equivalence groups expressed as
            a percentage of the original number of groups.

        - suggested_action:
            Recommended anonymization action based on the estimated
            reduction of equivalence groups.

        Interpretation
        --------------
        Columns producing large reductions in equivalence groups
        are strong candidates for generalization or suppression,
        since they contribute significantly to record uniqueness.

        A high cardinality combined with a high
        group_reduction_frequency usually indicates that the column
        has a strong impact on re-identification risk.
        """
        self.suggestions_df.show(truncate=truncate)













