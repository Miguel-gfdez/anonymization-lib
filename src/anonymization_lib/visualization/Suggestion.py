from pyspark.sql import DataFrame
from pyspark.sql import functions as F


class AnonymizationAdvisor:
    """
    Suggests anonymization improvements based on k-anonymity analysis.

    The advisor evaluates the current level of k-anonymity of a dataset
    and analyzes the influence of each quasi-identifier on the formation
    of equivalence groups.

    For each quasi-identifier, the advisor estimates how the number of
    equivalence groups would change if that attribute were removed from
    the quasi-identifier set. This information can help identify attributes
    with a high impact on data fragmentation and therefore potential
    candidates for generalization, suppression, or other anonymization
    transformations.

    Parameters
    ----------
    quasi_identifiers : list of str
        Columns used to form equivalence groups (quasi-identifiers).
        These are attributes that could potentially re-identify individuals
        when combined.

    k : int
        Target k-anonymity value. Each equivalence group should contain
        at least ``k`` records for the dataset to satisfy the configured
        k-anonymity requirement.

        The value must be an integer greater than 1.

    Notes
    -----
    The advisor does not modify the input dataset or automatically apply
    anonymization techniques. It only analyzes the dataset and generates
    recommendations that can guide subsequent anonymization decisions.

    The analysis is performed using Spark DataFrame operations so that
    large datasets can be processed in a distributed manner.

    The generated suggestions are based on the reduction in the number
    of equivalence groups produced by removing each quasi-identifier
    individually. A larger reduction indicates that the attribute has a
    stronger influence on the fragmentation of the dataset.
    """

    def __init__(self, quasi_identifiers: list, k: int):
        if not isinstance(quasi_identifiers, list) or not quasi_identifiers:
            raise ValueError("'quasi_identifiers' must be a non-empty list.")

        if not all(
            isinstance(col, str) and col.strip()
            for col in quasi_identifiers
        ):
            raise ValueError(
                "All quasi-identifiers must be non-empty strings."
            )

        if len(quasi_identifiers) != len(set(quasi_identifiers)):
            raise ValueError(
                "'quasi_identifiers' must not contain duplicates."
            )

        if not isinstance(k, int) or k <= 1:
            raise ValueError(
                "'k' must be an integer greater than 1."
            )

        self.quasi_identifiers = quasi_identifiers
        self.k = k

    def suggest(
        self,
        df: DataFrame
    ) -> "AnonymizationAdvisorResult":
        """
        Analyzes the dataset and generates recommendations to improve
        k-anonymity.

        Parameters
        ----------
        df : DataFrame
            Input Spark DataFrame.

        Returns
        -------
        AnonymizationAdvisorResult
            Object containing the summary and column-level suggestions.
        """
        if df is None:
            raise ValueError("Input DataFrame cannot be None.")

        # Compatible with standard Spark and Spark Connect.
        if not hasattr(df, "groupBy"):
            raise ValueError(
                "Input must be a Spark-compatible DataFrame."
            )

        missing_columns = [
            column
            for column in self.quasi_identifiers
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Columns not found in DataFrame: {missing_columns}"
            )

        equivalence_groups = self._build_equivalence_groups(
            df,
            self.quasi_identifiers
        )

        summary_df, current_groups = self._build_summary(
            df,
            equivalence_groups
        )

        suggestions_df = self._build_suggestions(
            df,
            current_groups
        )

        return AnonymizationAdvisorResult(
            summary_df=summary_df,
            suggestions_df=suggestions_df
        )

    def _build_equivalence_groups(
        self,
        df: DataFrame,
        columns: list
    ) -> DataFrame:
        """
        Builds equivalence groups using the selected quasi-identifiers.
        """
        return (
            df
            .groupBy(*columns)
            .agg(F.count("*").alias("group_size"))
        )

    def _build_summary(
        self,
        df: DataFrame,
        equivalence_groups: DataFrame
    ):
        """
        Builds the global k-anonymity summary.
        """
        total_records = df.count()

        stats = (
            equivalence_groups
            .agg(
                F.count("*").alias("total_groups"),

                F.sum(
                    F.when(
                        F.col("group_size") < self.k,
                        1
                    ).otherwise(0)
                ).alias("risky_groups"),

                F.sum(
                    F.when(
                        F.col("group_size") < self.k,
                        F.col("group_size")
                    ).otherwise(0)
                ).alias("risky_records"),

                F.min("group_size").alias("current_k")
            )
            .first()
        )

        total_groups = stats["total_groups"]
        risky_groups = stats["risky_groups"] or 0
        risky_records = stats["risky_records"] or 0
        current_k = stats["current_k"]

        risky_records_frequency = (
            round(risky_records / total_records, 4)
            if total_records > 0
            else 0.0
        )

        summary_df = df.sparkSession.createDataFrame(
            [(
                self.k,
                current_k,
                total_records,
                total_groups,
                risky_groups,
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

        return summary_df, total_groups

    def _build_suggestions(
        self,
        df: DataFrame,
        current_groups: int
    ) -> DataFrame:
        """
        Builds column-level anonymization suggestions.
        """
        qis = self.quasi_identifiers

        # Calculate all column cardinalities in a single Spark action.
        cardinalities = (
            df
            .agg(*[
                F.countDistinct(F.col(column)).alias(column)
                for column in qis
            ])
            .first()
        )

        # Calculate the number of equivalence groups that would remain
        # after removing each quasi-identifier.
        if len(qis) == 1:
            groups_without = {
                qis[0]: 1
            }

        else:
            groups_without = (
                df
                .agg(*[
                    F.countDistinct(
                        F.struct(*[
                            F.col(c)
                            for c in qis
                            if c != column
                        ])
                    ).alias(column)
                    for column in qis
                ])
                .first()
            )

        rows = []

        for column in qis:
            cardinality = cardinalities[column]
            groups_without_column = groups_without[column]

            group_reduction = (
                current_groups - groups_without_column
            )

            group_reduction_frequency = (
                round(
                    group_reduction / current_groups,
                    4
                )
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

        result = df.sparkSession.createDataFrame(
            rows,
            [
                "column",
                "cardinality",
                "current_equivalence_groups",
                "groups_without_column",
                "group_reduction",
                "group_reduction_frequency",
            ]
        )

        return (
            result
            .withColumn(
                "suggested_action",
                F.when(
                    F.col("group_reduction_frequency") >= 0.50,
                    F.lit(
                        "High impact on equivalence groups"
                    )
                )
                .when(
                    F.col("group_reduction_frequency") >= 0.20,
                    F.lit(
                        "Medium impact on equivalence groups"
                    )
                )
                .otherwise(
                    F.lit(
                        "Low impact on equivalence groups"
                    )
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

    def __init__(
        self,
        summary_df: DataFrame,
        suggestions_df: DataFrame
    ):
        self.summary_df = summary_df
        self.suggestions_df = suggestions_df

    def get_summary_df(self) -> DataFrame:
        """
        Returns the summary DataFrame.
        """
        return self.summary_df

    def get_suggestions_df(self) -> DataFrame:
        """
        Returns the suggestions DataFrame.
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