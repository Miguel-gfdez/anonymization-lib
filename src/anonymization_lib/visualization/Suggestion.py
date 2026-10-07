from pyspark.sql import DataFrame
from pyspark.sql import functions as F


class AnonymizationAdvisor:
    """
    Suggests anonymization improvements based on k-anonymity analysis.

    The advisor evaluates the current level of k-anonymity of a dataset
    and analyzes the influence of each quasi-identifier on the formation
    of equivalence groups.

    For each quasi-identifier, the advisor evaluates two complementary
    properties:

    1. Cardinality frequency:
       Measures the proportion of distinct values in the column relative
       to the total number of records. High-cardinality attributes may
       have strong individual identifying power.

    2. Equivalence-group reduction:
       Estimates how the number of equivalence groups would change if
       the attribute were removed from the quasi-identifier set. A large
       reduction indicates that the attribute contributes significantly
       to the fragmentation of the dataset.

    Using both metrics avoids misleading recommendations when multiple
    high-cardinality quasi-identifiers coexist. For example, removing
    one unique identifier may produce no reduction in equivalence groups
    if another unique identifier remains in the quasi-identifier set.

    Parameters
    ----------
    quasi_identifiers : list of str
        Columns used to form equivalence groups (quasi-identifiers).
        These are attributes that could potentially re-identify
        individuals when combined.

    k : int
        Target k-anonymity value. Each equivalence group should contain
        at least ``k`` records for the dataset to satisfy the configured
        k-anonymity requirement.

        The value must be an integer greater than 1. Boolean values are
        not accepted.

    Notes
    -----
    The advisor does not modify the input dataset or automatically apply
    anonymization techniques. It only analyzes the dataset and generates
    recommendations that can guide subsequent anonymization decisions.

    The analysis is performed using Spark DataFrame operations so that
    large datasets can be processed in a distributed manner.

    The generated recommendations combine two complementary indicators:

    - ``cardinality_frequency``:
      Ratio between the number of distinct values in a quasi-identifier
      and the total number of records.

    - ``group_reduction_frequency``:
      Relative reduction in the number of equivalence groups when the
      quasi-identifier is removed from the current quasi-identifier set.

    The risk thresholds used by the advisor are heuristic and are intended
    to provide practical guidance rather than a formal privacy guarantee.
    The actual anonymization strategy should also consider the semantics
    of each attribute and the intended use of the dataset.
    """

    HIGH_CARDINALITY_THRESHOLD = 0.50
    MEDIUM_CARDINALITY_THRESHOLD = 0.10

    HIGH_GROUP_REDUCTION_THRESHOLD = 0.50
    MEDIUM_GROUP_REDUCTION_THRESHOLD = 0.20

    def __init__(
        self,
        quasi_identifiers: list,
        k: int,
    ):
        if (
            not isinstance(quasi_identifiers, list)
            or not quasi_identifiers
        ):
            raise ValueError(
                "'quasi_identifiers' must be a non-empty list."
            )

        if not all(
            isinstance(column, str) and column.strip()
            for column in quasi_identifiers
        ):
            raise ValueError(
                "All quasi-identifiers must be non-empty strings."
            )

        if len(quasi_identifiers) != len(set(quasi_identifiers)):
            raise ValueError(
                "'quasi_identifiers' must not contain duplicates."
            )

        if (
            not isinstance(k, int)
            or isinstance(k, bool)
            or k <= 1
        ):
            raise ValueError(
                "'k' must be an integer greater than 1."
            )

        self.quasi_identifiers = quasi_identifiers
        self.k = k

    def suggest(
        self,
        df: DataFrame,
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
            Object containing the global privacy summary and the
            column-level anonymization recommendations.

        Raises
        ------
        ValueError
            If the input is None, is not compatible with a Spark
            DataFrame, or does not contain all configured
            quasi-identifiers.
        """
        if df is None:
            raise ValueError(
                "Input DataFrame cannot be None."
            )

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
            self.quasi_identifiers,
        )

        (
            summary_df,
            current_groups,
            total_records,
        ) = self._build_summary(
            df,
            equivalence_groups,
        )

        suggestions_df = self._build_suggestions(
            df,
            current_groups,
            total_records,
        )

        return AnonymizationAdvisorResult(
            summary_df=summary_df,
            suggestions_df=suggestions_df,
        )

    def _build_equivalence_groups(
        self,
        df: DataFrame,
        columns: list,
    ) -> DataFrame:
        """
        Builds equivalence groups using the selected quasi-identifiers.

        Parameters
        ----------
        df : DataFrame
            Input Spark DataFrame.

        columns : list of str
            Columns used to define the equivalence groups.

        Returns
        -------
        DataFrame
            DataFrame containing one row per equivalence group and a
            ``group_size`` column with the number of records belonging
            to each group.
        """
        return (
            df
            .groupBy(*columns)
            .agg(
                F.count("*").alias("group_size")
            )
        )

    def _build_summary(
        self,
        df: DataFrame,
        equivalence_groups: DataFrame,
    ):
        """
        Builds the global k-anonymity summary.

        The total number of records is calculated from the sizes of the
        already generated equivalence groups instead of executing an
        additional ``df.count()`` action over the original dataset.

        Parameters
        ----------
        df : DataFrame
            Input Spark DataFrame. It is used to create the resulting
            summary DataFrame.

        equivalence_groups : DataFrame
            DataFrame containing the equivalence groups and their sizes.

        Returns
        -------
        tuple
            A tuple containing:

            - summary_df:
              Spark DataFrame with the global k-anonymity statistics.

            - total_groups:
              Number of equivalence groups in the dataset.

            - total_records:
              Number of records in the dataset.
        """
        stats = (
            equivalence_groups
            .agg(
                F.count("*").alias(
                    "total_groups"
                ),

                F.sum("group_size").alias(
                    "total_records"
                ),

                F.sum(
                    F.when(
                        F.col("group_size") < self.k,
                        1,
                    ).otherwise(0)
                ).alias(
                    "risky_groups"
                ),

                F.sum(
                    F.when(
                        F.col("group_size") < self.k,
                        F.col("group_size"),
                    ).otherwise(0)
                ).alias(
                    "risky_records"
                ),

                F.min("group_size").alias(
                    "current_k"
                ),
            )
            .first()
        )

        total_groups = stats["total_groups"] or 0
        total_records = stats["total_records"] or 0
        risky_groups = stats["risky_groups"] or 0
        risky_records = stats["risky_records"] or 0
        current_k = stats["current_k"]

        risky_records_frequency = (
            round(
                risky_records / total_records,
                4,
            )
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
                risky_records_frequency,
            )],
            [
                "target_k",
                "current_k",
                "total_records",
                "total_equivalence_groups",
                "risky_groups",
                "risky_records",
                "risky_records_frequency",
            ],
        )

        return (
            summary_df,
            total_groups,
            total_records,
        )

    def _build_suggestions(
        self,
        df: DataFrame,
        current_groups: int,
        total_records: int,
    ) -> DataFrame:
        """
        Builds column-level anonymization suggestions.

        Each quasi-identifier is evaluated using two complementary
        indicators:

        ``cardinality_frequency``
            Ratio between the number of distinct values in the column
            and the total number of records.

        ``group_reduction_frequency``
            Relative reduction in the number of equivalence groups that
            would occur if the column were removed from the current
            quasi-identifier set.

        Combining both indicators allows the advisor to detect
        high-cardinality attributes even when their marginal removal
        produces little or no reduction because another identifying
        attribute remains in the quasi-identifier set.

        Parameters
        ----------
        df : DataFrame
            Input Spark DataFrame.

        current_groups : int
            Number of equivalence groups generated using all configured
            quasi-identifiers.

        total_records : int
            Total number of records in the dataset.

        Returns
        -------
        DataFrame
            Spark DataFrame containing cardinality information,
            equivalence-group impact, risk level, and a suggested
            anonymization action for each quasi-identifier.
        """
        qis = self.quasi_identifiers

        # Calculate all column cardinalities in a single Spark action.
        cardinalities = (
            df
            .agg(*[
                F.countDistinct(
                    F.col(column)
                ).alias(column)
                for column in qis
            ])
            .first()
        )

        # Calculate the number of equivalence groups that would remain
        # after removing each quasi-identifier.
        #
        # When there is only one quasi-identifier, removing it leaves
        # all records in one equivalence group.
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
                            F.col(other_column)
                            for other_column in qis
                            if other_column != column
                        ])
                    ).alias(column)
                    for column in qis
                ])
                .first()
            )

        rows = []

        for column in qis:
            cardinality = (
                cardinalities[column]
                or 0
            )

            groups_without_column = (
                groups_without[column]
                or 0
            )

            cardinality_frequency = (
                cardinality / total_records
                if total_records > 0
                else 0.0
            )

            group_reduction = (
                current_groups
                - groups_without_column
            )

            group_reduction_frequency = (
                group_reduction / current_groups
                if current_groups > 0
                else 0.0
            )

            rows.append((
                column,
                cardinality,
                cardinality_frequency,
                current_groups,
                groups_without_column,
                group_reduction,
                group_reduction_frequency,
            ))

        result = df.sparkSession.createDataFrame(
            rows,
            [
                "column",
                "cardinality",
                "cardinality_frequency",
                "current_equivalence_groups",
                "groups_without_column",
                "group_reduction",
                "group_reduction_frequency",
            ],
        )

        high_risk_condition = (
            (
                F.col("cardinality_frequency")
                >= self.HIGH_CARDINALITY_THRESHOLD
            )
            |
            (
                F.col("group_reduction_frequency")
                >= self.HIGH_GROUP_REDUCTION_THRESHOLD
            )
        )

        medium_risk_condition = (
            (
                F.col("cardinality_frequency")
                >= self.MEDIUM_CARDINALITY_THRESHOLD
            )
            |
            (
                F.col("group_reduction_frequency")
                >= self.MEDIUM_GROUP_REDUCTION_THRESHOLD
            )
        )

        return (
            result
            .withColumn(
                "risk_level",
                F.when(
                    high_risk_condition,
                    F.lit("High"),
                )
                .when(
                    medium_risk_condition,
                    F.lit("Medium"),
                )
                .otherwise(
                    F.lit("Low")
                ),
            )
            .withColumn(
                "suggested_action",
                F.when(
                    F.col("cardinality_frequency")
                    >= self.HIGH_CARDINALITY_THRESHOLD,
                    F.lit(
                        "Consider suppression or strong generalization"
                    ),
                )
                .when(
                    F.col("group_reduction_frequency")
                    >= self.HIGH_GROUP_REDUCTION_THRESHOLD,
                    F.lit(
                        "Consider generalization"
                    ),
                )
                .when(
                    (
                        F.col("cardinality_frequency")
                        >= self.MEDIUM_CARDINALITY_THRESHOLD
                    )
                    |
                    (
                        F.col("group_reduction_frequency")
                        >= self.MEDIUM_GROUP_REDUCTION_THRESHOLD
                    ),
                    F.lit(
                        "Consider moderate generalization"
                    ),
                )
                .otherwise(
                    F.lit(
                        "Low priority for anonymization"
                    )
                ),
            )
            .orderBy(
                F.when(F.col("risk_level") == "High", 3)
                .when(F.col("risk_level") == "Medium", 2)
                .otherwise(1)
                .desc(),
                F.desc("cardinality_frequency"),
                F.desc("group_reduction_frequency"),
                F.desc("cardinality"),
            )
        )


class AnonymizationAdvisorResult:
    """
    Stores the results generated by AnonymizationAdvisor.

    Attributes
    ----------
    summary_df : DataFrame
        Global statistics describing the current equivalence groups
        and the current k-anonymity level.

    suggestions_df : DataFrame
        Column-level anonymization recommendations based on attribute
        cardinality and equivalence-group impact.
    """

    def __init__(
        self,
        summary_df: DataFrame,
        suggestions_df: DataFrame,
    ):
        self.summary_df = summary_df
        self.suggestions_df = suggestions_df

    def get_summary_df(self) -> DataFrame:
        """
        Returns the summary DataFrame.

        Returns
        -------
        DataFrame
            Spark DataFrame containing the global k-anonymity summary.
        """
        return self.summary_df

    def get_suggestions_df(self) -> DataFrame:
        """
        Returns the suggestions DataFrame.

        Returns
        -------
        DataFrame
            Spark DataFrame containing the column-level anonymization
            recommendations.
        """
        return self.suggestions_df

    def show_summary(
        self,
        truncate=False,
    ):
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
            Total number of records belonging to risky equivalence
            groups.

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
        self.summary_df.show(
            truncate=truncate
        )

    def show_suggestions(
        self,
        truncate=False,
    ):
        """
        Displays anonymization recommendations for each quasi-identifier.

        The analysis combines the individual cardinality of each
        quasi-identifier with the impact of removing that attribute from
        the current equivalence-group definition.

        The output includes:

        - column:
            Evaluated quasi-identifier.

        - cardinality:
            Number of distinct non-null values in the column.

        - cardinality_frequency:
            Ratio between the number of distinct values and the total
            number of records.

            Values close to 1 indicate that the attribute has a large
            number of distinct values relative to the dataset size and
            may therefore have strong identifying power.

        - current_equivalence_groups:
            Number of equivalence groups generated using all selected
            quasi-identifiers.

        - groups_without_column:
            Number of equivalence groups that would remain if the column
            were excluded from the quasi-identifier set.

        - group_reduction:
            Absolute reduction in equivalence groups obtained by removing
            the column.

        - group_reduction_frequency:
            Relative reduction in equivalence groups with respect to the
            original number of groups.

        - risk_level:
            Heuristic classification of the quasi-identifier as High,
            Medium, or Low risk.

            The classification considers both cardinality frequency and
            equivalence-group reduction.

        - suggested_action:
            Suggested anonymization strategy based on the observed
            characteristics of the quasi-identifier.

        Interpretation
        --------------
        ``cardinality_frequency`` and ``group_reduction_frequency`` measure
        different properties.

        A high cardinality frequency indicates that the attribute has
        substantial individual identifying power.

        A high group reduction frequency indicates that removing the
        attribute significantly reduces the fragmentation of the dataset.

        These measures should be interpreted together. For example, two
        unique identifiers may both have a cardinality frequency close to
        1 while individually producing almost no group reduction because
        the other unique identifier continues to separate every record.

        The risk levels and suggested actions are heuristic guidance.
        They do not themselves constitute a formal privacy guarantee.
        """
        self.suggestions_df.show(
            truncate=truncate
        )