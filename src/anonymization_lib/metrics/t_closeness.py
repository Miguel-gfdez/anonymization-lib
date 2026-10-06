from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from ..utils import EquivalenceGroups


class TCloseness(EquivalenceGroups):
    """
    Calculates t-closeness for a dataset based on quasi-identifiers
    and one sensitive attribute.

    T-closeness measures the distance between the distribution of a
    sensitive attribute within each equivalence group and the distribution
    of that attribute in the complete dataset.

    An equivalence group satisfies t-closeness when its distance from the
    global sensitive-attribute distribution does not exceed the configured
    threshold.

    Parameters
    ----------
    quasi_identifiers : list of str
        Columns used to form equivalence groups (quasi-identifiers).
        These are attributes that could potentially re-identify individuals.

    sensitive_attribute : str
        Column representing the sensitive attribute whose distribution
        must be protected (e.g., disease, salary, diagnosis).

    t_threshold : float
        Maximum allowed distance between the local distribution of the
        sensitive attribute within an equivalence group and its global
        distribution in the dataset.

    distance_metric : str, optional
        Metric used to compare distributions. Supported values are:

        - "l1":
            Total variation distance. It is calculated as half of the
            L1 distance between the local and global probability
            distributions.

        - "jsd":
            Jensen-Shannon divergence. It is a symmetric and bounded
            divergence derived from the Kullback-Leibler divergence.

        Default is "l1".

    Notes
    -----
    The implementation is based entirely on Spark DataFrame operations
    and avoids collecting the dataset to the driver.

    All values of the sensitive attribute that occur in the global
    distribution are considered when evaluating each equivalence group.
    If a sensitive value is absent from a particular group, its local
    probability is treated as zero.
    """

    VALID_METRICS = {"l1", "jsd"}

    def __init__(
        self,
        quasi_identifiers=None,
        sensitive_attribute=None,
        t_threshold=None,
        distance_metric="l1"
    ):
        if quasi_identifiers is None:
            quasi_identifiers = []

        if not isinstance(quasi_identifiers, list):
            raise ValueError(
                "'quasi_identifiers' must be a list of column names."
            )

        if not quasi_identifiers:
            raise ValueError(
                "At least one quasi-identifier must be provided."
            )

        if not all(
            isinstance(col, str) and col.strip()
            for col in quasi_identifiers
        ):
            raise ValueError(
                "'quasi_identifiers' must contain only non-empty strings."
            )

        if len(quasi_identifiers) != len(set(quasi_identifiers)):
            raise ValueError(
                "'quasi_identifiers' must not contain duplicate columns."
            )

        if (
            not isinstance(sensitive_attribute, str)
            or not sensitive_attribute.strip()
        ):
            raise ValueError(
                "'sensitive_attribute' must be a non-empty column name."
            )

        if sensitive_attribute in quasi_identifiers:
            raise ValueError(
                "'sensitive_attribute' must not be included in "
                "'quasi_identifiers'."
            )

        if (
            not isinstance(t_threshold, (int, float))
            or isinstance(t_threshold, bool)
        ):
            raise ValueError(
                "'t_threshold' must be a numeric value."
            )

        if t_threshold <= 0:
            raise ValueError(
                "'t_threshold' must be greater than 0."
            )

        if distance_metric not in self.VALID_METRICS:
            raise ValueError(
                f"Unsupported metric. Use {sorted(self.VALID_METRICS)}"
            )

        self.quasi_identifiers = quasi_identifiers
        self.sensitive_attribute = sensitive_attribute
        self.t_threshold = t_threshold
        self.distance_metric = distance_metric

    def summary(self, df: DataFrame):
        """
        Computes t-closeness in a distributed manner.

        The method compares the sensitive-attribute distribution of each
        equivalence group with the global sensitive-attribute distribution.

        The computation follows these steps:

        1. Compute the global distribution of the sensitive attribute.
        2. Build the equivalence groups defined by the quasi-identifiers.
        3. Compute the sensitive-attribute distribution within each group.
        4. Generate all combinations between equivalence groups and global
           sensitive values so that values absent from a group receive a
           local probability of zero.
        5. Compute the selected distance between each local distribution
           and the global distribution.
        6. Identify groups whose distance exceeds the configured threshold.
        7. Generate aggregated statistics describing the resulting
           t-closeness values.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input dataset containing the quasi-identifiers and the
            sensitive attribute.

        Returns
        -------
        TClosenessResult
            Container object with:

            - summary_df:
                Aggregated statistics describing t-closeness across all
                equivalence groups.

            - violating_groups:
                Equivalence groups whose t-closeness value exceeds the
                configured threshold.

        Raises
        ------
        ValueError
            If the input DataFrame is None, is not Spark-compatible, or
            does not contain all required columns.
        """
        if df is None:
            raise ValueError(
                "Input DataFrame cannot be None."
            )

        # Compatible with both standard Spark and Spark Connect.
        if not hasattr(df, "groupBy"):
            raise ValueError(
                "Input must be a Spark-compatible DataFrame."
            )

        self._validate_columns(
            df,
            self.quasi_identifiers + [self.sensitive_attribute]
        )

        # ------------------------------------------------------
        # GLOBAL SENSITIVE-ATTRIBUTE DISTRIBUTION
        # ------------------------------------------------------

        global_dist = (
            df
            .groupBy(self.sensitive_attribute)
            .agg(
                F.count("*").alias("global_count")
            )
            .withColumn(
                "global_total",
                F.sum("global_count").over(Window.partitionBy())
            )
            .withColumn(
                "global_prob",
                F.col("global_count") / F.col("global_total")
            )
            .select(
                self.sensitive_attribute,
                "global_prob"
            )
        )

        # ------------------------------------------------------
        # LOCAL SENSITIVE-ATTRIBUTE DISTRIBUTIONS
        # ------------------------------------------------------

        group_window = Window.partitionBy(
            *self.quasi_identifiers
        )

        group_dist = (
            df
            .groupBy(
                *self.quasi_identifiers,
                self.sensitive_attribute
            )
            .agg(
                F.count("*").alias("count")
            )
            .withColumn(
                "group_total",
                F.sum("count").over(group_window)
            )
            .withColumn(
                "group_prob",
                F.col("count") / F.col("group_total")
            )
            .select(
                *self.quasi_identifiers,
                self.sensitive_attribute,
                "group_prob"
            )
        )

        # ------------------------------------------------------
        # EQUIVALENCE GROUPS
        # ------------------------------------------------------

        groups = (
            group_dist
            .select(*self.quasi_identifiers)
            .distinct()
        )

        # ------------------------------------------------------
        # COMPLETE DISTRIBUTION DOMAIN
        # ------------------------------------------------------
        #
        # Every equivalence group must be compared against every
        # sensitive value that exists globally.
        #
        # If a value does not occur in a particular group, its
        # group probability is zero.
        # ------------------------------------------------------

        complete_domain = (
            groups
            .crossJoin(
                F.broadcast(global_dist)
            )
        )

        joined = (
            complete_domain
            .join(
                group_dist,
                on=(
                    self.quasi_identifiers
                    + [self.sensitive_attribute]
                ),
                how="left"
            )
            .fillna(
                {"group_prob": 0.0}
            )
        )

        # ------------------------------------------------------
        # DISTANCE COMPUTATION
        # ------------------------------------------------------

        t_closeness_df = self._compute_distance(joined)

        # ------------------------------------------------------
        # VIOLATING GROUPS
        # ------------------------------------------------------

        violating_groups = (
            t_closeness_df
            .filter(
                F.col("t_closeness") > self.t_threshold
            )
        )

        # ------------------------------------------------------
        # SUMMARY
        # ------------------------------------------------------

        summary_df = (
            t_closeness_df
            .agg(
                F.count("*").alias(
                    "num_equivalence_groups"
                ),
                F.min("t_closeness").alias(
                    "min_t"
                ),
                F.max("t_closeness").alias(
                    "max_t"
                ),
                F.round(
                    F.avg("t_closeness"),
                    5
                ).alias(
                    "avg_t"
                ),
                F.sum(
                    F.when(
                        F.col("t_closeness")
                        > self.t_threshold,
                        1
                    ).otherwise(0)
                ).alias(
                    "num_violating_groups"
                )
            )
            .withColumn(
                "t_threshold",
                F.lit(self.t_threshold)
            )
            .withColumn(
                "satisfies_t_closeness",
                F.col("max_t")
                <= F.col("t_threshold")
            )
        )

        return TClosenessResult(
            summary_df=summary_df,
            violating_groups=violating_groups
        )

    def _compute_distance(
        self,
        df: DataFrame
    ) -> DataFrame:
        """
        Computes the selected distance between each equivalence group's
        sensitive-attribute distribution and the global distribution.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            DataFrame containing one row for each combination of
            equivalence group and globally observed sensitive value.

            The DataFrame must contain:

            - the configured quasi-identifiers,
            - ``group_prob``: probability of the sensitive value inside
              the equivalence group,
            - ``global_prob``: probability of the sensitive value in the
              complete dataset.

        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame containing one row per equivalence group and a
            ``t_closeness`` column with the calculated distance.

        Notes
        -----
        For ``l1``, the implementation uses total variation distance:

            0.5 * sum(|P(x) - Q(x)|)

        For ``jsd``, the implementation uses Jensen-Shannon divergence
        based on logarithms in base 2.
        """
        if self.distance_metric == "l1":
            return (
                df
                .withColumn(
                    "abs_diff",
                    F.abs(
                        F.col("group_prob")
                        - F.col("global_prob")
                    )
                )
                .groupBy(
                    *self.quasi_identifiers
                )
                .agg(
                    (
                        F.sum("abs_diff") / 2
                    ).alias("t_closeness")
                )
            )

        if self.distance_metric == "jsd":
            epsilon = F.lit(1e-12)

            p = F.col("group_prob")
            q = F.col("global_prob")

            m = (p + q) / 2

            # Terms with probability zero contribute zero to
            # the corresponding KL divergence.
            p_term = F.when(
                p > 0,
                p * F.log2((p + epsilon) / (m + epsilon))
            ).otherwise(
                F.lit(0.0)
            )

            q_term = F.when(
                q > 0,
                q * F.log2((q + epsilon) / (m + epsilon))
            ).otherwise(
                F.lit(0.0)
            )

            jsd_expr = (
                F.lit(0.5) * p_term
                + F.lit(0.5) * q_term
            )

            return (
                df
                .groupBy(
                    *self.quasi_identifiers
                )
                .agg(
                    F.sum(
                        jsd_expr
                    ).alias("t_closeness")
                )
            )

        raise ValueError(
            f"Unsupported distance metric: "
            f"{self.distance_metric}"
        )


class TClosenessResult:
    """
    Container for t-closeness results.

    The object stores both the global summary generated by the
    t-closeness analysis and the equivalence groups that violate the
    configured privacy threshold.

    Attributes
    ----------
    summary_df : pyspark.sql.DataFrame
        Aggregated statistics of t-closeness values across all equivalence
        groups, including minimum, maximum and average distances.

    violating_groups : pyspark.sql.DataFrame
        Subset of equivalence groups whose t-closeness value exceeds the
        configured threshold, representing potential privacy risks.
    """

    def __init__(
        self,
        summary_df: DataFrame,
        violating_groups: DataFrame
    ):
        """
        Initializes the result object.

        Parameters
        ----------
        summary_df : pyspark.sql.DataFrame
            Aggregated statistics describing the t-closeness values of
            the dataset.

        violating_groups : pyspark.sql.DataFrame
            Equivalence groups whose t-closeness value exceeds the
            configured threshold.
        """
        self.summary_df = summary_df
        self.violating_groups = violating_groups

    def show_summary(self):
        """
        Displays aggregated statistics of t-closeness values.

        The summary includes:

        - num_equivalence_groups:
            Total number of equivalence groups.

        - min_t:
            Minimum t-closeness distance among all equivalence groups.

        - max_t:
            Maximum t-closeness distance among all equivalence groups.

        - avg_t:
            Average t-closeness distance across all equivalence groups.

        - num_violating_groups:
            Number of equivalence groups whose distance exceeds the
            configured t-threshold.

        - t_threshold:
            Maximum allowed distance configured by the user.

        - satisfies_t_closeness:
            Boolean indicating whether every equivalence group satisfies
            the configured t-closeness requirement.

        Interpretation
        --------------
        The dataset satisfies t-closeness when:

            max_t <= t_threshold

        and therefore:

            num_violating_groups = 0
        """
        self.summary_df.show(
            truncate=False
        )

    def show_violating_groups(
        self,
        n: int = 10,
        sort: bool = False
    ):
        """
        Displays equivalence groups that do not satisfy t-closeness.

        Parameters
        ----------
        n : int, optional
            Number of violating groups to display.
            Default is 10.

        sort : bool, optional
            Whether to sort violating groups by t-closeness distance in
            descending order.

            Sorting may increase computational cost for large datasets
            because it requires an additional distributed ordering
            operation.

            Default is False.

        Notes
        -----
        The displayed t-closeness values are rounded to five decimal
        places for readability. The underlying values stored in the
        DataFrame are not modified.
        """
        df = self.violating_groups

        if sort:
            df = df.orderBy(
                F.col("t_closeness").desc()
            )

        df = df.withColumn(
            "t_closeness",
            F.round(
                F.col("t_closeness"),
                5
            )
        )

        df.show(
            n,
            truncate=False
        )

    def get_summary_df(self) -> DataFrame:
        """
        Returns aggregated statistics of t-closeness values.

        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame containing the global t-closeness assessment and
            the configured privacy threshold.
        """
        return self.summary_df

    def get_violating_groups(self) -> DataFrame:
        """
        Returns equivalence groups that do not satisfy t-closeness.

        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame containing the equivalence groups whose t-closeness
            value exceeds the configured threshold.
        """
        return self.violating_groups