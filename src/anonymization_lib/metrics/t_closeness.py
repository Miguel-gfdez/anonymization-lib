from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from ..utils import EquivalenceGroups


class TCloseness(EquivalenceGroups):
    """
    Calculates t-closeness for a dataset based on quasi-identifiers
    and one sensitive attribute.
    """

    def __init__(self,quasi_identifiers=None,sensitive_attribute=None,t_threshold=None,distance_metric="emd"):
        """
        Parameters
        ----------
        quasi_identifiers : list of str
            Columns used to form equivalence groups (quasi-identifiers).
            These are attributes that could potentially re-identify individuals.

        sensitive_attribute : str
            Column representing the sensitive attribute whose distribution
            must be protected (e.g., disease, salary, diagnosis).

        t_threshold : float
            Maximum allowed distance between the local (group) distribution
            and the global distribution. Defines the privacy guarantee.

        distance_metric : str, optional
            Metric used to compare distributions. Supported values:
            - "l1"  : Total variation distance (default lightweight option)
            - "jsd" : Jensen-Shannon divergence (stable and symmetric)
            - "emd" : Earth Mover’s Distance (approximation in categorical case)
        """

        if not sensitive_attribute:
            raise ValueError("A sensitive attribute must be provided.")

        if not isinstance(sensitive_attribute, str):
            raise ValueError("'sensitive_attribute' must be a string.")

        if not quasi_identifiers:
            raise ValueError("At least one quasi-identifier must be provided.")

        if not isinstance(quasi_identifiers, list):
            raise ValueError("'quasi_identifiers' must be a list of strings.")

        if sensitive_attribute in quasi_identifiers:
            raise ValueError("Sensitive attribute cannot be a quasi-identifier.")

        if len(quasi_identifiers) != len(set(quasi_identifiers)):
            raise ValueError("Duplicate quasi-identifiers detected.")

        
        if not isinstance(t_threshold, (int, float)):
            raise ValueError("'t_threshold' must be a numeric value.")

        if t_threshold <= 0:
            raise ValueError("'t_threshold' must be greater than 0.")

        valid_metrics = ["l1", "jsd", "emd"]
        if distance_metric not in valid_metrics:
            raise ValueError(f"Unsupported metric. Use {valid_metrics}")

        self.quasi_identifiers = quasi_identifiers
        self.sensitive_attribute = sensitive_attribute
        self.t_threshold = t_threshold
        self.distance_metric = distance_metric


    def summary(self, df: DataFrame):
        """
        Computes t-closeness in a distributed manner.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input dataset containing quasi-identifiers and the sensitive attribute.

        Returns
        -------
        TClosenessResult
            Container object with:
            - summary_df: aggregated statistics
            - violating_groups: groups exceeding t_threshold
        """

        if df is None:
            raise ValueError("Input DataFrame cannot be None.")

        if not isinstance(df, DataFrame):
            raise ValueError("Input must be a pyspark.sql.DataFrame.")

        self._validate_columns(df, self.quasi_identifiers + [self.sensitive_attribute])

        # -----------------------------
        # GLOBAL DISTRIBUTION
        # -----------------------------
        total_count = df.count()

        global_dist = (
            df.groupBy(self.sensitive_attribute)
            .agg(F.count("*").alias("count"))
            .withColumn("global_prob", F.col("count") / F.lit(total_count))
            .drop("count")
        )

        # -----------------------------
        # GROUP DISTRIBUTIONS
        # -----------------------------
        group_dist = (
            df.groupBy(self.quasi_identifiers + [self.sensitive_attribute])
            .agg(F.count("*").alias("count"))
        )

        group_totals = group_dist.groupBy(self.quasi_identifiers).agg(
            F.sum("count").alias("group_total")
        )

        group_dist = group_dist.join(group_totals, on=self.quasi_identifiers)

        group_dist = group_dist.withColumn(
            "group_prob",
            F.col("count") / F.col("group_total")
        )

        # -----------------------------
        # JOIN GLOBAL
        # -----------------------------
        joined = group_dist.join(
            global_dist,
            on=self.sensitive_attribute,
            how="left"
        )

        # -----------------------------
        # DISTANCE COMPUTATION
        # -----------------------------
        t_closeness_df = self._compute_distance(joined)

        # -----------------------------
        # VIOLATIONS
        # -----------------------------
        violating_groups = (
            t_closeness_df
            .filter(F.col("t_closeness") > self.t_threshold))

        # -----------------------------
        # SUMMARY
        # -----------------------------
        summary_df = t_closeness_df.agg(
            F.count("*").alias("num_equivalence_groups"),
            F.min("t_closeness").alias("min_t"),
            F.max("t_closeness").alias("max_t"),
            F.avg("t_closeness").alias("avg_t"),
            F.sum(
                F.when(F.col("t_closeness") > self.t_threshold, 1).otherwise(0)
            ).alias("num_violating_groups")
        )

        summary_df = (
            summary_df
            .withColumn("t_threshold", F.lit(self.t_threshold))
            .withColumn(
                "satisfies_t_closeness",
                F.col("max_t") <= F.col("t_threshold")
            )
        )


        return TClosenessResult(
            summary_df=summary_df,
            violating_groups=violating_groups
        )

    def _compute_distance(self, df):
        """
        Computes distance depending on selected metric.
        """

        if self.distance_metric == "l1":
            diff = df.withColumn(
                "abs_diff",
                F.abs(F.col("group_prob") - F.col("global_prob"))
            )

            return diff.groupBy(self.quasi_identifiers).agg(
                (F.sum("abs_diff") / 2).alias("t_closeness")
            )

        elif self.distance_metric == "jsd":
            epsilon = 1e-12

            p = F.col("group_prob") + F.lit(epsilon)
            q = F.col("global_prob") + F.lit(epsilon)
            m = (p + q) / 2

            jsd_expr = F.lit(0.5) * ((p * F.log2(p / m)) + (q * F.log2(q / m)))

            return df.groupBy(self.quasi_identifiers).agg(
                F.sum(jsd_expr).alias("t_closeness")
            )

        elif self.distance_metric == "emd":
            # simplified EMD (L1 approximation for categorical distributions)
            diff = df.withColumn(
                "abs_diff",
                F.abs(F.col("group_prob") - F.col("global_prob"))
            )

            return diff.groupBy(self.quasi_identifiers).agg(
                (F.sum("abs_diff") / 2).alias("t_closeness")
            )



class TClosenessResult:
    """
    Container for t-closeness results.
    """

    def __init__(self,summary_df,violating_groups):
        """
        Initializes the result object.

        Parameters
        ----------
        summary_df : pyspark.sql.DataFrame
            Aggregated statistics of t-closeness values across all groups,
            including minimum, maximum, and average distances.

        violating_groups : pyspark.sql.DataFrame
            Subset of equivalence groups whose t-closeness value exceeds
            the specified threshold, representing privacy risks.

        """
        self.summary_df = summary_df
        self.violating_groups = violating_groups
        
    def show_summary(self):
        """
        Displays aggregated statistics of t-closeness values across all
        equivalence groups, including min, max, and average distance.
        """
        self.summary_df.show(truncate=False)


    def show_violating_groups(self, n: int = 10, sort: bool = False):
        """
        Displays equivalence groups that do not satisfy t-closeness.

        Parameters
        ----------
        n : int, optional
            Number of rows to display (default is 10).

        sort : bool, optional
            Whether to sort groups before displaying them.
            Sorting may increase computational cost for large datasets.
            Default is False.
        """
        df = self.violating_groups

        if sort:
            df = df.orderBy(F.col("t_closeness").desc())

        df = df.withColumn(
            "t_closeness",
            F.round(F.col("t_closeness"), 5)
        )

        df.show(n, truncate=False)



