from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from ..utils import EquivalenceGroups


class KAnonymity(EquivalenceGroups):
    """
    Calculates k-anonymity for a dataset based on quasi-identifiers.
    """

    def __init__(self, quasi_identifiers=None, k_threshold: int = 2):
        if quasi_identifiers is not None:
            if not isinstance(quasi_identifiers, list):
                raise ValueError("'quasi_identifiers' must be a list of column names.")

            if not all(isinstance(col, str) for col in quasi_identifiers):
                raise ValueError("'quasi_identifiers' must contain only strings.")

            if len(quasi_identifiers) != len(set(quasi_identifiers)):
                raise ValueError("'quasi_identifiers' must not contain duplicate columns.")

        if not isinstance(k_threshold, int) or k_threshold <= 1:
            raise ValueError("'k_threshold' must be an integer greater than 1.")

        self.quasi_identifiers = quasi_identifiers or []
        self.k_threshold = k_threshold

    def summary(self, df: DataFrame):
        """
        Computes k-anonymity in a distributed manner.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input dataset containing the quasi-identifiers.

        Returns
        -------
        KAnonymityResult
            Container object with:
            - summary_df: Aggregated statistics of k-anonymity across equivalence groups
            - violating_groups: Groups that do not satisfy the configured k-threshold
        """

        if df is None:
            raise ValueError("Input DataFrame cannot be None.")

        if not isinstance(df, DataFrame):
            raise ValueError("Input must be a pyspark.sql.DataFrame.")

        self._validate_columns(df, self.quasi_identifiers)

        equivalence_groups = self._build_equivalence_groups(df,self.quasi_identifiers)

        violating_groups = equivalence_groups.filter(F.col("group_size") < self.k_threshold)

        summary_df = equivalence_groups.agg(
            F.count("*").alias("num_equivalence_groups"),
            F.min("group_size").alias("k_value"),
            F.max("group_size").alias("max_group_size"),
            F.round(F.avg("group_size"), 2).alias("avg_group_size"),
            F.sum(
                F.when(F.col("group_size") < self.k_threshold, 1).otherwise(0)
            ).alias("num_violating_groups")
        )

        summary_df = (
            summary_df
            .withColumn("k_threshold", F.lit(self.k_threshold))
            .withColumn(
                "satisfies_k_anonymity",
                F.col("k_value") >= F.col("k_threshold")
            )
        )

        return KAnonymityResult(
            summary_df=summary_df,
            violating_groups=violating_groups
        )


class KAnonymityResult:
    """
    Container for k-anonymity results.
    """

    def __init__(self, summary_df, violating_groups):
        """
        Initializes the result object.

        Parameters
        ----------
        summary_df : pyspark.sql.DataFrame
            Aggregated statistics of k-anonymity across equivalence groups.

        violating_groups : pyspark.sql.DataFrame
            Subset of equivalence groups whose size is below the configured
            k-threshold, representing privacy risks.
        """
        self.summary_df = summary_df
        self.violating_groups = violating_groups

    def show_summary(self):
        """
        Displays summary statistics of k-anonymity.
        """
        self.summary_df.show(truncate=False)

    def show_violating_groups(self, n: int = 10, sort: bool = False):
        """
        Displays equivalence groups that do not satisfy k-anonymity.

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
            df = df.orderBy(F.col("group_size").asc())

        df.show(n, truncate=False)