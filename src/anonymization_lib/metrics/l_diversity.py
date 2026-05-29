from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from ..utils import EquivalenceGroups


class LDiversity(EquivalenceGroups):
    """
    Calculates l-diversity for a dataset based on quasi-identifiers
    and one sensitive attribute.
    """

    def __init__(self, quasi_identifiers=None, sensitive_attribute=None, l_threshold: int = 2):
        self.quasi_identifiers = quasi_identifiers or []

        if not sensitive_attribute:
            raise ValueError("A sensitive attribute must be provided.")

        if not isinstance(sensitive_attribute, str):
            raise ValueError("'sensitive_attribute' must be a single column name (string).")

        if not self.quasi_identifiers:
            raise ValueError("At least one quasi-identifier must be provided.")

        if not isinstance(self.quasi_identifiers, list) or not all(isinstance(col, str) for col in self.quasi_identifiers):
            raise ValueError("'quasi_identifiers' must be a list of column names.")

        if sensitive_attribute in self.quasi_identifiers:
            raise ValueError("'sensitive_attribute' must not be included in 'quasi_identifiers'.")

        if len(self.quasi_identifiers) != len(set(self.quasi_identifiers)):
            raise ValueError("'quasi_identifiers' must not contain duplicate columns.")

        if not isinstance(l_threshold, int) or l_threshold <= 1:
            raise ValueError("'l_threshold' must be an integer greater than 1.")

        self.sensitive_attribute = sensitive_attribute
        self.l_threshold = l_threshold

    def summary(self, df: DataFrame):
        """
        Computes l-diversity in a distributed manner.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input dataset containing quasi-identifiers and the sensitive attribute.

        Returns
        -------
        LDiversityResult
            Container object with:
            - summary_df: Aggregated statistics of l-diversity across equivalence groups
            - violating_groups: Groups that do not satisfy the configured l-threshold
        """
        if df is None:
            raise ValueError("Input DataFrame cannot be None.")

        if not isinstance(df, DataFrame):
            raise ValueError("Input must be a pyspark.sql.DataFrame.")

        self._validate_columns(df, self.quasi_identifiers + [self.sensitive_attribute])

        equivalence_groups = self._build_equivalence_groups(
            df,
            self.quasi_identifiers,
            extra_aggs=[
                F.countDistinct(self.sensitive_attribute).alias("l_diversity")
            ]
        )

        violating_groups = equivalence_groups.filter(
            F.col("l_diversity") < self.l_threshold
        )

        summary_df = equivalence_groups.agg(
            F.count("*").alias("num_equivalence_groups"),
            F.min("l_diversity").alias("l_value"),
            F.max("l_diversity").alias("max_l"),
            F.round(F.avg("l_diversity"), 2).alias("avg_l"),
            F.sum(
                F.when(F.col("l_diversity") < self.l_threshold, 1).otherwise(0)
            ).alias("num_violating_groups")
        )

        summary_df = (
            summary_df
            .withColumn("l_threshold", F.lit(self.l_threshold))
            .withColumn(
                "satisfies_l_diversity",
                F.col("l_value") >= F.col("l_threshold")
            )
        )

        return LDiversityResult(
            summary_df=summary_df,
            violating_groups=violating_groups
        )


class LDiversityResult:
    """
    Container for l-diversity results.
    """

    def __init__(self, summary_df, violating_groups):
        """
        Initializes the result object.

        Parameters
        ----------
        summary_df : pyspark.sql.DataFrame
            Aggregated statistics of l-diversity across equivalence groups.

        violating_groups : pyspark.sql.DataFrame
            Subset of equivalence groups whose size is below the configured
            l-threshold, representing privacy risks.
        """
        self.summary_df = summary_df
        self.violating_groups = violating_groups

    def show_summary(self):
        """
        Displays summary statistics of l-diversity.
        """
        self.summary_df.show(truncate=False)

    def show_violating_groups(self, n: int = 10, sort: bool = False):
        """
        Displays equivalence groups that do not satisfy l-diversity.

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
            df = df.orderBy(F.col("l_diversity").asc())

        df.show(n, truncate=False)