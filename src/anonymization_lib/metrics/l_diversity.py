from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ..utils import EquivalenceGroups


class LDiversity(EquivalenceGroups):
    """
    Calculates distinct l-diversity for a dataset based on
    quasi-identifiers and one sensitive attribute.

    L-diversity extends k-anonymity by requiring each equivalence group
    to contain sufficient diversity in its sensitive attribute.

    This implementation uses distinct l-diversity. An equivalence group
    satisfies l-diversity when it contains at least ``l_threshold``
    distinct values of the sensitive attribute.

    Parameters
    ----------
    quasi_identifiers : list of str
        Columns used to form equivalence groups (quasi-identifiers).
        These are attributes that could potentially re-identify
        individuals when combined.

    sensitive_attribute : str
        Column representing the sensitive attribute whose diversity
        must be protected (e.g., disease, salary category, diagnosis).

    l_threshold : int, optional
        Minimum number of distinct sensitive-attribute values required
        in every equivalence group.

        The value must be an integer greater than 1.
        Default is 2.

    Notes
    -----
    The implementation uses Spark DataFrame operations and performs
    the computation in a distributed manner.

    The metric implemented is distinct l-diversity. Other variants,
    such as entropy l-diversity or recursive (c, l)-diversity, are not
    computed by this class.
    """

    def __init__(
        self,
        quasi_identifiers=None,
        sensitive_attribute=None,
        l_threshold: int = 2
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
                "'sensitive_attribute' must not be included "
                "in 'quasi_identifiers'."
            )

        if (
            not isinstance(l_threshold, int)
            or isinstance(l_threshold, bool)
            or l_threshold <= 1
        ):
            raise ValueError(
                "'l_threshold' must be an integer greater than 1."
            )

        self.quasi_identifiers = quasi_identifiers
        self.sensitive_attribute = sensitive_attribute
        self.l_threshold = l_threshold

    def summary(self, df: DataFrame):
        """
        Computes distinct l-diversity in a distributed manner.

        The method first creates equivalence groups using the configured
        quasi-identifiers. For each equivalence group, it calculates the
        number of distinct values of the sensitive attribute.

        An equivalence group violates l-diversity when:

            l_diversity < l_threshold

        The method also calculates aggregated statistics describing the
        l-diversity values of all equivalence groups.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input dataset containing the quasi-identifiers and the
            sensitive attribute.

        Returns
        -------
        LDiversityResult
            Container object with:

            - summary_df:
                Aggregated statistics describing l-diversity across all
                equivalence groups.

            - violating_groups:
                Equivalence groups whose number of distinct sensitive
                values is below the configured l-threshold.

        Raises
        ------
        ValueError
            If the input DataFrame is None, is not Spark-compatible, or
            does not contain all required columns.

        Notes
        -----
        The method is lazy. It builds the Spark transformations required
        to calculate the metric, but execution occurs when an action such
        as ``show()``, ``collect()`` or ``count()`` is performed on one
        of the returned DataFrames.
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

        self._validate_columns(
            df,
            self.quasi_identifiers + [self.sensitive_attribute]
        )

        # ------------------------------------------------------
        # EQUIVALENCE GROUPS AND DISTINCT L-DIVERSITY
        # ------------------------------------------------------

        equivalence_groups = self._build_equivalence_groups(
            df,
            self.quasi_identifiers,
            extra_aggs=[
                F.countDistinct(
                    self.sensitive_attribute
                ).alias("l_diversity")
            ]
        )

        # ------------------------------------------------------
        # VIOLATING GROUPS
        # ------------------------------------------------------

        violating_groups = (
            equivalence_groups
            .filter(
                F.col("l_diversity") < self.l_threshold
            )
        )

        # ------------------------------------------------------
        # SUMMARY
        # ------------------------------------------------------

        summary_df = (
            equivalence_groups
            .agg(
                F.count("*").alias(
                    "num_equivalence_groups"
                ),
                F.min("l_diversity").alias(
                    "l_value"
                ),
                F.max("l_diversity").alias(
                    "max_l"
                ),
                F.round(
                    F.avg("l_diversity"),
                    2
                ).alias(
                    "avg_l"
                ),
                F.sum(
                    F.when(
                        F.col("l_diversity")
                        < self.l_threshold,
                        1
                    ).otherwise(0)
                ).alias(
                    "num_violating_groups"
                )
            )
            .withColumn(
                "l_threshold",
                F.lit(self.l_threshold)
            )
            .withColumn(
                "satisfies_l_diversity",
                F.col("l_value")
                >= F.col("l_threshold")
            )
        )

        return LDiversityResult(
            summary_df=summary_df,
            violating_groups=violating_groups
        )


class LDiversityResult:
    """
    Container for l-diversity results.

    The object stores both the aggregated summary generated by the
    l-diversity analysis and the equivalence groups that violate the
    configured l-threshold.

    Attributes
    ----------
    summary_df : pyspark.sql.DataFrame
        Aggregated statistics of l-diversity values across all
        equivalence groups.

    violating_groups : pyspark.sql.DataFrame
        Equivalence groups whose number of distinct sensitive-attribute
        values is below the configured l-threshold.
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
            Aggregated statistics describing the l-diversity values
            across all equivalence groups.

        violating_groups : pyspark.sql.DataFrame
            Equivalence groups that do not satisfy the configured
            l-diversity threshold.
        """
        self.summary_df = summary_df
        self.violating_groups = violating_groups

    def show_summary(self):
        """
        Displays aggregated statistics of l-diversity.

        The summary includes:

        - num_equivalence_groups:
            Total number of equivalence groups.

        - l_value:
            Minimum number of distinct sensitive values found among all
            equivalence groups. This represents the l-diversity value
            of the complete dataset.

        - max_l:
            Maximum number of distinct sensitive values found in an
            equivalence group.

        - avg_l:
            Average number of distinct sensitive values across all
            equivalence groups.

        - num_violating_groups:
            Number of equivalence groups whose l-diversity is below the
            configured threshold.

        - l_threshold:
            Minimum l-diversity value required by the user.

        - satisfies_l_diversity:
            Boolean indicating whether every equivalence group satisfies
            the configured l-diversity requirement.

        Interpretation
        --------------
        The dataset satisfies the configured l-diversity requirement when:

            l_value >= l_threshold

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
        Displays equivalence groups that do not satisfy l-diversity.

        Parameters
        ----------
        n : int, optional
            Number of violating equivalence groups to display.
            Default is 10.

        sort : bool, optional
            Whether to sort violating groups by their l-diversity value
            in ascending order.

            When enabled, the groups with the lowest diversity and
            therefore the greatest potential privacy risk are displayed
            first.

            Sorting may increase computational cost for large datasets
            because it requires an additional distributed ordering
            operation.

            Default is False.
        """
        df = self.violating_groups

        if sort:
            df = df.orderBy(
                F.col("l_diversity").asc()
            )

        df.show(
            n,
            truncate=False
        )

    def get_summary_df(self) -> DataFrame:
        """
        Returns aggregated statistics of l-diversity.

        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame containing the global l-diversity assessment,
            aggregated statistics and the configured privacy threshold.
        """
        return self.summary_df

    def get_violating_groups(self) -> DataFrame:
        """
        Returns equivalence groups that do not satisfy l-diversity.

        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame containing the equivalence groups whose number of
            distinct sensitive-attribute values is below the configured
            l-threshold.
        """
        return self.violating_groups