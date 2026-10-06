from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ..utils import EquivalenceGroups


class KAnonymity(EquivalenceGroups):
    """
    Calculates k-anonymity for a dataset based on a set of
    quasi-identifiers.

    K-anonymity measures whether each combination of quasi-identifier
    values is shared by at least ``k`` records.

    Records with the same values for all quasi-identifiers form an
    equivalence group. The size of the smallest equivalence group
    represents the k-anonymity level of the dataset.

    Parameters
    ----------
    quasi_identifiers : list of str, optional
        Columns used as quasi-identifiers to construct equivalence groups.

    k_threshold : int, default=2
        Minimum required size for every equivalence group.

        A dataset satisfies the configured k-anonymity requirement when
        every equivalence group contains at least ``k_threshold`` records.

    Notes
    -----
    Equivalence groups are computed using native Spark operations, allowing
    the calculation to be performed in a distributed manner.

    No dataset records are collected to the driver.

    The computation is lazy. Spark execution occurs only when an action is
    performed on one of the resulting DataFrames.
    """

    def __init__(
        self,
        quasi_identifiers=None,
        k_threshold: int = 2
    ):
        """
        Initializes the k-anonymity metric.

        Parameters
        ----------
        quasi_identifiers : list of str, optional
            Columns used as quasi-identifiers.

            Column names must be non-empty strings and duplicate columns
            are not allowed.

        k_threshold : int, default=2
            Minimum required size of every equivalence group.

            The value must be an integer greater than 1.

        Raises
        ------
        ValueError
            If ``quasi_identifiers`` is not a list, contains invalid or
            duplicate column names, or ``k_threshold`` is not an integer
            greater than 1.
        """
        if quasi_identifiers is not None:
            if not isinstance(quasi_identifiers, list):
                raise ValueError(
                    "'quasi_identifiers' must be a list of column names."
                )

            if not all(
                isinstance(column, str) and column.strip()
                for column in quasi_identifiers
            ):
                raise ValueError(
                    "'quasi_identifiers' must contain only "
                    "non-empty strings."
                )

            if len(quasi_identifiers) != len(set(quasi_identifiers)):
                raise ValueError(
                    "'quasi_identifiers' must not contain "
                    "duplicate columns."
                )

        if (
            not isinstance(k_threshold, int)
            or isinstance(k_threshold, bool)
            or k_threshold <= 1
        ):
            raise ValueError(
                "'k_threshold' must be an integer greater than 1."
            )

        self.quasi_identifiers = quasi_identifiers or []
        self.k_threshold = k_threshold

    def summary(
        self,
        df: DataFrame
    ):
        """
        Computes k-anonymity statistics for the input dataset.

        Records are grouped according to their values for the configured
        quasi-identifiers. Each resulting equivalence group contains a
        ``group_size`` representing the number of records sharing that
        combination of quasi-identifier values.

        The k-anonymity value of the dataset is defined as the size of the
        smallest equivalence group:

            k = min(group_size)

        Equivalence groups whose size is smaller than ``k_threshold`` are
        considered violating groups.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input dataset containing the configured quasi-identifiers.

        Returns
        -------
        KAnonymityResult
            Container containing:

            - ``summary_df``:
                DataFrame with aggregated k-anonymity statistics.

            - ``violating_groups``:
                Equivalence groups whose size is below the configured
                ``k_threshold``.

        Raises
        ------
        ValueError
            If the input DataFrame is None, is not Spark-compatible, or
            does not contain one or more configured quasi-identifiers.

        Notes
        -----
        The computation uses native Spark aggregations and is performed
        in a distributed manner.

        This method does not trigger Spark execution by itself. Both
        returned DataFrames are lazy and are evaluated only when a Spark
        action is performed.

        ``summary_df`` and ``violating_groups`` share the same logical
        equivalence-group computation. Without persistence, separate
        actions on both result DataFrames may cause Spark to recompute
        that shared lineage.
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
            self.quasi_identifiers
        )

        # ------------------------------------------------------
        # EQUIVALENCE GROUPS
        # ------------------------------------------------------

        equivalence_groups = self._build_equivalence_groups(
            df,
            self.quasi_identifiers
        )

        # ------------------------------------------------------
        # VIOLATING GROUPS
        # ------------------------------------------------------

        violating_groups = equivalence_groups.filter(
            F.col("group_size") < self.k_threshold
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
                F.min("group_size").alias(
                    "k_value"
                ),
                F.max("group_size").alias(
                    "max_group_size"
                ),
                F.round(
                    F.avg("group_size"),
                    2
                ).alias(
                    "avg_group_size"
                ),
                F.sum(
                    F.when(
                        F.col("group_size") < self.k_threshold,
                        1
                    ).otherwise(0)
                ).alias(
                    "num_violating_groups"
                )
            )
            .withColumn(
                "k_threshold",
                F.lit(self.k_threshold)
            )
            .withColumn(
                "satisfies_k_anonymity",
                F.col("k_value")
                >= F.col("k_threshold")
            )
        )

        return KAnonymityResult(
            summary_df=summary_df,
            violating_groups=violating_groups
        )


class KAnonymityResult:
    """
    Container for k-anonymity results.

    The object stores both the aggregated summary of the k-anonymity
    analysis and the equivalence groups that violate the configured
    k-threshold.

    Parameters
    ----------
    summary_df : pyspark.sql.DataFrame
        DataFrame containing aggregated k-anonymity statistics.

    violating_groups : pyspark.sql.DataFrame
        Equivalence groups whose size is below the configured k-threshold.

    Notes
    -----
    The stored DataFrames remain lazy Spark DataFrames. Creating this
    result object does not trigger Spark execution.
    """

    def __init__(
        self,
        summary_df: DataFrame,
        violating_groups: DataFrame
    ):
        """
        Initializes the k-anonymity result container.

        Parameters
        ----------
        summary_df : pyspark.sql.DataFrame
            Aggregated statistics of k-anonymity across equivalence groups.

        violating_groups : pyspark.sql.DataFrame
            Subset of equivalence groups whose size is below the configured
            k-threshold, representing potential privacy risks.
        """
        self.summary_df = summary_df
        self.violating_groups = violating_groups

    def show_summary(self) -> None:
        """
        Displays the aggregated k-anonymity summary.

        Returns
        -------
        None
            The summary is displayed using Spark's ``show`` method.

        Notes
        -----
        Calling this method triggers Spark execution.
        """
        self.summary_df.show(
            truncate=False
        )

    def show_violating_groups(
        self,
        n: int = 10,
        sort: bool = False
    ) -> None:
        """
        Displays equivalence groups that do not satisfy k-anonymity.

        Parameters
        ----------
        n : int, default=10
            Maximum number of violating groups to display.

        sort : bool, default=False
            Whether to sort violating groups by ``group_size`` in
            ascending order before displaying them.

            Sorting may increase computational cost for large datasets
            because it can require an additional distributed ordering
            operation.

        Returns
        -------
        None
            The selected violating groups are displayed using Spark's
            ``show`` method.

        Raises
        ------
        ValueError
            If ``n`` is not a positive integer or ``sort`` is not a
            boolean.

        Notes
        -----
        Calling this method triggers Spark execution.
        """
        if (
            not isinstance(n, int)
            or isinstance(n, bool)
            or n <= 0
        ):
            raise ValueError(
                "'n' must be a positive integer."
            )

        if not isinstance(sort, bool):
            raise ValueError(
                "'sort' must be a boolean."
            )

        df = self.violating_groups

        if sort:
            df = df.orderBy(
                F.col("group_size").asc()
            )

        df.show(
            n,
            truncate=False
        )

    def get_summary_df(self) -> DataFrame:
        """
        Returns the aggregated k-anonymity summary.

        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame containing the aggregated k-anonymity statistics.

        Notes
        -----
        Returning the DataFrame does not trigger Spark execution.
        """
        return self.summary_df

    def get_violating_groups(self) -> DataFrame:
        """
        Returns the equivalence groups that violate k-anonymity.

        Returns
        -------
        pyspark.sql.DataFrame
            Equivalence groups whose size is below the configured
            k-threshold.

        Notes
        -----
        Returning the DataFrame does not trigger Spark execution.
        """
        return self.violating_groups