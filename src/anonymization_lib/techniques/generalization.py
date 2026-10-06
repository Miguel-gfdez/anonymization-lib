import json
import os
import warnings

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ..utils import infer_semantic_type


class Generalization:
    """
    Applies generalization to a column in a Spark DataFrame.

    The transformation supports:

    - Categorical generalization using mapping rules.
    - Numerical generalization using interval rules.
    - Temporal generalization using year, month, quarter or semester.

    Generalization rules can be loaded from JSON files stored in locations
    accessible either by the local Python process or by Spark. This allows
    the transformation to work with local files as well as distributed or
    remote storage systems supported by the active Spark environment.

    Examples of supported locations may include:

    - Local filesystem paths.
    - Databricks Workspace files.
    - Databricks Volumes.
    - DBFS paths.
    - Cloud object storage such as S3 or ABFS, when configured in Spark.

    Parameters
    ----------
    column : str
        Name of the column to be generalized.

    rules_path : str, optional
        Path to a JSON file containing the generalization configuration
        and rules.

        The path may refer to a local file or to a location readable by
        Spark.

        Expected JSON structure for categorical generalization::

            {
                "type": "categorical",
                "rules": [
                    {"from": ["A", "B"], "to": "Group 1"},
                    {"from": "C", "to": "Group 2"}
                ]
            }

        Expected JSON structure for numerical generalization::

            {
                "type": "numeric",
                "rules": [
                    {"min": 0, "max": 18, "value": "0-18"},
                    {"min": 19, "max": 35, "value": "19-35"}
                ]
            }

        For temporal generalization, the JSON configuration may contain::

            {
                "type": "date"
            }

        A rules file is required for categorical and numerical
        generalization. Temporal generalization can also be inferred
        directly from the column semantic type when no rules file is
        provided.

    mode : str, optional
        Temporal generalization mode.

        Supported values are:

        - ``"year"``
        - ``"month"``
        - ``"quarter"``
        - ``"semester"``

        This parameter is required when temporal generalization is applied.

    default_value : str, optional
        Value assigned when no categorical rule matches or when a numeric
        value falls outside all configured intervals.

        If not provided, the original value is preserved.

    include_year : bool, default=True
        Controls whether temporal generalization preserves the year.

        Examples with ``include_year=True``:

        - month: ``2024-05``
        - quarter: ``2024-Q2``
        - semester: ``2024-S1``

        Examples with ``include_year=False``:

        - month: ``05``
        - quarter: ``Q2``
        - semester: ``S1``

        The option has no effect when ``mode="year"``.

    output_column : str, optional
        Name assigned to the generalized column.

        If not provided, the original column is overwritten.

        If provided and different from ``column``, the original column is
        generalized and then renamed to ``output_column``. Therefore, the
        original column is not preserved in the resulting DataFrame.

    Notes
    -----
    The transformation itself is performed using Spark expressions and is
    therefore distributed.

    Generalization rules are small configuration objects and are loaded on
    the driver. Dataset records are not collected to the driver.

    Spark-compatible paths are read through Spark, while ordinary local
    paths can be read directly through Python.
    """

    VALID_TEMPORAL_MODES = {
        "year",
        "month",
        "quarter",
        "semester"
    }

    VALID_RULE_TYPES = {
        "categorical",
        "numeric",
        "date"
    }

    def __init__(
        self,
        column: str,
        rules_path: str = None,
        mode: str = None,
        default_value: str = None,
        include_year: bool = True,
        output_column: str = None
    ):
        """
        Initializes the generalization transformation.

        Parameters
        ----------
        column : str
            Name of the column to be generalized.

        rules_path : str, optional
            Path to a JSON configuration containing the generalization
            rules. The path may refer to a local file or to a location
            readable by Spark.

        mode : str, optional
            Temporal generalization mode. Supported values are ``year``,
            ``month``, ``quarter`` and ``semester``.

        default_value : str, optional
            Value assigned when no categorical or numerical rule applies.
            If omitted, the original value is preserved.

        include_year : bool, default=True
            Whether temporal generalization should preserve the year for
            month, quarter and semester modes.

        output_column : str, optional
            Name of the generalized output column. If omitted, the
            original column is overwritten.

        Raises
        ------
        ValueError
            If any configuration parameter has an invalid value or type.
        """
        if not isinstance(column, str) or not column.strip():
            raise ValueError(
                "'column' must be a non-empty string."
            )

        if rules_path is not None:
            if not isinstance(rules_path, str) or not rules_path.strip():
                raise ValueError(
                    "'rules_path' must be a non-empty string."
                )

        if mode is not None:
            if not isinstance(mode, str):
                raise ValueError(
                    "'mode' must be a string."
                )

            if mode not in self.VALID_TEMPORAL_MODES:
                raise ValueError(
                    f"Unsupported mode '{mode}'. "
                    f"Use {sorted(self.VALID_TEMPORAL_MODES)}."
                )

        if not isinstance(include_year, bool):
            raise ValueError(
                "'include_year' must be a boolean."
            )

        if output_column is not None:
            if (
                not isinstance(output_column, str)
                or not output_column.strip()
            ):
                raise ValueError(
                    "'output_column' must be a non-empty string."
                )

        self.column = column
        self.rules_path = rules_path
        self.mode = mode
        self.default_value = default_value
        self.include_year = include_year
        self.output_column = output_column or column

    def _rename_output_column(
        self,
        df: DataFrame
    ) -> DataFrame:
        """
        Renames the generalized column when an output column is configured.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            DataFrame containing the generalized column.

        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame with the generalized column renamed when
            ``output_column`` differs from the original column name.
        """
        if self.output_column != self.column:
            return df.withColumnRenamed(
                self.column,
                self.output_column
            )

        return df

    def _load_rules_config(
        self,
        df: DataFrame
    ) -> dict:
        """
        Loads the generalization configuration from a JSON file.

        The method supports both local files and paths accessible through
        Spark.

        Local files are read directly using Python. Other paths are read
        through Spark's text reader, allowing the active Spark environment
        to resolve distributed filesystems and cloud storage locations.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Spark DataFrame used to access the active Spark session.

        Returns
        -------
        dict
            Parsed JSON configuration.

        Raises
        ------
        ValueError
            If no rules path is configured, the file cannot be read, the
            file is empty, the JSON is invalid, or the root JSON value is
            not an object.

        Notes
        -----
        Only the rules configuration is transferred to the driver. These
        files are expected to be small configuration files rather than
        large datasets.
        """
        if self.rules_path is None:
            raise ValueError(
                "'rules_path' must be provided."
            )

        content = None

        # ------------------------------------------------------
        # LOCAL FILE
        # ------------------------------------------------------

        if os.path.isfile(self.rules_path):
            try:
                with open(
                    self.rules_path,
                    "r",
                    encoding="utf-8"
                ) as file:
                    content = file.read()

            except OSError as exc:
                raise ValueError(
                    f"Unable to read rules file "
                    f"'{self.rules_path}'."
                ) from exc

        # ------------------------------------------------------
        # SPARK-COMPATIBLE PATH
        # ------------------------------------------------------

        else:
            try:
                rows = (
                    df.sparkSession
                    .read
                    .text(self.rules_path)
                    .select("value")
                    .collect()
                )

                if rows:
                    content = "\n".join(
                        row["value"]
                        for row in rows
                    )

            except Exception as exc:
                raise ValueError(
                    f"Unable to read rules file "
                    f"'{self.rules_path}' using either the local "
                    f"filesystem or Spark."
                ) from exc

        if not content or not content.strip():
            raise ValueError(
                f"Rules file '{self.rules_path}' is empty."
            )

        try:
            config = json.loads(content)

        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in rules file "
                f"'{self.rules_path}'."
            ) from exc

        if not isinstance(config, dict):
            raise ValueError(
                "The rules JSON root must be an object."
            )

        return config

    def _get_rules_type(
        self,
        config: dict
    ) -> str:
        """
        Returns and validates the generalization type defined in a rules
        configuration.

        Parameters
        ----------
        config : dict
            Parsed generalization configuration.

        Returns
        -------
        str
            Generalization type defined by the ``type`` field.

        Raises
        ------
        ValueError
            If the configuration does not contain a supported
            generalization type.
        """
        rules_type = config.get("type")

        if rules_type not in self.VALID_RULE_TYPES:
            raise ValueError(
                f"Unsupported generalization type in JSON: "
                f"{rules_type}. "
                f"Use {sorted(self.VALID_RULE_TYPES)}."
            )

        return rules_type

    def _apply_temporal_with_mode(
        self,
        df: DataFrame
    ) -> DataFrame:
        """
        Applies temporal generalization after validating that a temporal
        mode has been configured.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input dataset.

        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame with the temporal column generalized.

        Raises
        ------
        ValueError
            If no temporal generalization mode has been provided.
        """
        if self.mode is None:
            raise ValueError(
                "'mode' must be provided for temporal generalization."
            )

        return self._apply_temporal_generalization(df)

    def _apply_categorical(
        self,
        df: DataFrame,
        config: dict
    ) -> DataFrame:
        """
        Applies categorical generalization using mapping rules.

        Each categorical rule maps one or more original values to a
        generalized value.

        A rule can contain either a single value or a list of values in
        the ``from`` field.

        Example
        -------
        A rule such as::

            {"from": ["Single", "Divorced"], "to": "Not married"}

        maps both original values to ``"Not married"``.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input dataset.

        config : dict
            Parsed JSON configuration containing categorical rules.

        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame with the selected column generalized.

        Raises
        ------
        ValueError
            If no rules or no valid categorical rules are found.

        Notes
        -----
        The mapping is converted into a Spark map expression. The
        transformation is therefore evaluated by Spark workers without
        requiring dataset records to be collected to the driver.
        """
        mapping = {}

        rules = config.get("rules", [])

        if not rules:
            raise ValueError(
                "No rules found in JSON configuration."
            )

        for rule in rules:
            if not isinstance(rule, dict):
                warnings.warn(
                    f"Invalid categorical rule ignored: {rule}"
                )
                continue

            try:
                original = rule["from"]
                general = str(rule["to"])

            except KeyError as exc:
                warnings.warn(
                    f"Invalid rule ignored. "
                    f"Missing key {exc}: {rule}"
                )
                continue

            except (TypeError, ValueError):
                warnings.warn(
                    f"Invalid categorical rule ignored: {rule}"
                )
                continue

            originals = (
                original
                if isinstance(original, list)
                else [original]
            )

            for value in originals:
                mapping[str(value)] = general

        if not mapping:
            raise ValueError(
                "No valid categorical rules found."
            )

        mapping_expr = F.create_map(
            *[
                item
                for key, value in mapping.items()
                for item in (
                    F.lit(key),
                    F.lit(value)
                )
            ]
        )

        source_col = (
            F.col(self.column)
            .cast("string")
        )

        new_col = mapping_expr[source_col]

        if self.default_value is not None:
            new_col = F.coalesce(
                new_col,
                F.lit(str(self.default_value))
            )
        else:
            new_col = F.coalesce(
                new_col,
                source_col
            )

        result = df.withColumn(
            self.column,
            new_col.cast("string")
        )

        return self._rename_output_column(result)

    def _apply_numeric(
        self,
        df: DataFrame,
        config: dict
    ) -> DataFrame:
        """
        Applies numerical generalization using interval rules.

        Each rule defines a closed numerical interval using ``min`` and
        ``max`` together with the generalized label assigned to values
        inside that interval.

        Example
        -------
        A rule such as::

            {"min": 18, "max": 30, "value": "18-30"}

        assigns the value ``"18-30"`` to numeric values satisfying:

            18 <= value <= 30

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input dataset.

        config : dict
            Parsed JSON configuration containing numerical interval rules.

        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame with the selected column generalized.

        Raises
        ------
        ValueError
            If no rules or no valid numerical rules are found.

        Notes
        -----
        Interval rules are converted into Spark ``when`` expressions and
        are evaluated in a distributed manner.

        Intervals are evaluated in the order in which they appear in the
        configuration. If intervals overlap, the first matching rule takes
        precedence.
        """
        rules = config.get("rules", [])

        if not rules:
            raise ValueError(
                "No rules found in JSON configuration."
            )

        expr = None

        numeric_col = (
            F.col(self.column)
            .cast("double")
        )

        for rule in rules:
            if not isinstance(rule, dict):
                warnings.warn(
                    f"Invalid numeric rule ignored: {rule}"
                )
                continue

            try:
                start = float(rule["min"])
                end = float(rule["max"])
                label = str(rule["value"])

            except KeyError as exc:
                warnings.warn(
                    f"Invalid rule ignored. "
                    f"Missing key {exc}: {rule}"
                )
                continue

            except (TypeError, ValueError):
                warnings.warn(
                    f"Invalid numeric values in rule: {rule}"
                )
                continue

            if start > end:
                warnings.warn(
                    f"Invalid interval ignored: {rule}"
                )
                continue

            condition = (
                (numeric_col >= F.lit(start))
                & (numeric_col <= F.lit(end))
            )

            if expr is None:
                expr = F.when(
                    condition,
                    F.lit(label)
                )
            else:
                expr = expr.when(
                    condition,
                    F.lit(label)
                )

        if expr is None:
            raise ValueError(
                "No valid numerical rules found."
            )

        if self.default_value is not None:
            expr = expr.otherwise(
                F.lit(str(self.default_value))
            )
        else:
            expr = expr.otherwise(
                F.col(self.column).cast("string")
            )

        result = df.withColumn(
            self.column,
            expr.cast("string")
        )

        return self._rename_output_column(result)

    def _apply_temporal_generalization(
        self,
        df: DataFrame
    ) -> DataFrame:
        """
        Applies temporal generalization based on the selected aggregation
        level.

        Supported modes are year, month, quarter and semester.

        By default, ``include_year`` is True, meaning that generalized
        values preserve the year whenever applicable.

        Examples
        --------
        With ``include_year=True``:

        - year: ``2024``
        - month: ``2024-05``
        - quarter: ``2024-Q2``
        - semester: ``2024-S1``

        With ``include_year=False``:

        - year: ``2024``
        - month: ``05``
        - quarter: ``Q2``
        - semester: ``S1``

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input dataset.

        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame with the temporal column generalized.

        Raises
        ------
        ValueError
            If the configured temporal mode is unsupported.
        """
        col = F.col(self.column)

        if self.mode == "year":
            new_col = (
                F.year(col)
                .cast("string")
            )

        elif self.mode == "month":
            if self.include_year:
                new_col = F.date_format(
                    col,
                    "yyyy-MM"
                )
            else:
                new_col = F.date_format(
                    col,
                    "MM"
                )

        elif self.mode == "quarter":
            quarter = F.concat(
                F.lit("Q"),
                F.quarter(col)
            )

            if self.include_year:
                new_col = F.concat(
                    F.year(col),
                    F.lit("-"),
                    quarter
                )
            else:
                new_col = quarter

        elif self.mode == "semester":
            semester = F.concat(
                F.lit("S"),
                F.when(
                    F.month(col) <= 6,
                    1
                ).otherwise(2)
            )

            if self.include_year:
                new_col = F.concat(
                    F.year(col),
                    F.lit("-"),
                    semester
                )
            else:
                new_col = semester

        else:
            raise ValueError(
                f"Unsupported mode '{self.mode}'."
            )

        result = df.withColumn(
            self.column,
            new_col.cast("string")
        )

        return self._rename_output_column(result)

    def transform(
        self,
        df: DataFrame
    ) -> DataFrame:
        """
        Applies generalization to the configured column.

        The transformation strategy is selected from either the external
        rules configuration or the semantic type of the target column.

        When ``rules_path`` is provided, the JSON configuration is loaded
        once and its ``type`` field determines the transformation:

        - ``categorical``:
            Categorical mapping rules are applied.

        - ``numeric``:
            Numerical interval rules are applied.

        - ``date``:
            Temporal generalization is applied using the configured mode.

        When no rules file is provided, temporal columns can be generalized
        directly using their inferred semantic type. Numerical and
        categorical generalization require external rules.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input dataset.

        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame with the selected column generalized.

        Raises
        ------
        ValueError
            If the input DataFrame is None, is not Spark-compatible, does
            not contain the configured column, contains an unsupported
            rules type, or requires rules that were not provided.

        Notes
        -----
        The method only constructs Spark transformations. Dataset records
        are not collected to the driver.

        When external rules are used, only the small JSON configuration
        itself is loaded on the driver.
        """
        if df is None:
            raise ValueError(
                "Input DataFrame cannot be None."
            )

        # Compatible with standard Spark and Spark Connect.
        if not hasattr(df, "withColumn"):
            raise ValueError(
                "Input must be a Spark-compatible DataFrame."
            )

        if self.column not in df.columns:
            raise ValueError(
                f"Column '{self.column}' does not exist "
                f"in the DataFrame."
            )

        # ------------------------------------------------------
        # EXTERNAL RULES
        # ------------------------------------------------------

        if self.rules_path is not None:
            config = self._load_rules_config(df)
            rules_type = self._get_rules_type(config)

            if rules_type == "numeric":
                return self._apply_numeric(
                    df,
                    config
                )

            if rules_type == "categorical":
                return self._apply_categorical(
                    df,
                    config
                )

            if rules_type == "date":
                return self._apply_temporal_with_mode(df)

        # ------------------------------------------------------
        # AUTOMATIC TEMPORAL GENERALIZATION
        # ------------------------------------------------------

        semantic_type = infer_semantic_type(
            df,
            self.column
        )

        if semantic_type == "date":
            return self._apply_temporal_with_mode(df)

        raise ValueError(
            "For numeric or categorical generalization, "
            "'rules_path' must be provided."
        )