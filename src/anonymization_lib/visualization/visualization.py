from pyspark.sql import DataFrame
from pyspark.sql import functions as F
import plotly.graph_objects as go

from ..utils import infer_semantic_type


class Visualization:
    """
    Generates interactive visualizations for a given column in a Spark DataFrame.

    The visualization generated depends on the semantic type inferred for the
    selected column:

    - Numeric columns:
        Bar plot, histogram and box plot.

    - Categorical columns:
        Bar plot and treemap containing the most frequent categories.

    - Date columns:
        Year distribution, month distribution and year-month sunburst.

    The class is designed to perform data-intensive operations using Spark
    before transferring aggregated results to the driver for visualization
    with Plotly.

    Parameters
    ----------
    column : str
        Name of the column to be visualized.

    top_n_categories : int, default=15
        Number of most frequent categories to display for categorical data.

    histogram_bins : int, default=15
        Number of bins used in histogram visualizations for numeric data.

    Notes
    -----
    Plotly operates locally, so aggregated results are converted to Pandas
    before constructing the figures. The implementation minimizes the amount
    of data transferred to the driver by performing filtering, aggregation
    and binning with Spark whenever possible.
    """

    def __init__(
        self,
        column: str,
        top_n_categories: int = 15,
        histogram_bins: int = 15
    ):
        """
        Initializes the visualization object.

        Parameters
        ----------
        column : str
            Name of the column to be visualized.

        top_n_categories : int, default=15
            Number of most frequent categories to display for categorical
            data.

        histogram_bins : int, default=15
            Number of bins used in histogram visualizations for numeric data.

        Raises
        ------
        ValueError
            If ``column`` is not a non-empty string, if
            ``top_n_categories`` is not an integer greater than 1, or if
            ``histogram_bins`` is not a positive integer.
        """
        if not isinstance(column, str) or not column.strip():
            raise ValueError(
                "'column' must be a non-empty string."
            )

        if (
            not isinstance(top_n_categories, int)
            or isinstance(top_n_categories, bool)
        ):
            raise ValueError(
                "'top_n_categories' must be an integer."
            )

        if top_n_categories <= 1:
            raise ValueError(
                "'top_n_categories' must be greater than 1."
            )

        if (
            not isinstance(histogram_bins, int)
            or isinstance(histogram_bins, bool)
        ):
            raise ValueError(
                "'histogram_bins' must be an integer."
            )

        if histogram_bins <= 0:
            raise ValueError(
                "'histogram_bins' must be greater than 0."
            )

        self.column = column
        self.top_n_categories = top_n_categories
        self.histogram_bins = histogram_bins

    def _add_toggle_menu(
        self,
        fig: go.Figure,
        labels: list[str]
    ) -> None:
        """
        Adds an interactive menu to switch between Plotly traces.

        Each button makes one trace visible and hides the remaining traces,
        allowing several visualization types to be stored in the same
        Plotly figure.

        Parameters
        ----------
        fig : plotly.graph_objects.Figure
            Figure containing the traces that will be controlled by the
            toggle menu.

        labels : list of str
            Labels displayed in the interactive menu. The order of the
            labels must correspond to the order of the traces in the figure.

        Returns
        -------
        None
            The figure is modified in place.
        """
        buttons = []
        n = len(fig.data)

        for i, label in enumerate(labels):
            visible = [False] * n
            visible[i] = True

            buttons.append(
                {
                    "label": label,
                    "method": "update",
                    "args": [
                        {"visible": visible},
                        {"title": fig.layout.title.text}
                    ],
                }
            )

        fig.update_layout(
            updatemenus=[
                {
                    "type": "buttons",
                    "direction": "right",
                    "buttons": buttons,
                    "x": 0.5,
                    "xanchor": "center",
                    "y": 1.18,
                    "yanchor": "top",
                    "showactive": True,
                }
            ]
        )

    def _build_numeric_figure(
        self,
        df: DataFrame
    ) -> go.Figure:
        """
        Builds interactive visualizations for a numeric column.

        Three visualization types are generated:

        - Barplot:
            Displays the frequency distribution of the numeric values when
            the number of distinct values is small enough to visualize
            individually.

            When the number of distinct values exceeds
            ``top_n_categories``, the most frequent values are displayed
            instead of transferring the complete frequency distribution
            to the driver.

        - Histogram:
            Displays the distribution using ``histogram_bins`` bins.
            Histogram binning is performed by Spark so that only aggregated
            bin counts are transferred to the driver.

        - Boxplot:
            Displays approximate quartiles together with the minimum,
            maximum and mean values.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input Spark DataFrame containing the numeric column.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive Plotly figure containing the numeric
            visualizations.

        Raises
        ------
        ValueError
            If the selected column contains no valid numeric values.

        Notes
        -----
        The method avoids collecting all distinct numeric values to the
        driver. Aggregation and histogram binning are performed with Spark,
        making the visualization suitable for high-cardinality numeric
        columns and large datasets.

        Quartiles are calculated using Spark's approximate quantile
        algorithm with a relative error of 0.01.
        """
        values_df = (
            df
            .select(
                F.col(self.column)
                .cast("double")
                .alias(self.column)
            )
            .dropna()
        )

        # ------------------------------------------------------
        # BASIC STATISTICS
        # ------------------------------------------------------

        stats = (
            values_df
            .agg(
                F.min(self.column).alias("min_value"),
                F.max(self.column).alias("max_value"),
                F.mean(self.column).alias("mean_value"),
                F.count("*").alias("count")
            )
            .first()
        )

        if stats["count"] == 0:
            raise ValueError(
                f"Column '{self.column}' contains no valid numeric values."
            )

        min_value = stats["min_value"]
        max_value = stats["max_value"]
        mean_value = stats["mean_value"]

        # ------------------------------------------------------
        # QUARTILES
        # ------------------------------------------------------

        q1, median, q3 = values_df.approxQuantile(
            self.column,
            [0.25, 0.5, 0.75],
            0.01
        )

        # ------------------------------------------------------
        # BARPLOT
        # ------------------------------------------------------
        #
        # Only a bounded number of values is collected to the
        # driver. This prevents high-cardinality numeric columns
        # from generating very large Pandas DataFrames.
        # ------------------------------------------------------

        bar_counts = (
            values_df
            .groupBy(self.column)
            .agg(
                F.count("*").alias("frequency")
            )
            .orderBy(
                F.desc("frequency"),
                F.asc(self.column)
            )
            .limit(self.top_n_categories)
            .toPandas()
        )

        # Sort selected values numerically for a more natural
        # representation in the barplot.
        bar_counts = bar_counts.sort_values(
            self.column
        )

        # ------------------------------------------------------
        # HISTOGRAM
        # ------------------------------------------------------

        if min_value == max_value:
            histogram_df = (
                values_df
                .agg(
                    F.count("*").alias("frequency")
                )
                .withColumn(
                    "bin_center",
                    F.lit(float(min_value))
                )
                .select(
                    "bin_center",
                    "frequency"
                )
            )

            bin_width = 1.0

        else:
            bin_width = (
                (max_value - min_value)
                / self.histogram_bins
            )

            bin_index = F.floor(
                (F.col(self.column) - F.lit(min_value))
                / F.lit(bin_width)
            )

            # The maximum value would otherwise fall into bin
            # histogram_bins instead of histogram_bins - 1.
            bin_index = F.when(
                bin_index >= self.histogram_bins,
                F.lit(self.histogram_bins - 1)
            ).otherwise(bin_index)

            histogram_df = (
                values_df
                .withColumn(
                    "_bin",
                    bin_index.cast("int")
                )
                .groupBy("_bin")
                .agg(
                    F.count("*").alias("frequency")
                )
                .withColumn(
                    "bin_center",
                    F.lit(min_value)
                    + (
                        F.col("_bin") + F.lit(0.5)
                    ) * F.lit(bin_width)
                )
                .select(
                    "bin_center",
                    "frequency"
                )
                .orderBy("bin_center")
            )

        histogram = histogram_df.toPandas()

        # ------------------------------------------------------
        # FIGURE
        # ------------------------------------------------------

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=bar_counts[self.column],
                y=bar_counts["frequency"],
                name="Barplot"
            )
        )

        fig.add_trace(
            go.Bar(
                x=histogram["bin_center"],
                y=histogram["frequency"],
                width=bin_width,
                name="Histogram",
                visible=False
            )
        )

        fig.add_trace(
            go.Box(
                q1=[q1],
                median=[median],
                mean=[mean_value],
                q3=[q3],
                lowerfence=[min_value],
                upperfence=[max_value],
                name="Boxplot",
                visible=False,
                boxpoints=False
            )
        )

        fig.update_layout(
            title=(
                f"{self.column.title()} "
                f"\t|\t(histogram_bins={self.histogram_bins})"
            ),
            xaxis_title=self.column,
            yaxis_title="Count",
            legend_title="Chart",
            template="plotly_white"
        )

        self._add_toggle_menu(
            fig,
            ["Barplot", "Histogram", "Boxplot"]
        )

        return fig

    def _build_categorical_figure(
        self,
        df: DataFrame
    ) -> go.Figure:
        """
        Builds interactive visualizations for a categorical column.

        Two visualization types are generated:

        - Bar:
            Displays the frequency of the most common categories.

        - Treemap:
            Displays the same category frequencies using a hierarchical
            area-based representation.

        Only the ``top_n_categories`` most frequent values are transferred
        from Spark to the driver.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input Spark DataFrame containing the categorical column.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive Plotly figure containing the categorical
            visualizations.

        Raises
        ------
        ValueError
            If the selected column contains no non-null values.

        Notes
        -----
        Aggregation, ordering and limiting are performed by Spark before
        conversion to Pandas. Therefore, the amount of data transferred
        to the driver is bounded by ``top_n_categories``.
        """
        counts_df = (
            df
            .select(self.column)
            .dropna()
            .withColumn(
                self.column,
                F.col(self.column).cast("string")
            )
            .groupBy(self.column)
            .agg(
                F.count("*").alias("count")
            )
            .orderBy(
                F.desc("count")
            )
            .limit(
                self.top_n_categories
            )
        )

        counts = counts_df.toPandas()

        if counts.empty:
            raise ValueError(
                f"Column '{self.column}' contains no non-null values."
            )

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=counts[self.column],
                y=counts["count"],
                name="Bar",
                visible=True
            )
        )

        fig.add_trace(
            go.Treemap(
                labels=counts[self.column],
                parents=[""] * len(counts),
                values=counts["count"],
                name="Treemap",
                visible=False
            )
        )

        fig.update_layout(
            title=(
                f"{self.column.title()}"
                f"\t|\t(top_n_categories="
                f"{self.top_n_categories})"
            ),
            xaxis_title=self.column,
            yaxis_title="Count",
            legend_title="Chart",
            template="plotly_white"
        )

        self._add_toggle_menu(
            fig,
            ["Bar", "Treemap"]
        )

        return fig

    def _build_date_figure(
        self,
        df: DataFrame
    ) -> go.Figure:
        """
        Builds interactive visualizations for a date column.

        Three visualization types are generated:

        - Year:
            Displays the number of records for each year.

        - Month:
            Displays the number of records for each calendar month,
            aggregated across all years.

        - Sunburst:
            Displays the hierarchical year-month distribution.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input Spark DataFrame containing the date column.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive Plotly figure containing the date
            visualizations.

        Raises
        ------
        ValueError
            If the selected column contains no valid date values.

        Notes
        -----
        Spark performs a single aggregation by year and month. The
        resulting DataFrame contains at most one row for each observed
        year-month combination and is therefore small enough to convert
        to Pandas.

        Year-level and month-level distributions are subsequently derived
        locally from this already aggregated result, avoiding multiple
        Spark groupBy operations over the original dataset.
        """
        date_df = (
            df
            .select(
                F.to_date(
                    F.col(self.column)
                ).alias(self.column)
            )
            .dropna()
            .select(
                F.year(
                    F.col(self.column)
                ).alias("year"),
                F.month(
                    F.col(self.column)
                ).alias("month")
            )
        )

        # ------------------------------------------------------
        # SINGLE SPARK AGGREGATION
        # ------------------------------------------------------

        year_month_counts = (
            date_df
            .groupBy(
                "year",
                "month"
            )
            .agg(
                F.count("*").alias("count")
            )
            .orderBy(
                "year",
                "month"
            )
            .toPandas()
        )

        if year_month_counts.empty:
            raise ValueError(
                f"Column '{self.column}' contains no valid date values."
            )

        # ------------------------------------------------------
        # LOCAL AGGREGATIONS
        # ------------------------------------------------------

        year_counts = (
            year_month_counts
            .groupby(
                "year",
                as_index=False
            )["count"]
            .sum()
            .sort_values("year")
        )

        month_counts = (
            year_month_counts
            .groupby(
                "month",
                as_index=False
            )["count"]
            .sum()
            .sort_values("month")
        )

        month_nodes = year_month_counts.copy()

        month_names = {
            1: "January",
            2: "February",
            3: "March",
            4: "April",
            5: "May",
            6: "June",
            7: "July",
            8: "August",
            9: "September",
            10: "October",
            11: "November",
            12: "December"
        }

        month_counts["month_name"] = (
            month_counts["month"]
            .map(month_names)
        )

        # ------------------------------------------------------
        # SUNBURST NODES
        # ------------------------------------------------------

        year_nodes = year_counts.copy()

        year_nodes["id"] = (
            year_nodes["year"]
            .astype(str)
        )

        year_nodes["label"] = (
            year_nodes["year"]
            .astype(str)
        )

        year_nodes["parent"] = ""

        month_nodes["id"] = (
            month_nodes["year"].astype(str)
            + "-"
            + month_nodes["month"]
            .astype(str)
            .str.zfill(2)
        )

        month_nodes["label"] = (
            month_nodes["month"]
            .map(month_names)
        )

        month_nodes["parent"] = (
            month_nodes["year"]
            .astype(str)
        )

        sunburst_ids = (
            year_nodes["id"].tolist()
            + month_nodes["id"].tolist()
        )

        sunburst_labels = (
            year_nodes["label"].tolist()
            + month_nodes["label"].tolist()
        )

        sunburst_parents = (
            year_nodes["parent"].tolist()
            + month_nodes["parent"].tolist()
        )

        sunburst_values = (
            year_nodes["count"].tolist()
            + month_nodes["count"].tolist()
        )

        # ------------------------------------------------------
        # FIGURE
        # ------------------------------------------------------

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=year_counts["year"].astype(str),
                y=year_counts["count"],
                name="Year",
                visible=True
            )
        )

        fig.add_trace(
            go.Bar(
                x=month_counts["month_name"],
                y=month_counts["count"],
                name="Month",
                visible=False
            )
        )

        fig.add_trace(
            go.Sunburst(
                ids=sunburst_ids,
                labels=sunburst_labels,
                parents=sunburst_parents,
                values=sunburst_values,
                branchvalues="total",
                name="Sunburst",
                visible=False,
                sort=False
            )
        )

        fig.update_layout(
            title=f"{self.column.title()}",
            xaxis_title="Date",
            yaxis_title="Count",
            legend_title="Chart",
            template="plotly_white"
        )

        self._add_toggle_menu(
            fig,
            ["Year", "Month", "Sunburst"]
        )

        return fig

    def transform(
        self,
        df: DataFrame
    ) -> go.Figure:
        """
        Generates an interactive visualization for the configured column.

        The semantic type of the selected column is inferred automatically
        and the appropriate visualization strategy is applied.

        Numeric columns are visualized using a barplot, histogram and
        boxplot. Date columns are visualized using yearly, monthly and
        hierarchical year-month distributions. Other columns are treated
        as categorical and visualized using a bar chart and treemap.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input Spark DataFrame containing the column to visualize.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive Plotly figure corresponding to the inferred
            semantic type of the selected column.

        Raises
        ------
        ValueError
            If the input DataFrame is None, is not Spark-compatible, or
            does not contain the configured column.

        Notes
        -----
        The method delegates the visualization process to one of the
        specialized internal methods:

        - ``_build_numeric_figure``
        - ``_build_date_figure``
        - ``_build_categorical_figure``

        Data-intensive operations are performed with Spark before the
        aggregated results are transferred to the driver for Plotly
        visualization.
        """
        if df is None:
            raise ValueError(
                "Input DataFrame cannot be None."
            )

        # Compatible with standard Spark and Spark Connect.
        if not hasattr(df, "select"):
            raise ValueError(
                "Input must be a Spark-compatible DataFrame."
            )

        if self.column not in df.columns:
            raise ValueError(
                f"Column '{self.column}' not found in DataFrame."
            )

        semantic_type = infer_semantic_type(
            df,
            self.column
        )

        if semantic_type == "numeric":
            fig = self._build_numeric_figure(df)

        elif semantic_type == "date":
            fig = self._build_date_figure(df)

        else:
            fig = self._build_categorical_figure(df)

        return fig