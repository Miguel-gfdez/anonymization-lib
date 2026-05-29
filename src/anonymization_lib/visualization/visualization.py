from pyspark.sql import DataFrame
from pyspark.sql import functions as F
import plotly.graph_objects as go
from ..utils import infer_semantic_type


class Visualization:
    """
    Generates interactive visualizations for a given column in a Spark DataFrame.

    """
    def __init__(self, column: str, top_n_categories: int = 15, histogram_bins: int = 15):
        """
        Initializes the visualization object.

        Parameters
        ----------
        column : str
            Name of the column to be visualized.

        top_n_categories : int, default=15
            Number of most frequent categories to display for categorical data.

        histogram_bins : int, default=15
            Number of bins used in histogram visualizations for numeric data.

        """
        if not isinstance(column, str) or not column.strip():
            raise ValueError("'column' must be a non-empty string.")

        if not isinstance(top_n_categories, int):
            raise ValueError("'top_n_categories' must be an integer.")
        if top_n_categories <= 1:
            raise ValueError("'top_n_categories' must be greater than 1.")

        if not isinstance(histogram_bins, int):
            raise ValueError("'histogram_bins' must be an integer.")
        if histogram_bins <= 0:
            raise ValueError("'histogram_bins' must be greater than 0.")

        self.column = column
        self.top_n_categories = top_n_categories
        self.histogram_bins = histogram_bins

    def _add_toggle_menu(self, fig: go.Figure, labels: list[str]) -> None:
        buttons = []
        n = len(fig.data)

        for i, label in enumerate(labels):
            visible = [False] * n
            visible[i] = True
            buttons.append(
                dict(
                    label=label,
                    method="update",
                    args=[
                        {"visible": visible},
                        {"title": fig.layout.title.text}
                    ],
                )
            )

        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    buttons=buttons,
                    x=0.5,
                    xanchor="center",
                    y=1.18,
                    yanchor="top",
                    showactive=True,
                )
            ]
        )

    def _build_numeric_figure(self, df: DataFrame) -> go.Figure:
        counts_df = (df.select(self.column)
            .dropna()
            .groupBy(self.column)
            .agg(F.count("*").alias("frequency"))
            .orderBy(self.column))
        
        pdf = counts_df.toPandas()
        if pdf.empty:
            raise ValueError(f"Column '{self.column}' contains no valid numeric values.")
        
        values_df = df.select(F.col(self.column).cast("double").alias(self.column)).dropna()

        q1, median, q3 = values_df.approxQuantile(
            self.column,
            [0.25, 0.5, 0.75],
            0.01
        )

        min_value = values_df.agg(F.min(self.column)).first()[0]
        max_value = values_df.agg(F.max(self.column)).first()[0]
        mean_value = values_df.agg(F.mean(self.column)).first()[0]


        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=pdf[self.column],
                y=pdf["frequency"],
                name="Barplot"
            )
        )

        fig.add_trace(
            go.Histogram(
                x=pdf[self.column],
                y=pdf["frequency"],
                histfunc="sum",
                nbinsx=self.histogram_bins,
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

        self._add_toggle_menu(fig, ["Barplot", "Histogram", "Boxplot"])
        return fig


    def _build_categorical_figure(self, df: DataFrame) -> go.Figure:
        counts_df = (
            df.select(self.column)
            .dropna()
            .withColumn(self.column, F.col(self.column).cast("string"))
            .groupBy(self.column)
            .agg(F.count("*").alias("count"))
            .orderBy(F.desc("count"))
            .limit(self.top_n_categories)
        )

        counts = counts_df.toPandas()

        if counts.empty:
            raise ValueError(f"Column '{self.column}' contains no non-null values.")

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
            title=f"{self.column.title()}\t|\t(top_n_categories={self.top_n_categories})",
            xaxis_title=self.column,
            yaxis_title="Count",
            legend_title="Chart",
            template="plotly_white"
        )

        self._add_toggle_menu(fig, ["Bar", "Treemap"])
        return fig


    def _build_date_figure(self, df: DataFrame) -> go.Figure:
        date_df = (
            df.select(F.to_date(F.col(self.column)).alias(self.column))
            .dropna()
            .withColumn("year", F.year(F.col(self.column)))
            .withColumn("month", F.month(F.col(self.column)))
        )

        year_counts = (
            date_df.groupBy("year")
                .agg(F.count("*").alias("count"))
                .orderBy("year")
                .toPandas()
        )

        month_counts = (
            date_df.groupBy("month")
                .agg(F.count("*").alias("count"))
                .orderBy("month")
                .toPandas()
        )

        month_nodes = (
            date_df.groupBy("year", "month")
                .agg(F.count("*").alias("count"))
                .orderBy("year", "month")
                .toPandas()
        )

        if year_counts.empty:
            raise ValueError(f"Column '{self.column}' contains no valid date values.")

        month_names = {
            1: "January", 2: "February", 3: "March", 4: "April",
            5: "May", 6: "June", 7: "July", 8: "August",
            9: "September", 10: "October", 11: "November", 12: "December"
        }

        month_counts["month_name"] = month_counts["month"].map(month_names)

        year_nodes = year_counts.copy()
        year_nodes["id"] = year_nodes["year"].astype(str)
        year_nodes["label"] = year_nodes["year"].astype(str)
        year_nodes["parent"] = ""

        month_nodes["id"] = (
            month_nodes["year"].astype(str)
            + "-"
            + month_nodes["month"].astype(str).str.zfill(2)
        )
        month_nodes["label"] = month_nodes["month"].map(month_names)
        month_nodes["parent"] = month_nodes["year"].astype(str)

        sunburst_ids = year_nodes["id"].tolist() + month_nodes["id"].tolist()
        sunburst_labels = year_nodes["label"].tolist() + month_nodes["label"].tolist()
        sunburst_parents = year_nodes["parent"].tolist() + month_nodes["parent"].tolist()
        sunburst_values = year_nodes["count"].tolist() + month_nodes["count"].tolist()

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

        self._add_toggle_menu(fig, ["Year", "Month", "Sunburst"])

        return fig


    def transform(self, df: DataFrame) -> go.Figure:
        if df is None:
            raise ValueError("Input DataFrame cannot be None.")

        if not isinstance(df, DataFrame):
            raise ValueError("Input must be a pyspark.sql.DataFrame.")

        if self.column not in df.columns:
            raise ValueError(f"Column '{self.column}' not found in DataFrame.")

        semantic_type = infer_semantic_type(df, self.column)

        if semantic_type == "numeric":
            fig = self._build_numeric_figure(df)
        elif semantic_type == "date":
            fig = self._build_date_figure(df)
        else:
            fig = self._build_categorical_figure(df)

        return fig





