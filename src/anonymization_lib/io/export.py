from pyspark.sql import DataFrame
import os


class DataExporter:
    """
    Utility class for exporting Spark DataFrames to different storage formats.

    Supported formats:
    - csv
    - parquet
    - orc
    """

    VALID_FORMATS = {"csv", "parquet", "orc"}
    VALID_MODES = {"overwrite", "append", "ignore", "error", "errorifexists"}

    @staticmethod
    def export(
        df: DataFrame,
        path: str,
        file_format: str = "parquet",
        mode: str = "overwrite",
        header: bool = True
    ) -> str:
        """
        Exports a Spark DataFrame to the specified storage format.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Spark DataFrame to export.
        path : str
            Destination path where the dataset will be written.
        file_format : str, default="parquet"
            Output format. Supported values are:
            - 'csv'
            - 'parquet'
            - 'orc'
        mode : str, default="overwrite"
            Spark write mode. Supported values are:
            - 'overwrite'
            - 'append'
            - 'ignore'
            - 'error'
            - 'errorifexists'
        header : bool, default=True
            Whether to include a header when exporting to CSV.
            Ignored for Parquet and ORC.

        Returns
        -------
        str
            Confirmation message indicating that the export was completed.
        """
        if not isinstance(df, DataFrame):
            raise TypeError("df must be a pyspark.sql.DataFrame.")

        if file_format not in DataExporter.VALID_FORMATS:
            raise ValueError("file_format must be 'csv', 'parquet' or 'orc'.")

        if mode not in DataExporter.VALID_MODES:
            raise ValueError(
                "mode must be one of: 'overwrite', 'append', 'ignore', "
                "'error', 'errorifexists'."
            )

        if not isinstance(path, str) or not path.strip():
            raise ValueError("The destination path must be a non-empty string.")

        parent_dir = os.path.dirname(os.path.abspath(path))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        writer = df.write.mode(mode)

        if file_format == "csv":
            writer.option("header", header).csv(path)
        elif file_format == "parquet":
            writer.parquet(path)
        elif file_format == "orc":
            writer.orc(path)

        return (
            f"Dataset successfully exported in '{file_format}' format to: "
            f"{os.path.abspath(path)}"
        )  

