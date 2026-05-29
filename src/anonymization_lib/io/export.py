from pyspark.sql import DataFrame
import os


class DataExporter:
    """
    Exports anonymized Spark DataFrames to different storage formats.

    Supported formats:
    - csv
    - parquet
    - orc
    """

    def __init__(self, file_format: str = "parquet", mode: str = "overwrite", header: bool = True):
        """
        Parameters
        ----------
        file_format : str, default="parquet"
            Output format. Supported values are 'csv' and 'parquet'.
        mode : str, default="overwrite"
            Write mode used by Spark. Supported values are:
            - 'overwrite': overwrites existing data at the target path.
            - 'append': adds data to the existing dataset.
            - 'ignore': does nothing if the path already exists.
            - 'error' / 'errorifexists': raises an error if the path already exists.
        header : bool, default=True
            Whether to include header when exporting to CSV.
            Ignored for Parquet.
        """
        valid_formats = {"csv", "parquet", "orc"}
        valid_modes = {"overwrite", "append", "ignore", "error", "errorifexists"}

        if file_format not in valid_formats:
            raise ValueError("file_format must be 'csv', 'parquet' or 'orc.")

        if mode not in valid_modes:
            raise ValueError("mode must be one of: 'overwrite', 'append', 'ignore', 'error', 'errorifexists'.")

        self.file_format = file_format
        self.mode = mode
        self.header = header

    def _validate_path(self, path: str):
        """
        Validates that the destination path is a non-empty string and that its
        parent directory exists or can be resolved.

        Parameters
        ----------
        path : str
            Destination path.

        Raises
        ------
        ValueError
            If path is empty or not a string.
        FileNotFoundError
            If the parent directory does not exist.
        """
        if not isinstance(path, str) or not path.strip():
            raise ValueError("The destination path must be a non-empty string.")

        normalized_path = os.path.abspath(path)
        parent_dir = os.path.dirname(normalized_path)

        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
            # raise FileNotFoundError(f"The destination directory does not exist: {parent_dir}")

    def export(self, df: DataFrame, path: str) -> str:
        """
        Exports the provided Spark DataFrame to the configured format.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input dataset to export.
        path : str
            Destination path.

        Returns
        -------
        str
            Confirmation message indicating that the export was completed.

        Raises
        ------
        TypeError
            If df is not a Spark DataFrame.
        """
        if not isinstance(df, DataFrame):
            raise TypeError("df must be a pyspark.sql.DataFrame.")

        self._validate_path(path)

        writer = df.write.mode(self.mode)

        if self.file_format == "csv":
            writer.option("header", self.header).csv(path)

        elif self.file_format == "parquet":
            writer.parquet(path)
        
        elif self.file_format == "orc":
            writer.orc(path)

        return (f"Dataset successfully exported in '{self.file_format}' format to: {os.path.abspath(path)}")