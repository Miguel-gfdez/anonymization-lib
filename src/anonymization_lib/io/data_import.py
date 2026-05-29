from pyspark.sql import DataFrame, SparkSession
import os


class DataImporter:
    """
    Imports datasets into Spark DataFrames from different storage formats.

    Supported formats:
    - csv
    - parquet
    - orc
    - s3
    """

    def __init__(self,file_format: str = "parquet",header: bool = True,infer_schema: bool = True):
        """
        Parameters
        ----------
        file_format : str, default="parquet"
            Input format.

            Supported values:
            - 'csv'
            - 'parquet'
            - 'orc'
            - 's3'

        header : bool, default=True
            Whether the CSV file contains a header row.
            Ignored for non-CSV formats.

        infer_schema : bool, default=True
            Whether Spark should infer the schema automatically for CSV files.
            Ignored for non-CSV formats.
        """

        valid_formats = {"csv","parquet","orc","s3"}

        if file_format not in valid_formats:
            raise ValueError(f"Unsupported format. Use one of: {valid_formats}")

        self.file_format = file_format
        self.header = header
        self.infer_schema = infer_schema

    def _validate_path(self, path: str):
        """
        Validates that the source path exists and is valid.

        Parameters
        ----------
        path : str
            Source path.

        Raises
        ------
        ValueError
            If path is empty or not a string.

        FileNotFoundError
            If the local path does not exist.
            S3 paths are excluded from local validation.
        """

        if not isinstance(path, str) or not path.strip():
            raise ValueError("The source path must be a non-empty string.")

        # Skip local validation for S3
        if path.startswith("s3a://"):
            return

        normalized_path = os.path.abspath(path)

        if not os.path.exists(normalized_path):
            raise FileNotFoundError(f"The source path does not exist: {normalized_path}")

    def import_data(self,spark: SparkSession,path: str) -> DataFrame:
        """
        Imports a dataset into a Spark DataFrame.

        Parameters
        ----------
        spark : pyspark.sql.SparkSession
            Active Spark session.

        path : str
            Source dataset path.

        Returns
        -------
        pyspark.sql.DataFrame
            Imported Spark DataFrame.

        Raises
        ------
        TypeError
            If spark is not a SparkSession.
        """

        if not isinstance(spark, SparkSession):
            raise TypeError("spark must be a pyspark.sql.SparkSession.")

        self._validate_path(path)

        if self.file_format == "csv":
            df = (spark.read.option("header", self.header).option("inferSchema", self.infer_schema).csv(path))

        elif self.file_format == "parquet":
            df = spark.read.parquet(path)

        elif self.file_format == "orc":
            df = spark.read.orc(path)

        elif self.file_format == "s3":
            if not path.startswith("s3a://"):
                raise ValueError("S3 paths must start with 's3a://'")

            df = spark.read.parquet(path)

        return df