import os
from pyspark.sql import DataFrame, SparkSession


class DataImporter:
    """
    Utility class for importing datasets into Spark DataFrames.

    Supported formats:
    - csv
    - parquet
    - orc
    - s3
    """

    VALID_FORMATS = {"csv", "parquet", "orc", "s3"}

    @staticmethod
    def import_data(
        spark: SparkSession,
        path: str,
        file_format: str = "parquet",
        header: bool = True,
        infer_schema: bool = True
    ) -> DataFrame:
        """
        Imports a dataset into a Spark DataFrame.

        Parameters
        ----------
        spark : pyspark.sql.SparkSession
            Active Spark session.
        path : str
            Source dataset path.
        file_format : str, default="parquet"
            Input format. Supported values are:
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

        Returns
        -------
        pyspark.sql.DataFrame
            Imported Spark DataFrame.

        Raises
        ------
        TypeError
            If spark is not a SparkSession.
        ValueError
            If path or file_format are invalid.
        FileNotFoundError
            If the local source path does not exist.
        """
        if not isinstance(spark, SparkSession):
            raise TypeError("spark must be a pyspark.sql.SparkSession.")

        if file_format not in DataImporter.VALID_FORMATS:
            raise ValueError(
                f"Unsupported format. Use one of: {DataImporter.VALID_FORMATS}"
            )

        if not isinstance(path, str) or not path.strip():
            raise ValueError("The source path must be a non-empty string.")

        if file_format == "s3":
            if not path.startswith("s3a://"):
                raise ValueError("S3 paths must start with 's3a://'")
        else:
            normalized_path = os.path.abspath(path)
            if not os.path.exists(normalized_path):
                raise FileNotFoundError(
                    f"The source path does not exist: {normalized_path}"
                )

        if file_format == "csv":
            return (
                spark.read
                .option("header", header)
                .option("inferSchema", infer_schema)
                .csv(path)
            )

        if file_format == "parquet":
            return spark.read.parquet(path)

        if file_format == "orc":
            return spark.read.orc(path)

        if file_format == "s3":
            return spark.read.parquet(path)