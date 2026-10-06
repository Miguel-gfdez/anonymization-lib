import os
import shutil
import tempfile
import unittest

from pyspark.sql import SparkSession

from anonymization_lib import DataImporter
from pyspark.errors import AnalysisException


class TestDataImporter(unittest.TestCase):
    """
    Tests for the DataImporter utility.

    The tests verify supported input formats, default behavior,
    parameter validation, Spark-compatible DataFrame results,
    and path validation.
    """

    @classmethod
    def setUpClass(cls):
        """
        Creates a local Spark session shared by all tests.
        """
        cls.spark = (
            SparkSession.builder
            .master("local[2]")
            .appName("test_data_importer")
            .getOrCreate()
        )

    @classmethod
    def tearDownClass(cls):
        """
        Stops the Spark session after all tests have finished.
        """
        cls.spark.stop()

    def setUp(self):
        """
        Creates an isolated temporary directory and a reusable
        DataFrame for each test.
        """
        self.temp_dir = tempfile.mkdtemp()

        self.df = self.spark.createDataFrame(
            [
                (1, "Alice"),
                (2, "Bob"),
                (3, "Charlie"),
            ],
            ["id", "name"],
        )

    def tearDown(self):
        """
        Removes the temporary directory created for the test.
        """
        shutil.rmtree(
            self.temp_dir,
            ignore_errors=True,
        )

    def test_invalid_format_raises_value_error(self):
        """
        Verifies that unsupported input formats are rejected.
        """
        path = os.path.join(
            self.temp_dir,
            "data.parquet",
        )

        with self.assertRaises(ValueError):
            DataImporter.import_data(
                self.spark,
                path,
                file_format="json",
            )

    def test_invalid_path_raises_value_error(self):
        """
        Verifies that an empty input path is rejected.
        """
        with self.assertRaises(ValueError):
            DataImporter.import_data(
                self.spark,
                "",
                file_format="parquet",
            )

    def test_non_existing_path_raises_analysis_exception(self):
        missing_path = os.path.join(
            self.temp_dir,
            "missing.parquet",
        )

        with self.assertRaises(AnalysisException):
            DataImporter.import_data(
                spark=self.spark,
                path=missing_path,
                file_format="parquet",
            )

    def test_invalid_spark_session(self):
        """
        Verifies that an invalid Spark session object is rejected.
        """
        path = os.path.join(
            self.temp_dir,
            "data.parquet",
        )

        self.df.write.mode(
            "overwrite"
        ).parquet(path)

        with self.assertRaises((TypeError, ValueError)):
            DataImporter.import_data(
                "not_a_spark_session",
                path,
                file_format="parquet",
            )

    def test_import_parquet(self):
        """
        Verifies that Parquet datasets can be imported correctly.
        """
        path = os.path.join(
            self.temp_dir,
            "data.parquet",
        )

        self.df.write.mode(
            "overwrite"
        ).parquet(path)

        imported_df = DataImporter.import_data(
            self.spark,
            path,
            file_format="parquet",
        )

        self.assertTrue(
            hasattr(imported_df, "select")
        )
        self.assertTrue(
            hasattr(imported_df, "count")
        )

        self.assertEqual(
            imported_df.count(),
            3,
        )

        self.assertEqual(
            set(imported_df.columns),
            {"id", "name"},
        )

    def test_import_parquet_by_default(self):
        """
        Verifies that Parquet is used as the default input format.
        """
        path = os.path.join(
            self.temp_dir,
            "data_default.parquet",
        )

        self.df.write.mode(
            "overwrite"
        ).parquet(path)

        imported_df = DataImporter.import_data(
            self.spark,
            path,
        )

        self.assertTrue(
            hasattr(imported_df, "select")
        )

        self.assertEqual(
            imported_df.count(),
            3,
        )

        self.assertEqual(
            set(imported_df.columns),
            {"id", "name"},
        )

    def test_import_csv_with_header_and_infer_schema(self):
        """
        Verifies CSV import using a header and schema inference.
        """
        path = os.path.join(
            self.temp_dir,
            "data.csv",
        )

        self.df.write.mode(
            "overwrite"
        ).option(
            "header",
            True,
        ).csv(path)

        imported_df = DataImporter.import_data(
            self.spark,
            path,
            file_format="csv",
            header=True,
            infer_schema=True,
        )

        self.assertTrue(
            hasattr(imported_df, "select")
        )

        self.assertEqual(
            imported_df.count(),
            3,
        )

        self.assertEqual(
            set(imported_df.columns),
            {"id", "name"},
        )

        schema = {
            field.name: field.dataType.simpleString()
            for field in imported_df.schema.fields
        }

        self.assertIn(
            schema["id"],
            {"int", "bigint"},
        )

    def test_import_orc(self):
        """
        Verifies that ORC datasets can be imported correctly.
        """
        path = os.path.join(
            self.temp_dir,
            "data.orc",
        )

        self.df.write.mode(
            "overwrite"
        ).orc(path)

        imported_df = DataImporter.import_data(
            self.spark,
            path,
            file_format="orc",
        )

        self.assertTrue(
            hasattr(imported_df, "select")
        )

        self.assertEqual(
            imported_df.count(),
            3,
        )

        self.assertEqual(
            set(imported_df.columns),
            {"id", "name"},
        )

    def test_s3_format_requires_s3a_path(self):
        """
        Verifies that the S3 input format requires an s3a:// path.
        """
        local_path = os.path.join(
            self.temp_dir,
            "data.parquet",
        )

        self.df.write.mode(
            "overwrite"
        ).parquet(local_path)

        with self.assertRaises(ValueError):
            DataImporter.import_data(
                self.spark,
                local_path,
                file_format="s3",
            )


if __name__ == "__main__":
    unittest.main()