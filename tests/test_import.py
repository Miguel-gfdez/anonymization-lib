import os
import shutil
import tempfile
import unittest

from pyspark.sql import SparkSession, DataFrame

from anonymization_lib import DataImporter


class TestDataImporter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder
            .master("local[2]")
            .appName("test_data_importer")
            .getOrCreate()
        )

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

        self.df = self.spark.createDataFrame(
            [
                (1, "Alice"),
                (2, "Bob"),
                (3, "Charlie"),
            ],
            ["id", "name"]
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_invalid_format_raises_value_error(self):
        path = os.path.join(self.temp_dir, "data.parquet")

        with self.assertRaises(ValueError):
            DataImporter.import_data(
                self.spark,
                path,
                file_format="json"
            )

    def test_invalid_path_type_raises_value_error(self):
        with self.assertRaises(ValueError):
            DataImporter.import_data(
                self.spark,
                "",
                file_format="parquet"
            )

    def test_non_existing_path_raises_file_not_found_error(self):
        non_existing_path = os.path.join(self.temp_dir, "missing.parquet")

        with self.assertRaises(FileNotFoundError):
            DataImporter.import_data(
                self.spark,
                non_existing_path,
                file_format="parquet"
            )

    def test_invalid_spark_session_raises_type_error(self):
        path = os.path.join(self.temp_dir, "data.parquet")
        self.df.write.mode("overwrite").parquet(path)

        with self.assertRaises(TypeError):
            DataImporter.import_data(
                "not_a_spark_session",
                path,
                file_format="parquet"
            )

    def test_import_parquet(self):
        path = os.path.join(self.temp_dir, "data.parquet")
        self.df.write.mode("overwrite").parquet(path)

        imported_df = DataImporter.import_data(
            self.spark,
            path,
            file_format="parquet"
        )

        self.assertIsInstance(imported_df, DataFrame)
        self.assertEqual(imported_df.count(), 3)
        self.assertEqual(set(imported_df.columns), {"id", "name"})

    def test_import_parquet_by_default(self):
        path = os.path.join(self.temp_dir, "data_default.parquet")
        self.df.write.mode("overwrite").parquet(path)

        imported_df = DataImporter.import_data(self.spark, path)

        self.assertIsInstance(imported_df, DataFrame)
        self.assertEqual(imported_df.count(), 3)
        self.assertEqual(set(imported_df.columns), {"id", "name"})

    def test_import_csv_with_header_and_infer_schema(self):
        path = os.path.join(self.temp_dir, "data.csv")
        self.df.write.mode("overwrite").option("header", True).csv(path)

        imported_df = DataImporter.import_data(
            self.spark,
            path,
            file_format="csv",
            header=True,
            infer_schema=True
        )

        self.assertIsInstance(imported_df, DataFrame)
        self.assertEqual(imported_df.count(), 3)
        self.assertEqual(set(imported_df.columns), {"id", "name"})

    def test_import_orc(self):
        path = os.path.join(self.temp_dir, "data.orc")
        self.df.write.mode("overwrite").orc(path)

        imported_df = DataImporter.import_data(
            self.spark,
            path,
            file_format="orc"
        )

        self.assertIsInstance(imported_df, DataFrame)
        self.assertEqual(imported_df.count(), 3)
        self.assertEqual(set(imported_df.columns), {"id", "name"})

    def test_s3_format_requires_s3a_path(self):
        local_path = os.path.join(self.temp_dir, "data.parquet")
        self.df.write.mode("overwrite").parquet(local_path)

        with self.assertRaises(ValueError):
            DataImporter.import_data(
                self.spark,
                local_path,
                file_format="s3"
            )


if __name__ == "__main__":
    unittest.main()