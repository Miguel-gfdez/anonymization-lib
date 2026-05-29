import os
import shutil
import unittest

from pyspark.sql import SparkSession
from anonymization_lib import DataExporter



class TestDataExporter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder
            .appName("test_data_exporter")
            .master("local[1]")
            .getOrCreate()
        )

        cls.columns = ["dni", "edad", "cp", "genero", "enfermedad"]
        cls.data = [
            ("*********", 28, "28---", "F", "covid"),
            ("*********", 28, "28---", "F", "hipertension"),
            ("*********", 35, "28---", "F", "covid"),
            ("*********", 35, "28---", "F", "covid"),
            ("*********", 35, "28---", "F", "anemia"),
            ("*********", 35, "28---", "F", "migraña")]

        cls.df = cls.spark.createDataFrame(cls.data, cls.columns)
        cls.base_output_dir = "data/export"
        os.makedirs(cls.base_output_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

        if os.path.exists(cls.base_output_dir):
            shutil.rmtree(cls.base_output_dir)

    def tearDown(self):
        if os.path.exists(self.base_output_dir):
            for name in os.listdir(self.base_output_dir):
                full_path = os.path.join(self.base_output_dir, name)
                if os.path.isdir(full_path):
                    shutil.rmtree(full_path)
                else:
                    os.remove(full_path)

    def test_export_parquet(self):
        path = os.path.join(self.base_output_dir, "parquet_data")
        exporter = DataExporter(file_format="parquet", mode="overwrite")

        message = exporter.export(self.df, path)

        self.assertTrue(os.path.exists(path))
        self.assertIn("Dataset successfully exported", message)
        self.assertIn("parquet", message)

    def test_export_csv(self):
        path = os.path.join(self.base_output_dir, "csv_data")
        exporter = DataExporter(file_format="csv", mode="overwrite", header=True)

        message = exporter.export(self.df, path)

        self.assertTrue(os.path.exists(path))
        self.assertIn("Dataset successfully exported", message)
        self.assertIn("csv", message)

    def test_invalid_format(self):
        with self.assertRaises(ValueError):
            DataExporter(file_format="json")

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            DataExporter(file_format="csv", mode="invalid_mode")

    def test_invalid_path(self):
        exporter = DataExporter(file_format="parquet", mode="overwrite")

        with self.assertRaises(ValueError):
            exporter.export(self.df, "")

    def test_invalid_dataframe(self):
        exporter = DataExporter(file_format="parquet", mode="overwrite")
        path = os.path.join(self.base_output_dir, "invalid_df")

        with self.assertRaises(TypeError):
            exporter.export("not_a_dataframe", path)

    def test_export_message(self):
        path = os.path.join(self.base_output_dir, "message_test")
        exporter = DataExporter(file_format="parquet", mode="overwrite")

        message = exporter.export(self.df, path)

        expected_message = (
            f"Dataset successfully exported in 'parquet' format "
            f"to: {os.path.abspath(path)}"
        )

        self.assertEqual(message, expected_message)


if __name__ == "__main__":
    unittest.main()