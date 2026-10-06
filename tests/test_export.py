import os
import shutil
import tempfile
import unittest

from pyspark.sql import SparkSession

from anonymization_lib import DataExporter


class TestDataExporter(unittest.TestCase):
    """
    Tests for the DataExporter utility.

    The tests verify supported export formats, default behavior,
    parameter validation, generated output paths, and returned messages.
    """

    @classmethod
    def setUpClass(cls):
        """
        Creates a local Spark session and a reusable test DataFrame.
        """
        cls.spark = (
            SparkSession.builder
            .master("local[2]")
            .appName("test_data_exporter")
            .getOrCreate()
        )

        cls.columns = [
            "dni",
            "edad",
            "cp",
            "genero",
            "enfermedad",
        ]

        cls.data = [
            ("*********", 28, "28---", "F", "covid"),
            ("*********", 28, "28---", "F", "hipertension"),
            ("*********", 35, "28---", "F", "covid"),
            ("*********", 35, "28---", "F", "covid"),
            ("*********", 35, "28---", "F", "anemia"),
            ("*********", 35, "28---", "F", "migraña"),
        ]

        cls.df = cls.spark.createDataFrame(
            cls.data,
            cls.columns,
        )

    @classmethod
    def tearDownClass(cls):
        """
        Stops the Spark session after all tests have finished.
        """
        cls.spark.stop()

    def setUp(self):
        """
        Creates an isolated temporary directory for each test.
        """
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """
        Removes the temporary directory created for the test.
        """
        shutil.rmtree(
            self.temp_dir,
            ignore_errors=True,
        )

    def test_export_parquet(self):
        """
        Verifies that a DataFrame can be exported as Parquet.
        """
        path = os.path.join(
            self.temp_dir,
            "parquet_data",
        )

        message = DataExporter.export(
            self.df,
            path,
            file_format="parquet",
            mode="overwrite",
        )

        self.assertTrue(os.path.exists(path))
        self.assertIn(
            "Dataset successfully exported",
            message,
        )
        self.assertIn(
            "parquet",
            message,
        )

    def test_export_csv(self):
        """
        Verifies that a DataFrame can be exported as CSV.
        """
        path = os.path.join(
            self.temp_dir,
            "csv_data",
        )

        message = DataExporter.export(
            self.df,
            path,
            file_format="csv",
            mode="overwrite",
            header=True,
        )

        self.assertTrue(os.path.exists(path))
        self.assertIn(
            "Dataset successfully exported",
            message,
        )
        self.assertIn(
            "csv",
            message,
        )

    def test_export_orc(self):
        """
        Verifies that a DataFrame can be exported as ORC.
        """
        path = os.path.join(
            self.temp_dir,
            "orc_data",
        )

        message = DataExporter.export(
            self.df,
            path,
            file_format="orc",
            mode="overwrite",
        )

        self.assertTrue(os.path.exists(path))
        self.assertIn(
            "Dataset successfully exported",
            message,
        )
        self.assertIn(
            "orc",
            message,
        )

    def test_invalid_format(self):
        """
        Verifies that unsupported output formats are rejected.
        """
        path = os.path.join(
            self.temp_dir,
            "invalid_format",
        )

        with self.assertRaises(ValueError):
            DataExporter.export(
                self.df,
                path,
                file_format="json",
            )

    def test_invalid_mode(self):
        """
        Verifies that unsupported Spark write modes are rejected.
        """
        path = os.path.join(
            self.temp_dir,
            "invalid_mode",
        )

        with self.assertRaises(ValueError):
            DataExporter.export(
                self.df,
                path,
                file_format="csv",
                mode="invalid_mode",
            )

    def test_invalid_path(self):
        """
        Verifies that an empty output path is rejected.
        """
        with self.assertRaises(ValueError):
            DataExporter.export(
                self.df,
                "",
                file_format="parquet",
                mode="overwrite",
            )

    def test_invalid_dataframe(self):
        """
        Verifies that non-Spark DataFrame objects are rejected.
        """
        path = os.path.join(
            self.temp_dir,
            "invalid_df",
        )

        with self.assertRaises((TypeError, ValueError)):
            DataExporter.export(
                "not_a_dataframe",
                path,
                file_format="parquet",
                mode="overwrite",
            )

    def test_export_message(self):
        """
        Verifies the message returned after a successful export.
        """
        path = os.path.join(
            self.temp_dir,
            "message_test",
        )

        message = DataExporter.export(
            self.df,
            path,
            file_format="parquet",
            mode="overwrite",
        )

        expected_message = (
            "Dataset successfully exported in 'parquet' format "
            f"to: {os.path.abspath(path)}"
        )

        self.assertEqual(
            message,
            expected_message,
        )

    def test_default_export_is_parquet_overwrite(self):
        """
        Verifies that the default export format is Parquet and that
        the export succeeds without explicitly providing format or mode.
        """
        path = os.path.join(
            self.temp_dir,
            "default_parquet",
        )

        message = DataExporter.export(
            self.df,
            path,
        )

        self.assertTrue(os.path.exists(path))
        self.assertIn(
            "parquet",
            message,
        )


if __name__ == "__main__":
    unittest.main()