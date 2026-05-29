import unittest
import plotly.graph_objects as go

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

from anonymization_lib import Visualization


class TestVisualization(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder
            .appName("test_visualization")
            .master("local[1]")
            .getOrCreate()
        )

        # Numeric dataset
        cls.numeric_columns = ["name", "age"]
        cls.numeric_data = [
            ("Ana", 28),
            ("Julia", 31),
            ("Marcos", 35),
            ("Lucia", 40)
        ]
        cls.df_numeric = cls.spark.createDataFrame(cls.numeric_data, cls.numeric_columns)

        # Categorical dataset
        cls.categorical_columns = ["name", "gender"]
        cls.categorical_data = [
            ("Ana", "F"),
            ("Julia", "F"),
            ("Marcos", "M"),
            ("Lucia", "F")
        ]
        cls.df_categorical = cls.spark.createDataFrame(cls.categorical_data, cls.categorical_columns)

        # Date dataset
        cls.date_columns = ["name", "birth_date"]
        cls.date_data = [
            ("Ana", "1998-03-15"),
            ("Julia", "1998-07-22"),
            ("Marcos", "1988-11-05"),
            ("Lucia", "1988-01-30")
        ]
        cls.df_date = cls.spark.createDataFrame(cls.date_data, cls.date_columns)
        cls.df_date = cls.df_date.withColumn("birth_date", F.to_date("birth_date", "yyyy-MM-dd"))

        # Boolean dataset
        cls.boolean_columns = ["name", "bool"]
        cls.boolean_data = [
            ("Ana", True),
            ("Julia", False),
            ("Marcos", True),
            ("Lucia", True)
        ]
        cls.df_boolean = cls.spark.createDataFrame(cls.boolean_data, cls.boolean_columns)

        # All-null dataset
        schema = StructType([
            StructField("name", StringType(), True),
            StructField("age", IntegerType(), True)
        ])
        cls.null_data = [
            ("Ana", None),
            ("Julia", None)
        ]
        cls.df_all_null = cls.spark.createDataFrame(cls.null_data, schema)

        # Small dataset for column-not-found / generic checks
        cls.small_data = [
            ("Ana", 28),
            ("Julia", 31)
        ]
        cls.small_columns = ["name", "age"]
        cls.df_small = cls.spark.createDataFrame(cls.small_data, cls.small_columns)

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_visualization_numeric_column(self):
        model = Visualization(column="age")
        fig = model.transform(self.df_numeric)

        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 3)
        self.assertEqual(fig.data[0].type, "bar")
        self.assertEqual(fig.data[1].type, "histogram")
        self.assertEqual(fig.data[2].type, "box")

    def test_visualization_categorical_column(self):
        model = Visualization(column="gender")
        fig = model.transform(self.df_categorical)

        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 2)
        self.assertEqual(fig.data[0].type, "bar")
        self.assertEqual(fig.data[1].type, "treemap")

    def test_visualization_date_column(self):
        model = Visualization(column="birth_date")
        fig = model.transform(self.df_date)

        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 3)
        self.assertEqual(fig.data[0].type, "bar")
        self.assertEqual(fig.data[1].type, "bar")
        self.assertEqual(fig.data[2].type, "sunburst")

    def test_visualization_boolean_column(self):
        model = Visualization(column="bool")
        fig = model.transform(self.df_boolean)

        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 2)
        self.assertEqual(fig.data[0].type, "bar")
        self.assertEqual(fig.data[1].type, "treemap")

    def test_visualization_column_not_found(self):
        model = Visualization(column="salary")

        with self.assertRaises(ValueError):
            model.transform(self.df_small)

    def test_visualization_all_null_column(self):
        model = Visualization(column="age")

        with self.assertRaises(ValueError):
            model.transform(self.df_all_null)

    def test_visualization_none_dataframe(self):
        model = Visualization(column="age")

        with self.assertRaises(ValueError):
            model.transform(None)


    def test_visualization_invalid_top_n_categories(self):
        with self.assertRaises(ValueError):
            Visualization(column="gender", top_n_categories=1)

    def test_visualization_invalid_top_n_categories_type(self):
        with self.assertRaises(ValueError):
            Visualization(column="sexo", top_n_categories="10")

    def test_visualization_invalid_histogram_bins(self):
        with self.assertRaises(ValueError):
            Visualization(column="age", histogram_bins=0)

    def test_visualization_invalid_histogram_bins_type(self):
        with self.assertRaises(ValueError):
            Visualization(column="edad", histogram_bins="20")

    def test_visualization_invalid_column_none(self):
        with self.assertRaises(ValueError):
            Visualization(column=None)

    def test_visualization_invalid_column_empty(self):
        with self.assertRaises(ValueError):
            Visualization(column="")

    def test_visualization_numeric_title_contains_histogram_bins(self):
        model = Visualization(column="age", histogram_bins=15)
        fig = model.transform(self.df_numeric)

        self.assertIn("histogram_bins=15", fig.layout.title.text)

    def test_visualization_categorical_title_contains_top_n_categories(self):
        model = Visualization(column="gender", top_n_categories=5)
        fig = model.transform(self.df_categorical)

        self.assertIn("top_n_categories=5", fig.layout.title.text)

    def test_visualization_invalid_dataframe_type(self):
        model = Visualization(column="edad")

        with self.assertRaises(ValueError):
            model.transform("not_a_dataframe")











if __name__ == "__main__":
    unittest.main()