import unittest
from pyspark.sql import SparkSession

from anonymization_lib import ParameterSuggestion


class TestParameterSuggestion(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder
            .master("local[2]")
            .appName("test_parameter_suggestion")
            .getOrCreate()
        )

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_invalid_column(self):
        with self.assertRaises(ValueError):
            ParameterSuggestion("")

    def test_invalid_threshold(self):
        with self.assertRaises(ValueError):
            ParameterSuggestion(
                column="age",
                rare_threshold=2.0
            )

    def test_column_not_exists(self):

        df = self.spark.createDataFrame(
            [(1, "A"), (2, "B")],
            ["id", "name"]
        )

        suggestion = ParameterSuggestion("age")

        with self.assertRaises(ValueError):
            suggestion.suggest(df)

    def test_empty_dataframe(self):

        df = self.spark.createDataFrame(
            [],
            "age INT"
        )

        suggestion = ParameterSuggestion("age")

        with self.assertRaises(ValueError):
            suggestion.suggest(df)

    def test_numeric_column(self):

        df = self.spark.createDataFrame(
            [
                (20,),
                (25,),
                (30,),
                (35,),
                (40,),
                (45,)
            ],
            ["age"]
        )

        result = ParameterSuggestion(
            column="age",
            num_bins=3
        ).suggest(df)

        self.assertEqual(
            result.suggestion_type,
            "numeric"
        )

        self.assertGreater(
            result.suggestions_df.count(),
            0
        )

        self.assertEqual(
            result.summary_df.count(),
            1
        )

    def test_categorical_column(self):

        df = self.spark.createDataFrame(
            [
                ("A",),
                ("A",),
                ("A",),
                ("B",),
                ("C",)
            ],
            ["category"]
        )

        result = ParameterSuggestion(
            column="category",
            rare_threshold=0.30
        ).suggest(df)

        self.assertEqual(
            result.suggestion_type,
            "categorical"
        )

        self.assertEqual(
            result.summary_df.count(),
            1
        )

        self.assertIsNotNone(
            result.preserved_categories_df
        )


if __name__ == "__main__":
    unittest.main()