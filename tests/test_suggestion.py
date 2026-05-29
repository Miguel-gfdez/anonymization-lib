import os
import tempfile
import unittest

from pyspark.sql import SparkSession, DataFrame

from anonymization_lib import ParameterSuggestion
from anonymization_lib.visualization.Suggestion import ParameterSuggestionResult


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

    def test_invalid_column_empty(self):
        with self.assertRaises(ValueError):
            ParameterSuggestion("")

    def test_invalid_column_type(self):
        with self.assertRaises(ValueError):
            ParameterSuggestion(123)

    def test_invalid_rare_threshold_type(self):
        with self.assertRaises(ValueError):
            ParameterSuggestion(
                column="age",
                rare_threshold="0.05"
            )

    def test_invalid_rare_threshold_value_zero(self):
        with self.assertRaises(ValueError):
            ParameterSuggestion(
                column="age",
                rare_threshold=0.0
            )

    def test_invalid_rare_threshold_value_one(self):
        with self.assertRaises(ValueError):
            ParameterSuggestion(
                column="age",
                rare_threshold=1.0
            )

    def test_invalid_rare_threshold_value_greater_than_one(self):
        with self.assertRaises(ValueError):
            ParameterSuggestion(
                column="age",
                rare_threshold=2.0
            )

    def test_invalid_max_categories_type(self):
        with self.assertRaises(ValueError):
            ParameterSuggestion(
                column="category",
                max_categories="20"
            )

    def test_invalid_max_categories_value(self):
        with self.assertRaises(ValueError):
            ParameterSuggestion(
                column="category",
                max_categories=1
            )

    def test_invalid_num_bins_type(self):
        with self.assertRaises(ValueError):
            ParameterSuggestion(
                column="age",
                num_bins="3"
            )

    def test_invalid_num_bins_value(self):
        with self.assertRaises(ValueError):
            ParameterSuggestion(
                column="age",
                num_bins=1
            )

    def test_suggest_invalid_dataframe_type(self):
        suggestion = ParameterSuggestion("age")

        with self.assertRaises(ValueError):
            suggestion.suggest("not_a_dataframe")

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

    def test_numeric_column_returns_result_object(self):
        df = self.spark.createDataFrame(
            [
                (20,),
                (25,),
                (30,),
                (35,),
                (40,),
                (45,),
            ],
            ["age"]
        )

        result = ParameterSuggestion(
            column="age",
            num_bins=3
        ).suggest(df)

        self.assertIsInstance(result, ParameterSuggestionResult)
        self.assertIsInstance(result.summary_df, DataFrame)
        self.assertIsInstance(result.suggestions_df, DataFrame)
        self.assertIsNone(result.preserved_categories_df)
        self.assertEqual(result.suggestion_type, "numeric")

    def test_numeric_summary_contains_expected_columns(self):
        df = self.spark.createDataFrame(
            [
                (20,),
                (25,),
                (30,),
                (35,),
                (40,),
                (45,),
            ],
            ["age"]
        )

        result = ParameterSuggestion(
            column="age",
            num_bins=3
        ).suggest(df)

        expected_columns = {
            "column",
            "inferred_type",
            "spark_type",
            "total_records",
            "non_null_records",
            "min_value",
            "max_value",
            "mean_value",
            "stddev_value",
            "suggested_technique",
            "binning_strategy",
        }

        self.assertEqual(set(result.summary_df.columns), expected_columns)

    def test_numeric_summary_values(self):
        df = self.spark.createDataFrame(
            [
                (20,),
                (25,),
                (30,),
                (35,),
                (40,),
                (45,),
            ],
            ["age"]
        )

        result = ParameterSuggestion(
            column="age",
            num_bins=3
        ).suggest(df)

        summary = result.summary_df.first()

        self.assertEqual(summary["column"], "age")
        self.assertEqual(summary["inferred_type"], "numeric")
        self.assertEqual(summary["spark_type"], "bigint")
        self.assertEqual(summary["total_records"], 6)
        self.assertEqual(summary["non_null_records"], 6)
        self.assertEqual(summary["min_value"], 20)
        self.assertEqual(summary["max_value"], 45)
        self.assertEqual(summary["suggested_technique"], "generalization")
        self.assertEqual(summary["binning_strategy"], "equal_frequency")

    def test_numeric_suggestions_contains_expected_columns(self):
        df = self.spark.createDataFrame(
            [
                (20,),
                (25,),
                (30,),
                (35,),
                (40,),
                (45,),
            ],
            ["age"]
        )

        result = ParameterSuggestion(
            column="age",
            num_bins=3
        ).suggest(df)

        expected_columns = {
            "start_value",
            "end_value",
            "count",
            "frequency",
        }

        self.assertEqual(set(result.suggestions_df.columns), expected_columns)
        self.assertGreater(result.suggestions_df.count(), 0)

    def test_numeric_column_with_constant_values_returns_empty_suggestions(self):
        df = self.spark.createDataFrame(
            [
                (20,),
                (20,),
                (20,),
            ],
            ["age"]
        )

        result = ParameterSuggestion(
            column="age",
            num_bins=3
        ).suggest(df)

        self.assertEqual(result.suggestion_type, "numeric")
        self.assertEqual(result.suggestions_df.count(), 0)
        self.assertEqual(result.summary_df.first()["min_value"], 20)
        self.assertEqual(result.summary_df.first()["max_value"], 20)

    def test_numeric_column_with_all_null_values_returns_empty_suggestions(self):
        df = self.spark.createDataFrame(
            [
                (None,),
                (None,),
            ],
            "age INT"
        )

        result = ParameterSuggestion(
            column="age",
            num_bins=3
        ).suggest(df)

        self.assertEqual(result.suggestion_type, "numeric")
        self.assertEqual(result.summary_df.first()["non_null_records"], 0)
        self.assertEqual(result.suggestions_df.count(), 0)

    def test_categorical_column_returns_result_object(self):
        df = self.spark.createDataFrame(
            [
                ("A",),
                ("A",),
                ("A",),
                ("B",),
                ("C",),
            ],
            ["category"]
        )

        result = ParameterSuggestion(
            column="category",
            rare_threshold=0.30
        ).suggest(df)

        self.assertIsInstance(result, ParameterSuggestionResult)
        self.assertIsInstance(result.summary_df, DataFrame)
        self.assertIsInstance(result.suggestions_df, DataFrame)
        self.assertIsInstance(result.preserved_categories_df, DataFrame)
        self.assertEqual(result.suggestion_type, "categorical")

    def test_categorical_summary_contains_expected_columns(self):
        df = self.spark.createDataFrame(
            [
                ("A",),
                ("A",),
                ("A",),
                ("B",),
                ("C",),
            ],
            ["category"]
        )

        result = ParameterSuggestion(
            column="category",
            rare_threshold=0.30
        ).suggest(df)

        expected_columns = {
            "column",
            "inferred_type",
            "spark_type",
            "total_records",
            "total_categories",
            "rare_categories",
            "common_categories",
            "rare_threshold",
            "suggested_technique",
            "default_generalization",
        }

        self.assertEqual(set(result.summary_df.columns), expected_columns)

    def test_categorical_summary_values(self):
        df = self.spark.createDataFrame(
            [
                ("A",),
                ("A",),
                ("A",),
                ("B",),
                ("C",),
            ],
            ["category"]
        )

        result = ParameterSuggestion(
            column="category",
            rare_threshold=0.30
        ).suggest(df)

        summary = result.summary_df.first()

        self.assertEqual(summary["column"], "category")
        self.assertEqual(summary["inferred_type"], "categorical")
        self.assertEqual(summary["spark_type"], "string")
        self.assertEqual(summary["total_records"], 5)
        self.assertEqual(summary["total_categories"], 3)
        self.assertEqual(summary["rare_categories"], 2)
        self.assertEqual(summary["common_categories"], 1)
        self.assertEqual(summary["rare_threshold"], 0.30)
        self.assertEqual(summary["suggested_technique"], "generalization")
        self.assertEqual(summary["default_generalization"], "default_other")

    def test_categorical_suggestions_values(self):
        df = self.spark.createDataFrame(
            [
                ("A",),
                ("A",),
                ("A",),
                ("B",),
                ("C",),
            ],
            ["category"]
        )

        result = ParameterSuggestion(
            column="category",
            rare_threshold=0.30
        ).suggest(df)

        suggestion = result.suggestions_df.first()

        self.assertEqual(suggestion["rare_categories_count"], 2)
        self.assertEqual(suggestion["affected_rows"], 2)
        self.assertAlmostEqual(suggestion["affected_frequency"], 0.4)

    def test_categorical_preserved_categories_contains_common_values(self):
        df = self.spark.createDataFrame(
            [
                ("A",),
                ("A",),
                ("A",),
                ("B",),
                ("C",),
            ],
            ["category"]
        )

        result = ParameterSuggestion(
            column="category",
            rare_threshold=0.30
        ).suggest(df)

        preserved = [
            row["category_value"]
            for row in result.preserved_categories_df.collect()
        ]

        self.assertEqual(preserved, ["A"])


    def test_save_invalid_path(self):
        df = self.spark.createDataFrame(
            [
                (20,),
                (25,),
                (30,),
            ],
            ["age"]
        )

        result = ParameterSuggestion(
            column="age",
            num_bins=2
        ).suggest(df)

        with self.assertRaises(ValueError):
            result.save("")

    def test_save_numeric_creates_txt_file(self):
        df = self.spark.createDataFrame(
            [
                (20,),
                (25,),
                (30,),
                (35,),
            ],
            ["age"]
        )

        result = ParameterSuggestion(
            column="age",
            num_bins=2
        ).suggest(df)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "numeric_rules")

            result.save(path)

            expected_path = path + ".txt"

            self.assertTrue(os.path.exists(expected_path))

            with open(expected_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn(";", content)
            self.assertGreater(len(content.strip()), 0)
    
    def test_save_categorical_creates_txt_file(self):
        df = self.spark.createDataFrame(
            [
                ("A",),
                ("A",),
                ("A",),
                ("B",),
                ("C",),
            ],
            ["category"]
        )

        result = ParameterSuggestion(
            column="category",
            rare_threshold=0.30
        ).suggest(df)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "categorical_rules")

            result.save(path)

            expected_path = path + ".txt"

            self.assertTrue(os.path.exists(expected_path))

            with open(expected_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn("A;A", content)
            self.assertNotIn("B;B", content)
            self.assertNotIn("C;C", content)


if __name__ == "__main__":
    unittest.main()