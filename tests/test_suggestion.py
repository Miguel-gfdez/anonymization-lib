import os
import tempfile
import unittest

from pyspark.sql import SparkSession, DataFrame

from anonymization_lib import AnonymizationAdvisor
from anonymization_lib.visualization.Suggestion import AnonymizationAdvisorResult


class TestAnonymizationAdvisor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder
            .master("local[2]")
            .appName("test_anonymization_advisor")
            .getOrCreate()
        )

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_anonymization_advisor(self):
        data = [
            (27, "28001", "H"),
            (28, "28001", "H"),
            (29, "28001", "H"),
            (30, "28001", "M"),
            (30, "28001", "M"),
            (35, "04070", "H"),
            (35, "04070", "H"),
            (35, "04070", "H"),
            (35, "04070", "H")
        ]

        df = self.spark.createDataFrame(
            data,
            ["EDAD", "CP", "SEXO"]
        )

        result = AnonymizationAdvisor(
            quasi_identifiers=["EDAD", "CP", "SEXO"],
            k=3
        ).suggest(df)

        self.assertIsInstance(result, AnonymizationAdvisorResult)
        self.assertIsInstance(result.get_summary_df(), DataFrame)
        self.assertIsInstance(result.get_suggestions_df(), DataFrame)

        summary = result.get_summary_df().collect()[0]

        self.assertEqual(summary["target_k"], 3)
        self.assertEqual(summary["current_k"], 1)
        self.assertEqual(summary["total_records"], 9)
        self.assertEqual(summary["total_equivalence_groups"], 5)
        self.assertEqual(summary["risky_groups"], 4)
        self.assertEqual(summary["risky_records"], 5)
        self.assertAlmostEqual(summary["risky_records_frequency"], 0.5556)

        suggestions_df = result.get_suggestions_df()

        expected_columns = {
            "column",
            "cardinality",
            "current_equivalence_groups",
            "groups_without_column",
            "group_reduction",
            "group_reduction_frequency",
            "suggested_action"
        }

        self.assertEqual(set(suggestions_df.columns), expected_columns)
        self.assertGreater(suggestions_df.count(), 0)

    def test_invalid_quasi_identifiers_empty(self):
        with self.assertRaises(ValueError):
            AnonymizationAdvisor(
                quasi_identifiers=[],
                k=2
            )

    def test_invalid_quasi_identifiers_type(self):
        with self.assertRaises(ValueError):
            AnonymizationAdvisor(
                quasi_identifiers="EDAD",
                k=2
            )

    def test_invalid_quasi_identifiers_values(self):
        with self.assertRaises(ValueError):
            AnonymizationAdvisor(
                quasi_identifiers=["EDAD", ""],
                k=2
            )

    def test_invalid_quasi_identifiers_duplicates(self):
        with self.assertRaises(ValueError):
            AnonymizationAdvisor(
                quasi_identifiers=["EDAD", "EDAD"],
                k=2
            )

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            AnonymizationAdvisor(
                quasi_identifiers=["EDAD"],
                k=1
            )

    def test_none_dataframe(self):
        advisor = AnonymizationAdvisor(
            quasi_identifiers=["EDAD"],
            k=2
        )

        with self.assertRaises(ValueError):
            advisor.suggest(None)

    def test_invalid_dataframe_type(self):
        advisor = AnonymizationAdvisor(
            quasi_identifiers=["EDAD"],
            k=2
        )

        with self.assertRaises(ValueError):
            advisor.suggest("not_a_dataframe")

    def test_missing_column(self):
        df = self.spark.createDataFrame(
            [(1,)],
            ["A"]
        )

        advisor = AnonymizationAdvisor(
            quasi_identifiers=["EDAD"],
            k=2
        )

        with self.assertRaises(ValueError):
            advisor.suggest(df)

    def test_anonymization_advisor_result_getters(self):
        summary_df = self.spark.createDataFrame(
            [(3, 1, 9)],
            ["target_k", "current_k", "total_records"]
        )

        suggestions_df = self.spark.createDataFrame(
            [("EDAD", 4, 5, 3, 2, 0.4, "Consider moderate generalization")],
            [
                "column",
                "cardinality",
                "current_equivalence_groups",
                "groups_without_column",
                "group_reduction",
                "group_reduction_frequency",
                "suggested_action"
            ]
        )

        result = AnonymizationAdvisorResult(
            summary_df=summary_df,
            suggestions_df=suggestions_df
        )

        self.assertEqual(
            result.get_summary_df().collect(),
            summary_df.collect()
        )

        self.assertEqual(
            result.get_suggestions_df().collect(),
            suggestions_df.collect()
        )









if __name__ == "__main__":
    unittest.main()