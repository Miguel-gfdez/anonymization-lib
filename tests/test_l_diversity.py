import unittest
from pyspark.sql import SparkSession, DataFrame

from anonymization_lib import LDiversity
from anonymization_lib.metrics.l_diversity import LDiversityResult


class TestLDiversity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder
            .appName("test_l_diversity")
            .master("local[1]")
            .getOrCreate()
        )

        cls.columns = ["dni", "nombre", "edad", "cp", "sexo", "ENFERMEDAD"]
        cls.data = [
            ("11111111A", "Ana", 28, "28001", "F", "Covid"),
            ("22222222B", "Julia", 28, "28001", "F", "Hipertension"),
            ("33333333C", "Maria", 35, "28002", "F", "Covid"),
            ("44444444D", "Lucia", 35, "28002", "F", "Covid"),
            ("55555555E", "Laura", 35, "28002", "F", "Migraña"),
            ("66666666F", "Nerea", 35, "28002", "F", "Anemia")
        ]

        cls.df = cls.spark.createDataFrame(cls.data, cls.columns)

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_l_diversity_summary_result_object(self):
        model = LDiversity(
            quasi_identifiers=["edad", "cp", "sexo"],
            sensitive_attribute="ENFERMEDAD",
            l_threshold=2
        )

        result = model.summary(self.df)

        self.assertIsInstance(result, LDiversityResult)
        self.assertTrue(hasattr(result, "summary_df"))
        self.assertTrue(hasattr(result, "violating_groups"))

        self.assertIsInstance(result.summary_df, DataFrame)
        self.assertIsInstance(result.violating_groups, DataFrame)

    def test_l_diversity_summary_contains_expected_columns(self):
        model = LDiversity(
            quasi_identifiers=["edad", "cp", "sexo"],
            sensitive_attribute="ENFERMEDAD",
            l_threshold=2
        )

        result = model.summary(self.df)

        expected_columns = {
            "num_equivalence_groups",
            "l_value",
            "max_l",
            "avg_l",
            "num_violating_groups",
            "l_threshold",
            "satisfies_l_diversity"
        }

        self.assertTrue(expected_columns.issubset(set(result.summary_df.columns)))

    def test_l_diversity_summary_values(self):
        model = LDiversity(
            quasi_identifiers=["edad", "cp", "sexo"],
            sensitive_attribute="ENFERMEDAD",
            l_threshold=2
        )

        result = model.summary(self.df)
        summary = result.summary_df.first()

        self.assertEqual(summary["num_equivalence_groups"], 2)
        self.assertEqual(summary["l_value"], 2)
        self.assertEqual(summary["max_l"], 3)
        self.assertEqual(summary["avg_l"], 2.5)
        self.assertEqual(summary["num_violating_groups"], 0)
        self.assertEqual(summary["l_threshold"], 2)
        self.assertTrue(summary["satisfies_l_diversity"])

    def test_l_diversity_satisfaction_false(self):
        model = LDiversity(
            quasi_identifiers=["edad", "cp", "sexo"],
            sensitive_attribute="ENFERMEDAD",
            l_threshold=4
        )

        result = model.summary(self.df)
        summary = result.summary_df.first()

        self.assertFalse(summary["satisfies_l_diversity"])
        self.assertEqual(summary["num_violating_groups"], 2)

    def test_l_diversity_violating_groups_are_below_threshold(self):
        threshold = 4

        model = LDiversity(
            quasi_identifiers=["edad", "cp", "sexo"],
            sensitive_attribute="ENFERMEDAD",
            l_threshold=threshold
        )

        result = model.summary(self.df)
        violations = result.violating_groups.collect()

        self.assertEqual(len(violations), 2)

        for row in violations:
            self.assertLess(row["l_diversity"], threshold)

    def test_l_diversity_without_quasi_identifiers(self):
        with self.assertRaises(ValueError):
            LDiversity(
                quasi_identifiers=[],
                sensitive_attribute="ENFERMEDAD",
                l_threshold=2
            )

    def test_l_diversity_without_sensitive_attribute(self):
        with self.assertRaises(ValueError):
            LDiversity(
                quasi_identifiers=["edad", "cp", "sexo"],
                sensitive_attribute="",
                l_threshold=2
            )

    def test_l_diversity_sensitive_attribute_not_string(self):
        with self.assertRaises(ValueError):
            LDiversity(
                quasi_identifiers=["edad", "cp", "sexo"],
                sensitive_attribute=["ENFERMEDAD"],
                l_threshold=2
            )

    def test_l_diversity_sensitive_attribute_in_quasi_identifiers(self):
        with self.assertRaises(ValueError):
            LDiversity(
                quasi_identifiers=["edad", "cp", "ENFERMEDAD"],
                sensitive_attribute="ENFERMEDAD",
                l_threshold=2
            )

    def test_l_diversity_with_nonexistent_column(self):
        model = LDiversity(
            quasi_identifiers=["edad", "cp", "ciudad"],
            sensitive_attribute="ENFERMEDAD",
            l_threshold=2
        )

        with self.assertRaises(ValueError):
            model.summary(self.df)

    def test_l_diversity_with_nonexistent_sensitive_attribute(self):
        model = LDiversity(
            quasi_identifiers=["edad", "cp", "sexo"],
            sensitive_attribute="DIAGNOSTICO",
            l_threshold=2
        )

        with self.assertRaises(ValueError):
            model.summary(self.df)

    def test_l_diversity_with_duplicate_quasi_identifiers(self):
        with self.assertRaises(ValueError):
            LDiversity(
                quasi_identifiers=["edad", "cp", "edad"],
                sensitive_attribute="ENFERMEDAD",
                l_threshold=2
            )

    def test_l_diversity_invalid_quasi_identifiers_type(self):
        with self.assertRaises(ValueError):
            LDiversity(
                quasi_identifiers="edad",
                sensitive_attribute="ENFERMEDAD",
                l_threshold=2
            )

    def test_l_diversity_quasi_identifiers_non_string(self):
        with self.assertRaises(ValueError):
            LDiversity(
                quasi_identifiers=["edad", 123, "sexo"],
                sensitive_attribute="ENFERMEDAD",
                l_threshold=2
            )

    def test_l_diversity_invalid_l_threshold_type(self):
        with self.assertRaises(ValueError):
            LDiversity(
                quasi_identifiers=["edad", "cp", "sexo"],
                sensitive_attribute="ENFERMEDAD",
                l_threshold="2"
            )

    def test_l_diversity_invalid_l_threshold_value(self):
        with self.assertRaises(ValueError):
            LDiversity(
                quasi_identifiers=["edad", "cp", "sexo"],
                sensitive_attribute="ENFERMEDAD",
                l_threshold=1
            )

    def test_l_diversity_with_none_dataframe(self):
        model = LDiversity(
            quasi_identifiers=["edad", "cp", "sexo"],
            sensitive_attribute="ENFERMEDAD",
            l_threshold=2
        )

        with self.assertRaises(ValueError):
            model.summary(None)

    def test_l_diversity_summary_invalid_dataframe_type(self):
        model = LDiversity(
            quasi_identifiers=["edad", "cp", "sexo"],
            sensitive_attribute="ENFERMEDAD",
            l_threshold=2
        )

        with self.assertRaises(ValueError):
            model.summary("not_a_dataframe")

    def test_l_diversity_result_getters(self):
        summary_df = self.spark.createDataFrame(
            [(2, 1, 10)],
            ["target_l", "current_l", "total_records"]
        )

        violating_groups = self.spark.createDataFrame(
            [("A", 1), ("B", 1)],
            ["group", "l_diversity"]
        )

        result = LDiversityResult(summary_df, violating_groups)

        self.assertEqual(result.get_summary_df().collect(), summary_df.collect())
        self.assertEqual(
            result.get_violating_groups().collect(),
            violating_groups.collect()
        )

if __name__ == "__main__":
    unittest.main()