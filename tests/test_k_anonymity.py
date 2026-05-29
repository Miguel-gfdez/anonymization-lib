import unittest
from pyspark.sql import SparkSession, DataFrame

from anonymization_lib import KAnonymity
from anonymization_lib.metrics.k_anonymity import KAnonymityResult


class TestKAnonymity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder
            .appName("test_k_anonymity")
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

    def test_k_anonymity_summary_result_object(self):
        model = KAnonymity(
            quasi_identifiers=["edad", "cp", "sexo"],
            k_threshold=2
        )

        result = model.summary(self.df)

        self.assertIsInstance(result, KAnonymityResult)
        self.assertTrue(hasattr(result, "summary_df"))
        self.assertTrue(hasattr(result, "violating_groups"))

        self.assertIsInstance(result.summary_df, DataFrame)
        self.assertIsInstance(result.violating_groups, DataFrame)

    def test_k_anonymity_summary_contains_expected_columns(self):
        model = KAnonymity(
            quasi_identifiers=["edad", "cp", "sexo"],
            k_threshold=2
        )

        result = model.summary(self.df)

        expected_columns = {
            "num_equivalence_groups",
            "k_value",
            "max_group_size",
            "avg_group_size",
            "num_violating_groups",
            "k_threshold",
            "satisfies_k_anonymity"
        }

        self.assertTrue(expected_columns.issubset(set(result.summary_df.columns)))

    def test_k_anonymity_summary_values(self):
        model = KAnonymity(
            quasi_identifiers=["edad", "cp", "sexo"],
            k_threshold=2
        )

        result = model.summary(self.df)
        summary = result.summary_df.first()

        self.assertEqual(summary["num_equivalence_groups"], 2)
        self.assertEqual(summary["k_value"], 2)
        self.assertEqual(summary["max_group_size"], 4)
        self.assertEqual(summary["avg_group_size"], 3.0)
        self.assertEqual(summary["num_violating_groups"], 0)
        self.assertEqual(summary["k_threshold"], 2)
        self.assertTrue(summary["satisfies_k_anonymity"])

    def test_k_anonymity_satisfaction_false(self):
        model = KAnonymity(
            quasi_identifiers=["edad", "cp", "sexo"],
            k_threshold=3
        )

        result = model.summary(self.df)
        summary = result.summary_df.first()

        self.assertFalse(summary["satisfies_k_anonymity"])
        self.assertEqual(summary["num_violating_groups"], 1)

    def test_k_anonymity_violating_groups_are_below_threshold(self):
        threshold = 3

        model = KAnonymity(
            quasi_identifiers=["edad", "cp", "sexo"],
            k_threshold=threshold
        )

        result = model.summary(self.df)
        violations = result.violating_groups.collect()

        self.assertEqual(len(violations), 1)

        for row in violations:
            self.assertLess(row["group_size"], threshold)

    def test_k_anonymity_without_quasi_identifiers(self):
        model = KAnonymity(
            quasi_identifiers=[],
            k_threshold=2
        )

        with self.assertRaises(ValueError):
            model.summary(self.df)

    def test_k_anonymity_with_nonexistent_column(self):
        model = KAnonymity(
            quasi_identifiers=["edad", "cp", "ciudad"],
            k_threshold=2
        )

        with self.assertRaises(ValueError):
            model.summary(self.df)

    def test_k_anonymity_invalid_quasi_identifiers_type(self):
        with self.assertRaises(ValueError):
            KAnonymity(
                quasi_identifiers="edad",
                k_threshold=2
            )

    def test_k_anonymity_quasi_identifiers_non_string(self):
        with self.assertRaises(ValueError):
            KAnonymity(
                quasi_identifiers=["edad", 123, "sexo"],
                k_threshold=2
            )

    def test_k_anonymity_quasi_identifiers_duplicates(self):
        with self.assertRaises(ValueError):
            KAnonymity(
                quasi_identifiers=["edad", "cp", "edad"],
                k_threshold=2
            )

    def test_k_anonymity_invalid_k_threshold_type(self):
        with self.assertRaises(ValueError):
            KAnonymity(
                quasi_identifiers=["edad", "cp", "sexo"],
                k_threshold="2"
            )

    def test_k_anonymity_invalid_k_threshold_value(self):
        with self.assertRaises(ValueError):
            KAnonymity(
                quasi_identifiers=["edad", "cp", "sexo"],
                k_threshold=1
            )

    def test_k_anonymity_invalid_dataframe_type(self):
        model = KAnonymity(
            quasi_identifiers=["edad", "cp", "sexo"],
            k_threshold=2
        )

        with self.assertRaises(ValueError):
            model.summary("not_a_dataframe")

    def test_k_anonymity_with_none_dataframe(self):
        model = KAnonymity(
            quasi_identifiers=["edad", "cp", "sexo"],
            k_threshold=2
        )

        with self.assertRaises(ValueError):
            model.summary(None)



if __name__ == "__main__":
    unittest.main()