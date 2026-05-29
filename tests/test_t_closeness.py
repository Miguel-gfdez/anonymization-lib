import unittest
from pyspark.sql import SparkSession, DataFrame
from anonymization_lib.metrics.t_closeness import TCloseness, TClosenessResult


class TestTCloseness(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder
            .appName("test_t_closeness")
            .master("local[1]")
            .getOrCreate()
        )

        cls.df = cls.spark.createDataFrame(
            [
                ("F", "20-30", "Asturias", "Gripe"),
                ("F", "20-30", "Asturias", "Gripe"),
                ("F", "20-30", "Asturias", "Covid"),
                ("M", "30-40", "Madrid", "Covid"),
                ("M", "30-40", "Madrid", "Covid"),
                ("M", "30-40", "Madrid", "Gripe"),
                ("F", "40-50", "Galicia", "Cancer"),
                ("F", "40-50", "Galicia", "Cancer"),
            ],
            ["GENERO", "EDAD", "PROVINCIA", "ENFERMEDAD"]
        )

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_summary_returns_tcloseness_result(self):
        model = TCloseness(
            quasi_identifiers=["GENERO", "EDAD", "PROVINCIA"],
            sensitive_attribute="ENFERMEDAD",
            t_threshold=0.5,
            distance_metric="l1"
        )

        result = model.summary(self.df)

        self.assertIsInstance(result, TClosenessResult)
        self.assertIsInstance(result.summary_df, DataFrame)
        self.assertIsInstance(result.violating_groups, DataFrame)

    def test_summary_contains_expected_columns(self):
        model = TCloseness(
            quasi_identifiers=["GENERO", "EDAD", "PROVINCIA"],
            sensitive_attribute="ENFERMEDAD",
            t_threshold=0.5,
            distance_metric="l1"
        )

        result = model.summary(self.df)

        expected_columns = {
            "num_equivalence_groups",
            "min_t",
            "max_t",
            "avg_t",
            "num_violating_groups",
            "t_threshold",
            "satisfies_t_closeness"
        }

        self.assertTrue(expected_columns.issubset(set(result.summary_df.columns)))

    def test_summary_satisfaction_is_boolean(self):
        model = TCloseness(
            quasi_identifiers=["GENERO", "EDAD", "PROVINCIA"],
            sensitive_attribute="ENFERMEDAD",
            t_threshold=1.0,
            distance_metric="l1"
        )

        result = model.summary(self.df)

        value = result.summary_df.collect()[0]["satisfies_t_closeness"]

        self.assertIsInstance(value, bool)
        self.assertTrue(value)

    def test_summary_num_violating_groups_matches_violating_groups(self):
        model = TCloseness(
            quasi_identifiers=["GENERO", "EDAD", "PROVINCIA"],
            sensitive_attribute="ENFERMEDAD",
            t_threshold=0.01,
            distance_metric="l1"
        )

        result = model.summary(self.df)

        summary_count = result.summary_df.collect()[0]["num_violating_groups"]
        violating_count = result.violating_groups.count()

        self.assertEqual(summary_count, violating_count)

    def test_violating_groups_are_above_threshold(self):
        threshold = 0.01

        model = TCloseness(
            quasi_identifiers=["GENERO", "EDAD", "PROVINCIA"],
            sensitive_attribute="ENFERMEDAD",
            t_threshold=threshold,
            distance_metric="l1"
        )

        result = model.summary(self.df)

        violations = result.violating_groups.collect()

        self.assertGreater(len(violations), 0)

        for row in violations:
            self.assertGreater(row["t_closeness"], threshold)

    def test_emd_metric_runs(self):
        model = TCloseness(
            quasi_identifiers=["GENERO", "EDAD", "PROVINCIA"],
            sensitive_attribute="ENFERMEDAD",
            t_threshold=0.5,
            distance_metric="emd"
        )

        result = model.summary(self.df)

        self.assertIsInstance(result.summary_df, DataFrame)

    def test_jsd_metric_runs(self):
        model = TCloseness(
            quasi_identifiers=["GENERO", "EDAD", "PROVINCIA"],
            sensitive_attribute="ENFERMEDAD",
            t_threshold=0.5,
            distance_metric="jsd"
        )

        result = model.summary(self.df)

        self.assertIsInstance(result.summary_df, DataFrame)

    def test_show_methods_do_not_fail(self):
        model = TCloseness(
            quasi_identifiers=["GENERO", "EDAD", "PROVINCIA"],
            sensitive_attribute="ENFERMEDAD",
            t_threshold=0.5,
            distance_metric="l1"
        )

        result = model.summary(self.df)

        result.show_summary()
        result.show_violating_groups()

    def test_invalid_sensitive_attribute_none(self):
        with self.assertRaises(ValueError):
            TCloseness(
                quasi_identifiers=["GENERO"],
                sensitive_attribute=None,
                t_threshold=0.1,
                distance_metric="l1"
            )

    def test_invalid_sensitive_attribute_type(self):
        with self.assertRaises(ValueError):
            TCloseness(
                quasi_identifiers=["GENERO"],
                sensitive_attribute=123,
                t_threshold=0.1,
                distance_metric="l1"
            )

    def test_empty_quasi_identifiers(self):
        with self.assertRaises(ValueError):
            TCloseness(
                quasi_identifiers=[],
                sensitive_attribute="ENFERMEDAD",
                t_threshold=0.1,
                distance_metric="l1"
            )

    def test_invalid_quasi_identifiers_type(self):
        with self.assertRaises(ValueError):
            TCloseness(
                quasi_identifiers="GENERO",
                sensitive_attribute="ENFERMEDAD",
                t_threshold=0.1,
                distance_metric="l1"
            )

    def test_sensitive_attribute_inside_quasi_identifiers(self):
        with self.assertRaises(ValueError):
            TCloseness(
                quasi_identifiers=["GENERO", "ENFERMEDAD"],
                sensitive_attribute="ENFERMEDAD",
                t_threshold=0.1,
                distance_metric="l1"
            )

    def test_duplicate_quasi_identifiers(self):
        with self.assertRaises(ValueError):
            TCloseness(
                quasi_identifiers=["GENERO", "GENERO"],
                sensitive_attribute="ENFERMEDAD",
                t_threshold=0.1,
                distance_metric="l1"
            )

    def test_invalid_t_threshold_none(self):
        with self.assertRaises(ValueError):
            TCloseness(
                quasi_identifiers=["GENERO"],
                sensitive_attribute="ENFERMEDAD",
                t_threshold=None,
                distance_metric="l1"
            )

    def test_invalid_t_threshold_type(self):
        with self.assertRaises(ValueError):
            TCloseness(
                quasi_identifiers=["GENERO"],
                sensitive_attribute="ENFERMEDAD",
                t_threshold="0.1",
                distance_metric="l1"
            )

    def test_invalid_t_threshold_zero(self):
        with self.assertRaises(ValueError):
            TCloseness(
                quasi_identifiers=["GENERO"],
                sensitive_attribute="ENFERMEDAD",
                t_threshold=0,
                distance_metric="l1"
            )

    def test_invalid_distance_metric(self):
        with self.assertRaises(ValueError):
            TCloseness(
                quasi_identifiers=["GENERO"],
                sensitive_attribute="ENFERMEDAD",
                t_threshold=0.1,
                distance_metric="invalid"
            )

    def test_summary_with_none_dataframe(self):
        model = TCloseness(
            quasi_identifiers=["GENERO"],
            sensitive_attribute="ENFERMEDAD",
            t_threshold=0.1,
            distance_metric="l1"
        )

        with self.assertRaises(ValueError):
            model.summary(None)

    def test_summary_with_invalid_dataframe_type(self):
        model = TCloseness(
            quasi_identifiers=["GENERO"],
            sensitive_attribute="ENFERMEDAD",
            t_threshold=0.1,
            distance_metric="l1"
        )

        with self.assertRaises(ValueError):
            model.summary("not_a_dataframe")

    def test_summary_with_missing_column(self):
        model = TCloseness(
            quasi_identifiers=["GENERO", "NO_EXISTE"],
            sensitive_attribute="ENFERMEDAD",
            t_threshold=0.1,
            distance_metric="l1"
        )

        with self.assertRaises(ValueError):
            model.summary(self.df)


if __name__ == "__main__":
    unittest.main()