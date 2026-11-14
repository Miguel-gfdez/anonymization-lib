import unittest
from unittest.mock import patch
from pyspark.sql import SparkSession

from anonymization_lib.metrics.t_closeness import (
    TCloseness,
    TClosenessResult,
)


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
            [
                "GENERO",
                "EDAD",
                "PROVINCIA",
                "ENFERMEDAD",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_summary_returns_tcloseness_result(self):
        model = TCloseness(
            quasi_identifiers=[
                "GENERO",
                "EDAD",
                "PROVINCIA",
            ],
            sensitive_attribute="ENFERMEDAD",
            t_threshold=0.5,
            distance_metric="l1",
        )

        result = model.summary(self.df)

        self.assertIsInstance(
            result,
            TClosenessResult,
        )

        self.assertTrue(
            hasattr(result, "summary_df")
        )
        self.assertTrue(
            hasattr(result, "violating_groups")
        )

        self.assertTrue(
            hasattr(result.summary_df, "select")
        )
        self.assertTrue(
            hasattr(result.violating_groups, "select")
        )

    def test_summary_contains_expected_columns(self):
        model = TCloseness(
            quasi_identifiers=[
                "GENERO",
                "EDAD",
                "PROVINCIA",
            ],
            sensitive_attribute="ENFERMEDAD",
            t_threshold=0.5,
            distance_metric="l1",
        )

        result = model.summary(self.df)

        expected_columns = {
            "num_equivalence_groups",
            "min_t",
            "max_t",
            "avg_t",
            "num_violating_groups",
            "t_threshold",
            "satisfies_t_closeness",
        }

        self.assertTrue(
            expected_columns.issubset(
                set(result.summary_df.columns)
            )
        )

    def test_summary_satisfaction_is_boolean(self):
        model = TCloseness(
            quasi_identifiers=[
                "GENERO",
                "EDAD",
                "PROVINCIA",
            ],
            sensitive_attribute="ENFERMEDAD",
            t_threshold=1.0,
            distance_metric="l1",
        )

        result = model.summary(self.df)

        value = result.summary_df.first()[
            "satisfies_t_closeness"
        ]

        self.assertIsInstance(
            value,
            bool,
        )
        self.assertTrue(value)

    def test_summary_num_violating_groups_matches_violating_groups(self):
        model = TCloseness(
            quasi_identifiers=[
                "GENERO",
                "EDAD",
                "PROVINCIA",
            ],
            sensitive_attribute="ENFERMEDAD",
            t_threshold=0.01,
            distance_metric="l1",
        )

        result = model.summary(self.df)

        summary_count = result.summary_df.first()[
            "num_violating_groups"
        ]

        violating_count = (
            result.violating_groups.count()
        )

        self.assertEqual(
            summary_count,
            violating_count,
        )

    def test_violating_groups_are_above_threshold(self):
        threshold = 0.01

        model = TCloseness(
            quasi_identifiers=[
                "GENERO",
                "EDAD",
                "PROVINCIA",
            ],
            sensitive_attribute="ENFERMEDAD",
            t_threshold=threshold,
            distance_metric="l1",
        )

        result = model.summary(self.df)

        violations = (
            result.violating_groups.collect()
        )

        self.assertGreater(
            len(violations),
            0,
        )

        for row in violations:
            self.assertGreater(
                row["t_closeness"],
                threshold,
            )

    def test_invalid_sensitive_attribute_none(self):
        with self.assertRaises(ValueError):
            TCloseness(
                quasi_identifiers=["GENERO"],
                sensitive_attribute=None,
                t_threshold=0.1,
                distance_metric="l1",
            )

    def test_invalid_sensitive_attribute_type(self):
        with self.assertRaises(ValueError):
            TCloseness(
                quasi_identifiers=["GENERO"],
                sensitive_attribute=123,
                t_threshold=0.1,
                distance_metric="l1",
            )

    def test_invalid_sensitive_attribute_empty_string(self):
        with self.assertRaises(ValueError):
            TCloseness(
                quasi_identifiers=["GENERO"],
                sensitive_attribute="",
                t_threshold=0.1,
                distance_metric="l1",
            )

    def test_invalid_sensitive_attribute_whitespace(self):
        with self.assertRaises(ValueError):
            TCloseness(
                quasi_identifiers=["GENERO"],
                sensitive_attribute="   ",
                t_threshold=0.1,
                distance_metric="l1",
            )

    def test_empty_quasi_identifiers(self):
        with self.assertRaises(ValueError):
            TCloseness(
                quasi_identifiers=[],
                sensitive_attribute="ENFERMEDAD",
                t_threshold=0.1,
                distance_metric="l1",
            )

    def test_invalid_quasi_identifiers_type(self):
        with self.assertRaises(ValueError):
            TCloseness(
                quasi_identifiers="GENERO",
                sensitive_attribute="ENFERMEDAD",
                t_threshold=0.1,
                distance_metric="l1",
            )

    def test_quasi_identifiers_non_string(self):
        with self.assertRaises(ValueError):
            TCloseness(
                quasi_identifiers=[
                    "GENERO",
                    123,
                ],
                sensitive_attribute="ENFERMEDAD",
                t_threshold=0.1,
                distance_metric="l1",
            )

    def test_quasi_identifiers_empty_string(self):
        with self.assertRaises(ValueError):
            TCloseness(
                quasi_identifiers=[
                    "GENERO",
                    "",
                ],
                sensitive_attribute="ENFERMEDAD",
                t_threshold=0.1,
                distance_metric="l1",
            )

    def test_quasi_identifiers_whitespace_string(self):
        with self.assertRaises(ValueError):
            TCloseness(
                quasi_identifiers=[
                    "GENERO",
                    "   ",
                ],
                sensitive_attribute="ENFERMEDAD",
                t_threshold=0.1,
                distance_metric="l1",
            )

    def test_sensitive_attribute_inside_quasi_identifiers(self):
        with self.assertRaises(ValueError):
            TCloseness(
                quasi_identifiers=[
                    "GENERO",
                    "ENFERMEDAD",
                ],
                sensitive_attribute="ENFERMEDAD",
                t_threshold=0.1,
                distance_metric="l1",
            )

    def test_duplicate_quasi_identifiers(self):
        with self.assertRaises(ValueError):
            TCloseness(
                quasi_identifiers=[
                    "GENERO",
                    "GENERO",
                ],
                sensitive_attribute="ENFERMEDAD",
                t_threshold=0.1,
                distance_metric="l1",
            )

    def test_invalid_t_threshold_none(self):
        with self.assertRaises(ValueError):
            TCloseness(
                quasi_identifiers=["GENERO"],
                sensitive_attribute="ENFERMEDAD",
                t_threshold=None,
                distance_metric="l1",
            )

    def test_invalid_t_threshold_type(self):
        with self.assertRaises(ValueError):
            TCloseness(
                quasi_identifiers=["sexo", "edad"],
                sensitive_attribute="diagnostico",
                t_threshold="0.5",
            )

    def test_invalid_t_threshold_bool(self):
        with self.assertRaises(ValueError):
            TCloseness(
                quasi_identifiers=["sexo", "edad"],
                sensitive_attribute="diagnostico",
                t_threshold=True,
            )


    def test_invalid_t_threshold_zero(self):
        with self.assertRaises(ValueError):
            TCloseness(
                quasi_identifiers=["sexo", "edad"],
                sensitive_attribute="diagnostico",
                t_threshold=0,
            )


    def test_invalid_t_threshold_negative(self):
        with self.assertRaises(ValueError):
            TCloseness(
                quasi_identifiers=["sexo", "edad"],
                sensitive_attribute="diagnostico",
                t_threshold=-0.1,
            )


    def test_invalid_distance_metric(self):
        with self.assertRaises(ValueError):
            TCloseness(
                quasi_identifiers=["GENERO"],
                sensitive_attribute="ENFERMEDAD",
                t_threshold=0.1,
                distance_metric="invalid",
            )

    def test_emd_metric_is_rejected(self):
        with self.assertRaises(ValueError):
            TCloseness(
                quasi_identifiers=["GENERO"],
                sensitive_attribute="ENFERMEDAD",
                t_threshold=0.1,
                distance_metric="emd",
            )

    def test_summary_with_none_dataframe(self):
        model = TCloseness(
            quasi_identifiers=["GENERO"],
            sensitive_attribute="ENFERMEDAD",
            t_threshold=0.1,
            distance_metric="l1",
        )

        with self.assertRaises(ValueError):
            model.summary(None)

    def test_summary_with_invalid_dataframe_type(self):
        model = TCloseness(
            quasi_identifiers=["GENERO"],
            sensitive_attribute="ENFERMEDAD",
            t_threshold=0.1,
            distance_metric="l1",
        )

        with self.assertRaises(ValueError):
            model.summary(
                "not_a_dataframe"
            )

    def test_summary_with_missing_quasi_identifier(self):
        model = TCloseness(
            quasi_identifiers=[
                "GENERO",
                "NO_EXISTE",
            ],
            sensitive_attribute="ENFERMEDAD",
            t_threshold=0.1,
            distance_metric="l1",
        )

        with self.assertRaises(ValueError):
            model.summary(self.df)

    def test_summary_with_missing_sensitive_attribute(self):
        model = TCloseness(
            quasi_identifiers=["GENERO"],
            sensitive_attribute="NO_EXISTE",
            t_threshold=0.1,
            distance_metric="l1",
        )

        with self.assertRaises(ValueError):
            model.summary(self.df)

    def test_t_closeness_result_getters(self):
        summary_df = self.spark.createDataFrame(
            [(0.2, 0.35, 10)],
            [
                "t_threshold",
                "max_t",
                "total_records",
            ],
        )

        violating_groups = self.spark.createDataFrame(
            [
                ("A", 0.31),
                ("B", 0.42),
            ],
            [
                "group",
                "t_closeness",
            ],
        )

        result = TClosenessResult(
            summary_df,
            violating_groups,
        )

        self.assertEqual(
            result.get_summary_df().collect(),
            summary_df.collect(),
        )

        self.assertEqual(
            result.get_violating_groups().collect(),
            violating_groups.collect(),
        )

    
    
    def test_distance_metrics_expected_values(self):
        # Probabilidades para un grupo de equivalencia:
        # P = [0.75, 0.25]
        # Q = [0.50, 0.50]
        probability_df = self.spark.createDataFrame(
            [
                ("A", 0.75, 0.50),
                ("A", 0.25, 0.50),
            ],
            ["GENERO", "group_prob", "global_prob"],
        )

        import math

        p = [0.75, 0.25]
        q = [0.50, 0.50]
        m = [(a + b) / 2 for a, b in zip(p, q)]

        expected_jsd = 0.5 * sum(
            a * math.log2(a / mid)
            for a, mid in zip(p, m)
        ) + 0.5 * sum(
            b * math.log2(b / mid)
            for b, mid in zip(q, m)
        )

        for metric, expected in [
            ("l1", 0.25),
            ("jsd", expected_jsd),
        ]:
            with self.subTest(metric=metric):
                model = TCloseness(
                    quasi_identifiers=["GENERO"],
                    sensitive_attribute="ENFERMEDAD",
                    t_threshold=0.5,
                    distance_metric=metric,
                )

                result = model._compute_distance(
                    probability_df
                ).first()

                self.assertAlmostEqual(
                    result["t_closeness"],
                    expected,
                    places=6,
                )

    def test_none_quasi_identifiers(self):
        with self.assertRaises(ValueError):
            TCloseness(
                quasi_identifiers=None,
                sensitive_attribute="ENFERMEDAD",
                t_threshold=0.5,
            )

    def test_result_show_summary(self):
        model = TCloseness(
            quasi_identifiers=["GENERO", "EDAD", "PROVINCIA"],
            sensitive_attribute="ENFERMEDAD",
            t_threshold=0.5,
        )
        result = model.summary(self.df)

        with patch.object(result.summary_df, "show") as mock_show:
            result.show_summary()

        mock_show.assert_called_once_with(truncate=False)

    def test_result_show_violating_groups(self):
        model = TCloseness(
            quasi_identifiers=["GENERO", "EDAD", "PROVINCIA"],
            sensitive_attribute="ENFERMEDAD",
            t_threshold=0.01,
        )
        result = model.summary(self.df)

        for n in [10, 5]:
            with self.subTest(n=n):
                with patch(
                    "pyspark.sql.dataframe.DataFrame.show",
                    autospec=True,
                ) as mock_show:
                    result.show_violating_groups(n=n)

                mock_show.assert_called_once()
                self.assertEqual(mock_show.call_args.args[1], n)
                self.assertFalse(
                    mock_show.call_args.kwargs["truncate"]
                )

    def test_result_show_violating_groups_sorted_and_rounded(self):
        model = TCloseness(
            quasi_identifiers=["GENERO", "EDAD", "PROVINCIA"],
            sensitive_attribute="ENFERMEDAD",
            t_threshold=0.01,
        )
        result = model.summary(self.df)

        shown = []

        def capture_show(df, n=20, truncate=True, vertical=False):
            shown.append((df, n, truncate))

        with patch(
            "pyspark.sql.dataframe.DataFrame.show",
            autospec=True,
            side_effect=capture_show,
        ):
            result.show_violating_groups(n=5, sort=True)

        self.assertEqual(len(shown), 1)

        displayed_df, n, truncate = shown[0]
        self.assertEqual(n, 5)
        self.assertFalse(truncate)

        actual = [
            row["t_closeness"]
            for row in displayed_df.select("t_closeness").collect()
        ]

        self.assertEqual(actual, sorted(actual, reverse=True))

        original = {
            row["PROVINCIA"]: row["t_closeness"]
            for row in result.violating_groups.collect()
        }

        displayed = {
            row["PROVINCIA"]: row["t_closeness"]
            for row in displayed_df.collect()
        }

        for province, value in original.items():
            self.assertAlmostEqual(
                displayed[province],
                round(value, 5),
                places=10,
            )



if __name__ == "__main__":
    unittest.main()