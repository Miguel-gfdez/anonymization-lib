import json
import os
import tempfile
import unittest

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from anonymization_lib import Generalization


class TestGeneralization(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder
            .appName("test_generalization")
            .master("local[1]")
            .getOrCreate()
        )

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def _register_temp_file_cleanup(self, path: str):
        """
        Register a temporary file for automatic cleanup after the test.

        The cleanup is executed even if the test fails before reaching its
        final assertion.
        """
        self.addCleanup(
            lambda file_path=path: (
                os.remove(file_path)
                if os.path.exists(file_path)
                else None
            )
        )

    def _create_temp_rules_file(self, content: dict):
        """
        Create a temporary JSON rules file and register it for automatic
        cleanup after the current test.
        """
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            delete=False,
            suffix=".json",
            encoding="utf-8",
        )

        json.dump(content, tmp)
        tmp.close()

        self._register_temp_file_cleanup(tmp.name)

        return tmp.name

    def test_categorical_generalization(self):
        df = self.spark.createDataFrame(
            [
                ("33001",),
                ("28001",),
                ("99999",),
            ],
            ["CP"],
        )

        rules_path = self._create_temp_rules_file({
            "column": "CP",
            "type": "categorical",
            "rules": [
                {
                    "from": "33001",
                    "to": "Asturias",
                },
                {
                    "from": "28001",
                    "to": "Madrid",
                },
            ],
        })

        result = Generalization(
            column="CP",
            rules_path=rules_path,
            default_value="Other",
        ).transform(df)

        values = [
            row["CP"]
            for row in result.collect()
        ]

        self.assertEqual(
            values,
            [
                "Asturias",
                "Madrid",
                "Other",
            ],
        )

    def test_numeric_generalization(self):
        df = self.spark.createDataFrame(
            [
                (20,),
                (35,),
                (80,),
            ],
            ["age"],
        )

        rules_path = self._create_temp_rules_file({
            "column": "age",
            "type": "numeric",
            "rules": [
                {
                    "min": 0,
                    "max": 30,
                    "value": "young",
                },
                {
                    "min": 31,
                    "max": 60,
                    "value": "adult",
                },
            ],
        })

        result = Generalization(
            column="age",
            rules_path=rules_path,
            default_value="unknown",
        ).transform(df)

        values = [
            row["age"]
            for row in result.collect()
        ]

        self.assertEqual(
            values,
            [
                "young",
                "adult",
                "unknown",
            ],
        )

    def test_temporal_year_generalization(self):
        df = (
            self.spark.createDataFrame(
                [
                    ("1998-05-10",),
                    ("2000-12-01",),
                ],
                ["birth_date"],
            )
            .withColumn(
                "birth_date",
                F.to_date("birth_date"),
            )
        )

        result = Generalization(
            column="birth_date",
            mode="year",
        ).transform(df)

        values = [
            row["birth_date"]
            for row in result.collect()
        ]

        self.assertEqual(
            values,
            [
                "1998",
                "2000",
            ],
        )

    def test_temporal_month_with_year_generalization(self):
        df = (
            self.spark.createDataFrame(
                [
                    ("1998-05-10",),
                    ("2000-12-01",),
                ],
                ["date"],
            )
            .withColumn(
                "date",
                F.to_date("date"),
            )
        )

        result = Generalization(
            column="date",
            mode="month",
            include_year=True,
        ).transform(df)

        values = [
            row["date"]
            for row in result.collect()
        ]

        self.assertEqual(
            values,
            [
                "1998-05",
                "2000-12",
            ],
        )

    def test_temporal_month_without_year_generalization(self):
        df = (
            self.spark.createDataFrame(
                [
                    ("1998-05-10",),
                    ("2000-12-01",),
                ],
                ["date"],
            )
            .withColumn(
                "date",
                F.to_date("date"),
            )
        )

        result = Generalization(
            column="date",
            mode="month",
            include_year=False,
        ).transform(df)

        values = [
            row["date"]
            for row in result.collect()
        ]

        self.assertEqual(
            values,
            [
                "05",
                "12",
            ],
        )

    def test_temporal_quarter_without_year_generalization(self):
        df = (
            self.spark.createDataFrame(
                [
                    ("2020-01-10",),
                    ("2020-05-10",),
                    ("2020-09-10",),
                    ("2020-12-10",),
                ],
                ["date"],
            )
            .withColumn(
                "date",
                F.to_date("date"),
            )
        )

        result = Generalization(
            column="date",
            mode="quarter",
            include_year=False,
        ).transform(df)

        values = [
            row["date"]
            for row in result.collect()
        ]

        self.assertEqual(
            values,
            [
                "Q1",
                "Q2",
                "Q3",
                "Q4",
            ],
        )

    def test_temporal_quarter_with_year_generalization(self):
        df = (
            self.spark.createDataFrame(
                [
                    ("2020-01-10",),
                    ("2020-05-10",),
                    ("2020-09-10",),
                    ("2020-12-10",),
                ],
                ["date"],
            )
            .withColumn(
                "date",
                F.to_date("date"),
            )
        )

        result = Generalization(
            column="date",
            mode="quarter",
            include_year=True,
        ).transform(df)

        values = [
            row["date"]
            for row in result.collect()
        ]

        self.assertEqual(
            values,
            [
                "2020-Q1",
                "2020-Q2",
                "2020-Q3",
                "2020-Q4",
            ],
        )

    def test_temporal_semester_with_year_generalization(self):
        df = (
            self.spark.createDataFrame(
                [
                    ("2024-03-10",),
                    ("2024-10-10",),
                ],
                ["date"],
            )
            .withColumn(
                "date",
                F.to_date("date"),
            )
        )

        result = Generalization(
            column="date",
            mode="semester",
            include_year=True,
        ).transform(df)

        values = [
            row["date"]
            for row in result.collect()
        ]

        self.assertEqual(
            values,
            [
                "2024-S1",
                "2024-S2",
            ],
        )

    def test_temporal_semester_without_year_generalization(self):
        df = (
            self.spark.createDataFrame(
                [
                    ("2024-03-10",),
                    ("2024-10-10",),
                ],
                ["date"],
            )
            .withColumn(
                "date",
                F.to_date("date"),
            )
        )

        result = Generalization(
            column="date",
            mode="semester",
            include_year=False,
        ).transform(df)

        values = [
            row["date"]
            for row in result.collect()
        ]

        self.assertEqual(
            values,
            [
                "S1",
                "S2",
            ],
        )

    def test_output_column(self):
        df = self.spark.createDataFrame(
            [
                (17,),
                (40,),
            ],
            ["age"],
        )

        rules_path = self._create_temp_rules_file({
            "column": "age",
            "type": "numeric",
            "rules": [
                {
                    "min": 0,
                    "max": 18,
                    "value": "-18",
                },
                {
                    "min": 19,
                    "max": 35,
                    "value": "19-35",
                },
                {
                    "min": 36,
                    "max": 50,
                    "value": "36-50",
                },
            ],
        })

        result = Generalization(
            column="age",
            rules_path=rules_path,
            output_column="age_group",
        ).transform(df)

        self.assertIn(
            "age_group",
            result.columns,
        )
        self.assertNotIn(
            "age",
            result.columns,
        )

        values = [
            row["age_group"]
            for row in result.collect()
        ]

        self.assertEqual(
            values,
            [
                "-18",
                "36-50",
            ],
        )

    def test_invalid_column_raises_error(self):
        df = self.spark.createDataFrame(
            [(1,)],
            ["age"],
        )

        generalization = Generalization(
            column="missing"
        )

        with self.assertRaises(ValueError):
            generalization.transform(df)

    def test_none_dataframe_raises_error(self):
        generalization = Generalization(
            column="age"
        )

        with self.assertRaises(ValueError):
            generalization.transform(None)

    def test_invalid_dataframe_type_raises_error(self):
        generalization = Generalization(
            column="age"
        )

        with self.assertRaises(ValueError):
            generalization.transform(
                "not_a_dataframe"
            )

    def test_invalid_column_name_raises_error(self):
        with self.assertRaises(ValueError):
            Generalization(
                column=""
            )

    def test_invalid_column_whitespace_raises_error(self):
        with self.assertRaises(ValueError):
            Generalization(
                column="   "
            )

    def test_invalid_output_column_raises_error(self):
        with self.assertRaises(ValueError):
            Generalization(
                column="age",
                output_column="",
            )

    def test_invalid_output_column_whitespace_raises_error(self):
        with self.assertRaises(ValueError):
            Generalization(
                column="age",
                output_column="   ",
            )

    def test_invalid_rules_path_type_raises_error(self):
        with self.assertRaises(ValueError):
            Generalization(
                column="age",
                rules_path=123,
            )

    def test_invalid_rules_path_empty_raises_error(self):
        with self.assertRaises(ValueError):
            Generalization(
                column="age",
                rules_path="",
            )

    def test_invalid_rules_path_whitespace_raises_error(self):
        with self.assertRaises(ValueError):
            Generalization(
                column="age",
                rules_path="   ",
            )

    def test_numeric_without_rules_path_raises_error(self):
        df = self.spark.createDataFrame(
            [
                (18,),
                (40,),
            ],
            ["age"],
        )

        generalization = Generalization(
            column="age"
        )

        with self.assertRaises(ValueError):
            generalization.transform(df)

    def test_temporal_without_mode_raises_error(self):
        df = (
            self.spark.createDataFrame(
                [("2024-01-01",)],
                ["date"],
            )
            .withColumn(
                "date",
                F.to_date("date"),
            )
        )

        generalization = Generalization(
            column="date"
        )

        with self.assertRaises(ValueError):
            generalization.transform(df)

    def test_unsupported_temporal_mode_raises_error(self):
        with self.assertRaises(ValueError):
            Generalization(
                column="date",
                mode="week",
            )

    def test_categorical_invalid_rule_is_ignored(self):
        df = self.spark.createDataFrame(
            [
                ("A",),
                ("B",),
            ],
            ["category"],
        )

        rules_path = self._create_temp_rules_file({
            "column": "category",
            "type": "categorical",
            "rules": [
                {
                    "from": "A",
                    "to": "Group A",
                },
                {
                    "from": "B",
                },
                {
                    "from": "B",
                    "to": "Group B",
                },
            ],
        })

        result = Generalization(
            column="category",
            rules_path=rules_path,
        ).transform(df)

        values = [
            row["category"]
            for row in result.collect()
        ]

        self.assertEqual(
            values,
            [
                "Group A",
                "Group B",
            ],
        )

    def test_numeric_invalid_rule_is_ignored(self):
        df = self.spark.createDataFrame(
            [
                (10,),
                (30,),
            ],
            ["age"],
        )

        rules_path = self._create_temp_rules_file({
            "column": "age",
            "type": "numeric",
            "rules": [
                {
                    "min": 0,
                    "max": 20,
                    "value": "young",
                },
                {
                    "invalid": "rule",
                },
                {
                    "min": 21,
                    "max": 40,
                    "value": "adult",
                },
            ],
        })

        result = Generalization(
            column="age",
            rules_path=rules_path,
        ).transform(df)

        values = [
            row["age"]
            for row in result.collect()
        ]

        self.assertEqual(
            values,
            [
                "young",
                "adult",
            ],
        )

    def test_numeric_invalid_numeric_values_are_ignored(self):
        df = self.spark.createDataFrame(
            [
                (10,),
                (30,),
            ],
            ["age"],
        )

        rules_path = self._create_temp_rules_file({
            "column": "age",
            "type": "numeric",
            "rules": [
                {
                    "min": "a",
                    "max": "b",
                    "value": "invalid",
                },
                {
                    "min": 0,
                    "max": 20,
                    "value": "young",
                },
                {
                    "min": 21,
                    "max": 40,
                    "value": "adult",
                },
            ],
        })

        result = Generalization(
            column="age",
            rules_path=rules_path,
        ).transform(df)

        values = [
            row["age"]
            for row in result.collect()
        ]

        self.assertEqual(
            values,
            [
                "young",
                "adult",
            ],
        )

    def test_numeric_invalid_interval_is_ignored(self):
        df = self.spark.createDataFrame(
            [
                (10,),
                (30,),
            ],
            ["age"],
        )

        rules_path = self._create_temp_rules_file({
            "column": "age",
            "type": "numeric",
            "rules": [
                {
                    "min": 50,
                    "max": 20,
                    "value": "invalid",
                },
                {
                    "min": 0,
                    "max": 20,
                    "value": "young",
                },
                {
                    "min": 21,
                    "max": 40,
                    "value": "adult",
                },
            ],
        })

        result = Generalization(
            column="age",
            rules_path=rules_path,
        ).transform(df)

        values = [
            row["age"]
            for row in result.collect()
        ]

        self.assertEqual(
            values,
            [
                "young",
                "adult",
            ],
        )

    def test_numeric_no_valid_rules_raises_error(self):
        df = self.spark.createDataFrame(
            [
                (10,),
                (30,),
            ],
            ["age"],
        )

        rules_path = self._create_temp_rules_file({
            "column": "age",
            "type": "numeric",
            "rules": [
                {
                    "invalid": "rule",
                },
                {
                    "min": "a",
                    "max": "b",
                    "value": "invalid",
                },
                {
                    "min": 50,
                    "max": 20,
                    "value": "invalid",
                },
            ],
        })

        generalization = Generalization(
            column="age",
            rules_path=rules_path,
        )

        with self.assertRaises(ValueError):
            generalization.transform(df)

    def test_json_date_type_uses_temporal_generalization(self):
        df = (
            self.spark.createDataFrame(
                [("2024-01-01",)],
                ["date"],
            )
            .withColumn(
                "date",
                F.to_date("date"),
            )
        )

        rules_path = self._create_temp_rules_file({
            "column": "date",
            "type": "date",
            "rules": [],
        })

        result = Generalization(
            column="date",
            rules_path=rules_path,
            mode="year",
        ).transform(df)

        values = [
            row["date"]
            for row in result.collect()
        ]

        self.assertEqual(
            values,
            ["2024"],
        )

    def test_unsupported_json_type_raises_error(self):
        df = self.spark.createDataFrame(
            [(10,)],
            ["age"],
        )

        rules_path = self._create_temp_rules_file({
            "column": "age",
            "type": "unsupported",
            "rules": [],
        })

        generalization = Generalization(
            column="age",
            rules_path=rules_path,
        )

        with self.assertRaises(ValueError):
            generalization.transform(df)

    def test_categorical_empty_rules_raises_error(self):
        df = self.spark.createDataFrame(
            [("A",)],
            ["category"],
        )

        rules_path = self._create_temp_rules_file({
            "column": "category",
            "type": "categorical",
            "rules": [],
        })

        generalization = Generalization(
            column="category",
            rules_path=rules_path,
        )

        with self.assertRaises(ValueError):
            generalization.transform(df)

    def test_categorical_no_valid_rules_raises_error(self):
        df = self.spark.createDataFrame(
            [("A",)],
            ["category"],
        )

        rules_path = self._create_temp_rules_file({
            "column": "category",
            "type": "categorical",
            "rules": [
                {
                    "from": "A",
                },
                {
                    "to": "Group A",
                },
            ],
        })

        generalization = Generalization(
            column="category",
            rules_path=rules_path,
        )

        with self.assertRaises(ValueError):
            generalization.transform(df)

    def test_categorical_invalid_rule_type_is_ignored(self):
        df = self.spark.createDataFrame(
            [
                ("A",),
                ("B",),
            ],
            ["category"],
        )

        rules_path = self._create_temp_rules_file({
            "column": "category",
            "type": "categorical",
            "rules": [
                {
                    "from": ["A"],
                    "to": "Invalid",
                },
                {
                    "from": "A",
                    "to": "Group A",
                },
                {
                    "from": "B",
                    "to": "Group B",
                },
            ],
        })

        result = Generalization(
            column="category",
            rules_path=rules_path,
        ).transform(df)

        values = [
            row["category"]
            for row in result.collect()
        ]

        self.assertEqual(
            values,
            [
                "Group A",
                "Group B",
            ],
        )

    def test_numeric_empty_rules_raises_error(self):
        df = self.spark.createDataFrame(
            [(10,)],
            ["age"],
        )

        rules_path = self._create_temp_rules_file({
            "column": "age",
            "type": "numeric",
            "rules": [],
        })

        generalization = Generalization(
            column="age",
            rules_path=rules_path,
        )

        with self.assertRaises(ValueError):
            generalization.transform(df)

    def test_invalid_json_raises_error(self):
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            delete=False,
            suffix=".json",
            encoding="utf-8",
        )

        tmp.write("{invalid json")
        tmp.close()

        self._register_temp_file_cleanup(
            tmp.name
        )

        df = self.spark.createDataFrame(
            [(10,)],
            ["age"],
        )

        generalization = Generalization(
            column="age",
            rules_path=tmp.name,
        )

        with self.assertRaises(ValueError):
            generalization.transform(df)

    def test_nonexistent_rules_path_raises_error(self):
        df = self.spark.createDataFrame(
            [(10,)],
            ["age"],
        )

        generalization = Generalization(
            column="age",
            rules_path="nonexistent_rules_file.json",
        )

        with self.assertRaises(Exception):
            generalization.transform(df)

    def test_numeric_generalization_preserves_null(self):
        df = self.spark.createDataFrame(
            [
                (20,),
                (None,),
            ],
            "age INT",
        )

        rules_path = self._create_temp_rules_file({
            "column": "age",
            "type": "numeric",
            "rules": [
                {
                    "min": 0,
                    "max": 30,
                    "value": "young",
                },
            ],
        })

        result = Generalization(
            column="age",
            rules_path=rules_path,
        ).transform(df)

        values = [
            row["age"]
            for row in result.collect()
        ]

        self.assertEqual(
            values[0],
            "young",
        )
        self.assertIsNone(
            values[1]
        )

    def test_categorical_generalization_preserves_null(self):
        df = self.spark.createDataFrame(
            [
                ("A",),
                (None,),
            ],
            "category STRING",
        )

        rules_path = self._create_temp_rules_file({
            "column": "category",
            "type": "categorical",
            "rules": [
                {
                    "from": "A",
                    "to": "Group A",
                },
            ],
        })

        result = Generalization(
            column="category",
            rules_path=rules_path,
        ).transform(df)

        values = [
            row["category"]
            for row in result.collect()
        ]

        self.assertEqual(
            values[0],
            "Group A",
        )
        self.assertIsNone(
            values[1]
        )


if __name__ == "__main__":
    unittest.main()