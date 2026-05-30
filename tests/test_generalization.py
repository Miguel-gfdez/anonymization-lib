import os
import json
import unittest
import tempfile
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

    def _create_temp_rules_file(self, content: dict):
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            delete=False,
            suffix=".json",
            encoding="utf-8"
        )
        json.dump(content, tmp)
        tmp.close()
        return tmp.name

    def test_categorical_generalization(self):
        df = self.spark.createDataFrame(
            [("33001",), ("28001",), ("99999",)],
            ["CP"]
        )

        rules_path = self._create_temp_rules_file({
            "column": "CP",
            "type": "categorical",
            "rules": [
                {"from": "33001", "to": "Asturias"},
                {"from": "28001", "to": "Madrid"}
            ]
        })

        result = Generalization(
            column="CP",
            rules_path=rules_path,
            default_value="Other"
        ).transform(df)

        values = [row["CP"] for row in result.collect()]
        self.assertEqual(values, ["Asturias", "Madrid", "Other"])

        os.remove(rules_path)

    def test_numeric_generalization(self):
        df = self.spark.createDataFrame(
            [(20,), (35,), (80,)],
            ["age"]
        )

        rules_path = self._create_temp_rules_file({
            "column": "age",
            "type": "numeric",
            "rules": [
                {"min": 0, "max": 30, "value": "young"},
                {"min": 31, "max": 60, "value": "adult"}
            ]
        })

        result = Generalization(
            column="age",
            rules_path=rules_path,
            default_value="unknown"
        ).transform(df)

        values = [row["age"] for row in result.collect()]
        self.assertEqual(values, ["young", "adult", "unknown"])

        os.remove(rules_path)

    def test_temporal_year_generalization(self):
        df = (
            self.spark.createDataFrame(
                [("1998-05-10",), ("2000-12-01",)],
                ["birth_date"]
            )
            .withColumn("birth_date", F.to_date("birth_date"))
        )

        result = Generalization(
            column="birth_date",
            mode="year"
        ).transform(df)

        values = [row["birth_date"] for row in result.collect()]
        self.assertEqual(values, ["1998", "2000"])

    def test_temporal_month_with_year_generalization(self):
        df = (
            self.spark.createDataFrame(
                [("1998-05-10",), ("2000-12-01",)],
                ["date"]
            )
            .withColumn("date", F.to_date("date"))
        )

        result = Generalization(
            column="date",
            mode="month",
            include_year=True
        ).transform(df)

        values = [row["date"] for row in result.collect()]
        self.assertEqual(values, ["1998-05", "2000-12"])

    def test_temporal_quarter_without_year_generalization(self):
        df = (
            self.spark.createDataFrame(
                [
                    ("2020-01-10",),
                    ("2020-05-10",),
                    ("2020-09-10",),
                    ("2020-12-10",)
                ],
                ["date"]
            )
            .withColumn("date", F.to_date("date"))
        )

        result = Generalization(
            column="date",
            mode="quarter",
            include_year=False
        ).transform(df)

        values = [row["date"] for row in result.collect()]
        self.assertEqual(values, ["Q1", "Q2", "Q3", "Q4"])

    def test_temporal_semester_with_year_generalization(self):
        df = (
            self.spark.createDataFrame(
                [("2024-03-10",), ("2024-10-10",)],
                ["date"]
            )
            .withColumn("date", F.to_date("date"))
        )

        result = Generalization(
            column="date",
            mode="semester",
            include_year=True
        ).transform(df)

        values = [row["date"] for row in result.collect()]
        self.assertEqual(values, ["2024-S1", "2024-S2"])

    def test_output_column(self):
        df = self.spark.createDataFrame(
            [(17,), (40,)],
            ["age"]
        )

        rules_path = self._create_temp_rules_file({
            "column": "age",
            "type": "numeric",
            "rules": [
                {"min": 0, "max": 18, "value": "-18"},
                {"min": 19, "max": 35, "value": "19-35"},
                {"min": 36, "max": 50, "value": "36-50"}
            ]
        })

        result = Generalization(
            column="age",
            rules_path=rules_path,
            output_column="age_group"
        ).transform(df)

        self.assertIn("age_group", result.columns)
        self.assertNotIn("age", result.columns)

        values = [row["age_group"] for row in result.collect()]
        self.assertEqual(values, ["-18", "36-50"])

        os.remove(rules_path)

    def test_invalid_column_raises_error(self):
        df = self.spark.createDataFrame([(1,)], ["age"])

        with self.assertRaises(ValueError):
            Generalization(column="missing").transform(df)

    def test_none_dataframe_raises_error(self):
        with self.assertRaises(ValueError):
            Generalization(column="age").transform(None)

    def test_invalid_column_name_raises_error(self):
        with self.assertRaises(ValueError):
            Generalization(column="")

    def test_numeric_without_rules_path_raises_error(self):
        df = self.spark.createDataFrame([(18,), (40,)], ["age"])

        with self.assertRaises(ValueError):
            Generalization(column="age").transform(df)

    def test_temporal_without_mode_raises_error(self):
        df = (
            self.spark.createDataFrame(
                [("2024-01-01",)],
                ["date"]
            )
            .withColumn("date", F.to_date("date"))
        )

        with self.assertRaises(ValueError):
            Generalization(column="date").transform(df)

    def test_invalid_output_column_raises_error(self):
        with self.assertRaises(ValueError):
            Generalization(column="age", output_column="")

    def test_invalid_dataframe_type_raises_error(self):
        with self.assertRaises(ValueError):
            Generalization(column="age").transform("not_a_dataframe")

    def test_unsupported_temporal_mode_raises_error(self):
        df = (
            self.spark.createDataFrame(
                [("2024-01-01",)],
                ["date"]
            )
            .withColumn("date", F.to_date("date"))
        )

        with self.assertRaises(ValueError):
            Generalization(column="date", mode="week").transform(df)

    def test_temporal_month_without_year_generalization(self):
        df = (
            self.spark.createDataFrame(
                [("1998-05-10",), ("2000-12-01",)],
                ["date"]
            )
            .withColumn("date", F.to_date("date"))
        )

        result = Generalization(
            column="date",
            mode="month",
            include_year=False
        ).transform(df)

        values = [row["date"] for row in result.collect()]
        self.assertEqual(values, ["May", "December"])

    def test_temporal_semester_without_year_generalization(self):
        df = (
            self.spark.createDataFrame(
                [("2024-03-10",), ("2024-10-10",)],
                ["date"]
            )
            .withColumn("date", F.to_date("date"))
        )

        result = Generalization(
            column="date",
            mode="semester",
            include_year=False
        ).transform(df)

        values = [row["date"] for row in result.collect()]
        self.assertEqual(values, ["S1", "S2"])

    def test_categorical_invalid_rule_is_ignored(self):
        df = self.spark.createDataFrame(
            [("A",), ("B",)],
            ["category"]
        )

        rules_path = self._create_temp_rules_file({
            "column": "category",
            "type": "categorical",
            "rules": [
                {"from": "A", "to": "Group A"},
                {"from": "B"},
                {"from": "B", "to": "Group B"}
            ]
        })

        result = Generalization(
            column="category",
            rules_path=rules_path
        ).transform(df)

        values = [row["category"] for row in result.collect()]
        self.assertEqual(values, ["Group A", "Group B"])

        os.remove(rules_path)

    def test_numeric_invalid_rule_is_ignored(self):
        df = self.spark.createDataFrame(
            [(10,), (30,)],
            ["age"]
        )

        rules_path = self._create_temp_rules_file({
            "column": "age",
            "type": "numeric",
            "rules": [
                {"min": 0, "max": 20, "value": "young"},
                {"invalid": "rule"},
                {"min": 21, "max": 40, "value": "adult"}
            ]
        })

        result = Generalization(
            column="age",
            rules_path=rules_path
        ).transform(df)

        values = [row["age"] for row in result.collect()]
        self.assertEqual(values, ["young", "adult"])

        os.remove(rules_path)

    def test_numeric_invalid_numeric_values_are_ignored(self):
        df = self.spark.createDataFrame(
            [(10,), (30,)],
            ["age"]
        )

        rules_path = self._create_temp_rules_file({
            "column": "age",
            "type": "numeric",
            "rules": [
                {"min": "a", "max": "b", "value": "invalid"},
                {"min": 0, "max": 20, "value": "young"},
                {"min": 21, "max": 40, "value": "adult"}
            ]
        })

        result = Generalization(
            column="age",
            rules_path=rules_path
        ).transform(df)

        values = [row["age"] for row in result.collect()]
        self.assertEqual(values, ["young", "adult"])

        os.remove(rules_path)

    def test_numeric_invalid_interval_is_ignored(self):
        df = self.spark.createDataFrame(
            [(10,), (30,)],
            ["age"]
        )

        rules_path = self._create_temp_rules_file({
            "column": "age",
            "type": "numeric",
            "rules": [
                {"min": 50, "max": 20, "value": "invalid"},
                {"min": 0, "max": 20, "value": "young"},
                {"min": 21, "max": 40, "value": "adult"}
            ]
        })

        result = Generalization(
            column="age",
            rules_path=rules_path
        ).transform(df)

        values = [row["age"] for row in result.collect()]
        self.assertEqual(values, ["young", "adult"])

        os.remove(rules_path)

    def test_numeric_no_valid_rules_raises_error(self):
        df = self.spark.createDataFrame(
            [(10,), (30,)],
            ["age"]
        )

        rules_path = self._create_temp_rules_file({
            "column": "age",
            "type": "numeric",
            "rules": [
                {"invalid": "rule"},
                {"min": "a", "max": "b", "value": "invalid"},
                {"min": 50, "max": 20, "value": "invalid"}
            ]
        })

        with self.assertRaises(ValueError):
            Generalization(
                column="age",
                rules_path=rules_path
            ).transform(df)

        os.remove(rules_path)

    def test_temporal_quarter_with_year_generalization(self):
        df = (
            self.spark.createDataFrame(
                [
                    ("2020-01-10",),
                    ("2020-05-10",),
                    ("2020-09-10",),
                    ("2020-12-10",)
                ],
                ["date"]
            )
            .withColumn("date", F.to_date("date"))
        )

        result = Generalization(
            column="date",
            mode="quarter",
            include_year=True
        ).transform(df)

        values = [row["date"] for row in result.collect()]
        self.assertEqual(values, ["2020-Q1", "2020-Q2", "2020-Q3", "2020-Q4"])

    def test_json_date_type_uses_temporal_generalization(self):
        df = (
            self.spark.createDataFrame(
                [("2024-01-01",)],
                ["date"]
            )
            .withColumn("date", F.to_date("date"))
        )

        rules_path = self._create_temp_rules_file({
            "column": "date",
            "type": "date",
            "rules": []
        })

        result = Generalization(
            column="date",
            rules_path=rules_path,
            mode="year"
        ).transform(df)

        values = [row["date"] for row in result.collect()]
        self.assertEqual(values, ["2024"])

        os.remove(rules_path)
    
    def test_unsupported_json_type_raises_error(self):
        df = self.spark.createDataFrame(
            [(10,)],
            ["age"]
        )

        rules_path = self._create_temp_rules_file({
            "column": "age",
            "type": "unsupported",
            "rules": []
        })

        with self.assertRaises(ValueError):
            Generalization(
                column="age",
                rules_path=rules_path
            ).transform(df)

        os.remove(rules_path)
    
    def test_categorical_empty_rules_raises_error(self):
        df = self.spark.createDataFrame(
            [("A",)],
            ["category"]
        )

        rules_path = self._create_temp_rules_file({
            "column": "category",
            "type": "categorical",
            "rules": []
        })

        with self.assertRaises(ValueError):
            Generalization(
                column="category",
                rules_path=rules_path
            ).transform(df)

        os.remove(rules_path)
    
    def test_categorical_no_valid_rules_raises_error(self):
        df = self.spark.createDataFrame(
            [("A",)],
            ["category"]
        )

        rules_path = self._create_temp_rules_file({
            "column": "category",
            "type": "categorical",
            "rules": [
                {"from": "A"},
                {"to": "Group A"}
            ]
        })

        with self.assertRaises(ValueError):
            Generalization(
                column="category",
                rules_path=rules_path
            ).transform(df)

        os.remove(rules_path)

    def test_categorical_invalid_rule_type_is_ignored(self):
        df = self.spark.createDataFrame(
            [("A",), ("B",)],
            ["category"]
        )

        rules_path = self._create_temp_rules_file({
            "column": "category",
            "type": "categorical",
            "rules": [
                {"from": ["A"], "to": "Invalid"},
                {"from": "A", "to": "Group A"},
                {"from": "B", "to": "Group B"}
            ]
        })

        result = Generalization(
            column="category",
            rules_path=rules_path
        ).transform(df)

        values = [row["category"] for row in result.collect()]
        self.assertEqual(values, ["Group A", "Group B"])

        os.remove(rules_path)

    def test_numeric_empty_rules_raises_error(self):
        df = self.spark.createDataFrame(
            [(10,)],
            ["age"]
        )

        rules_path = self._create_temp_rules_file({
            "column": "age",
            "type": "numeric",
            "rules": []
        })

        with self.assertRaises(ValueError):
            Generalization(
                column="age",
                rules_path=rules_path
            ).transform(df)

        os.remove(rules_path)







if __name__ == "__main__":
    unittest.main()