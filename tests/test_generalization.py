import os
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
    
    def _create_temp_rules_file(self, content: str):
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            delete=False,
            suffix=".txt",
            encoding="utf-8"
        )
        tmp.write(content)
        tmp.close()
        return tmp.name

    def test_categorical_generalization(self):
        data = [
            ("28001",),
            ("04070",),
            ("43781",),
            ("00005",)
        ]

        df = self.spark.createDataFrame(data, ["CP"])

        rules_path = "data/cp_region.txt"

        result = (Generalization(column="CP",rules_path=rules_path,default_value="Other").transform(df))

        values = [row["CP"] for row in result.collect()]

        self.assertEqual(values,["Madrid", "Andalucia", "Cataluna", "Other"])

    def test_numeric_generalization(self):
        data = [
            (18,),
            (35,),
            (89,),
            (1200,)
        ]

        df = self.spark.createDataFrame(data, ["age"])

        rules_path = "data/Generalizacion_numerica_edades.txt"

        result = (Generalization(column="age",rules_path=rules_path,default_value="unknown").transform(df))

        values = [row["age"] for row in result.collect()]

        self.assertEqual(values,["18-25", "26-35", "+80", "unknown"])

    def test_temporal_year_generalization(self):
        data = [
            ("1998-05-10",),
            ("2000-12-01",)
        ]

        df = (self.spark.createDataFrame(data, ["birth_date"]).withColumn("birth_date", F.to_date("birth_date")))

        result = (Generalization(column="birth_date",mode="year").transform(df))

        values = [row["birth_date"] for row in result.collect()]

        self.assertEqual(values, ["1998", "2000"])

    def test_temporal_month_with_year_generalization(self):
        data = [
            ("1998-05-10",),
            ("2000-12-01",)
        ]

        df = (self.spark.createDataFrame(data, ["date"]).withColumn("date", F.to_date("date")))

        result = (Generalization(column="date",mode="month",include_year=True).transform(df))

        values = [row["date"] for row in result.collect()]

        self.assertEqual(values, ["1998-05", "2000-12"])

    def test_temporal_quarter_without_year_generalization(self):
        data = [
            ("2020-01-10",),
            ("2020-05-10",),
            ("2020-09-10",),
            ("2020-12-10",)
        ]

        df = (self.spark.createDataFrame(data, ["date"]).withColumn("date", F.to_date("date")))

        result = (Generalization(column="date",mode="quarter",include_year=False).transform(df))

        values = [row["date"] for row in result.collect()]

        self.assertEqual(values, ["Q1", "Q2", "Q3", "Q4"])

    def test_temporal_semester_with_year_generalization(self):
        data = [
            ("2024-03-10",),
            ("2024-10-10",)
        ]

        df = (
            self.spark.createDataFrame(data, ["date"])
            .withColumn("date", F.to_date("date"))
        )

        result = (Generalization(column="date",mode="semester",include_year=True).transform(df))

        values = [row["date"] for row in result.collect()]

        self.assertEqual(values, ["2024-S1", "2024-S2"])

    def test_output_column(self):
        data = [
            (17,),
            (40,)
        ]

        df = self.spark.createDataFrame(data, ["age"])

        rules_path = "data/Generalizacion_numerica_edades.txt"

        result = (Generalization(column="age",rules_path=rules_path,output_column="age_group").transform(df))

        self.assertIn("age_group", result.columns)
        self.assertNotIn("age", result.columns)

        values = [row["age_group"] for row in result.collect()]
        self.assertEqual(values, ["-18", "36-50"])

    def test_invalid_column_raises_error(self):
        df = self.spark.createDataFrame([(1,)], ["age"])

        with self.assertRaises(ValueError):
            Generalization(column="missing", rules_path="data/Generalizacion_numerica_edades.txt").transform(df)

    def test_none_dataframe_raises_error(self):
        with self.assertRaises(ValueError):
            Generalization(column="age", rules_path="data/Generalizacion_numerica_edades.txt").transform(None)

    def test_invalid_column_name_raises_error(self):
        with self.assertRaises(ValueError):
            Generalization(column="")

    def test_numeric_without_rules_path_raises_error(self):
        df = self.spark.createDataFrame([(18,), (40,)], ["age"])

        with self.assertRaises(ValueError):
            Generalization(column="age").transform(df)

    def test_temporal_without_mode_raises_error(self):
        data = [("2024-01-01",)]

        df = (self.spark.createDataFrame(data, ["date"]).withColumn("date", F.to_date("date")))

        with self.assertRaises(ValueError):
            Generalization(column="date").transform(df)

    def test_invalid_output_column_raises_error(self):
        with self.assertRaises(ValueError):
            Generalization(column="age", output_column="")

    def test_invalid_dataframe_type_raises_error(self):
        with self.assertRaises(ValueError):
            Generalization(
                column="age",
                rules_path="data/Generalizacion_numerica_edades.txt"
            ).transform("not_a_dataframe")

    def test_unsupported_temporal_mode_raises_error(self):
        df = (
            self.spark.createDataFrame([("2024-01-01",)], ["date"])
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

    def test_categorical_invalid_rule_line_is_ignored(self):
        df = self.spark.createDataFrame(
            [("A",), ("B",)],
            ["category"]
        )

        rules_path = self._create_temp_rules_file(
            "A;Group A\n"
            "invalid_line_without_separator\n"
            "B;Group B\n"
        )

        result = Generalization(
            column="category",
            rules_path=rules_path
        ).transform(df)

        values = [row["category"] for row in result.collect()]
        self.assertEqual(values, ["Group A", "Group B"])

        os.remove(rules_path)

    def test_numeric_invalid_rule_line_is_ignored(self):
        df = self.spark.createDataFrame(
            [(10,), (30,)],
            ["age"]
        )

        rules_path = self._create_temp_rules_file(
            "0;20;young\n"
            "invalid_line\n"
            "21;40;adult\n"
        )

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

        rules_path = self._create_temp_rules_file(
            "a;b;invalid\n"
            "0;20;young\n"
            "21;40;adult\n"
        )

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

        rules_path = self._create_temp_rules_file(
            "50;20;invalid\n"
            "0;20;young\n"
            "21;40;adult\n"
        )

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

        rules_path = self._create_temp_rules_file(
            "invalid_line\n"
            "a;b;invalid\n"
            "50;20;invalid\n"
        )

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


if __name__ == "__main__":
    unittest.main()