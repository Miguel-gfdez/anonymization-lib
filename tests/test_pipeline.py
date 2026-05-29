import unittest
import os
import tempfile
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

from anonymization_lib import Suppression, Substitution, Generalization
from anonymization_lib.techniques import TransformationPipeline


class TestTransformationPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder
            .appName("test_transformation_pipeline")
            .master("local[1]")
            .getOrCreate()
        )

        cls.columns = [
            "DNI", "NOMBRE", "APELLIDOS", "TELEFONO",
            "CIUDAD", "FECHA_NACIMIENTO", "EDAD", "CP"
        ]

        cls.data = [
            ("11111111A", "Ana", "García López", "600111222", "Madrid", "1998-03-15", 28, "28001"),
            ("22222222B", "Julia", "Pérez Díaz", "600333444", "Oviedo", "1988-07-22", 35, "04070"),
        ]

        cls.df = cls.spark.createDataFrame(cls.data, cls.columns)

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_pipeline_applies_transformations_in_order(self):
        supp = Suppression(
            columns_modes={
                "NOMBRE": "null",
                "APELLIDOS": "drop",
                "TELEFONO": "drop",
                "CIUDAD": "drop",
                "FECHA_NACIMIENTO": "drop"
            }
        )

        sub = Substitution(
            column="DNI",
            replacement_char="*",
            mode="full"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            age_rules_path = os.path.join(tmpdir, "age_rules.txt")
            cp_rules_path = os.path.join(tmpdir, "cp_rules.txt")

            with open(age_rules_path, "w", encoding="utf-8") as f:
                f.write("0;18;-18\n")
                f.write("19;25;19-25\n")
                f.write("26;35;26-35\n")
                f.write("36;50;36-50\n")

            with open(cp_rules_path, "w", encoding="utf-8") as f:
                f.write("28001;Madrid\n")
                f.write("04070;Andalucia\n")

            gen_age = Generalization(
                column="EDAD",
                rules_path=age_rules_path
            )

            gen_cp = Generalization(
                column="CP",
                rules_path=cp_rules_path,
                output_column="PROVINCIA"
            )

            pipeline = [supp, sub, gen_age, gen_cp]

            result = TransformationPipeline(self.df, pipeline)

            self.assertIsInstance(result, DataFrame)

            self.assertIn("DNI", result.columns)
            self.assertIn("NOMBRE", result.columns)
            self.assertIn("EDAD", result.columns)
            self.assertIn("PROVINCIA", result.columns)

            self.assertNotIn("APELLIDOS", result.columns)
            self.assertNotIn("TELEFONO", result.columns)
            self.assertNotIn("CIUDAD", result.columns)
            self.assertNotIn("FECHA_NACIMIENTO", result.columns)
            self.assertNotIn("CP", result.columns)

            rows = result.collect()

            self.assertEqual(rows[0]["DNI"], "*********")
            self.assertIsNone(rows[0]["NOMBRE"])
            self.assertEqual(rows[0]["EDAD"], "26-35")
            self.assertEqual(rows[0]["PROVINCIA"], "Madrid")

            self.assertEqual(rows[1]["DNI"], "*********")
            self.assertIsNone(rows[1]["NOMBRE"])
            self.assertEqual(rows[1]["EDAD"], "26-35")
            self.assertEqual(rows[1]["PROVINCIA"], "Andalucia")


    def test_pipeline_with_substitution_only(self):
        sub = Substitution(
            column="DNI",
            replacement_char="*",
            mode="full"
        )

        result = TransformationPipeline(self.df, [sub])
        values = [row["DNI"] for row in result.collect()]

        self.assertEqual(values, ["*********", "*********"])

    def test_pipeline_with_no_transformations_raises_error(self):
        with self.assertRaises(ValueError):
            TransformationPipeline(self.df, None)

    def test_pipeline_with_empty_transformations_raises_error(self):
        with self.assertRaises(ValueError):
            TransformationPipeline(self.df, [])

    def test_pipeline_with_invalid_transformations_type_raises_error(self):
        with self.assertRaises(ValueError):
            TransformationPipeline(self.df, "not_a_list")

    def test_pipeline_with_none_transformation_raises_error(self):
        with self.assertRaises(ValueError):
            TransformationPipeline(self.df, [None])

    def test_pipeline_with_object_without_transform_method_raises_error(self):
        with self.assertRaises(ValueError):
            TransformationPipeline(self.df, [object()])

    def test_pipeline_returns_original_dataframe_when_identity_transformation(self):
        class IdentityTransformation:
            def transform(self, df):
                return df

        result = TransformationPipeline(self.df, [IdentityTransformation()])

        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.columns, self.df.columns)
        self.assertEqual(result.count(), self.df.count())


if __name__ == "__main__":
    unittest.main()