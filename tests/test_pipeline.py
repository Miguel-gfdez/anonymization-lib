import json
import os
import tempfile
import unittest

from pyspark.sql import SparkSession

from anonymization_lib import (
    Generalization,
    Substitution,
    Suppression,
)
from anonymization_lib.techniques import transformation_pipeline


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
            "DNI",
            "NOMBRE",
            "APELLIDOS",
            "TELEFONO",
            "CIUDAD",
            "FECHA_NACIMIENTO",
            "EDAD",
            "CP",
        ]

        cls.data = [
            (
                "11111111A",
                "Ana",
                "García López",
                "600111222",
                "Madrid",
                "1998-03-15",
                28,
                "28001",
            ),
            (
                "22222222B",
                "Julia",
                "Pérez Díaz",
                "600333444",
                "Almería",
                "1988-07-22",
                35,
                "04070",
            ),
        ]

        cls.df = cls.spark.createDataFrame(
            cls.data,
            cls.columns,
        )

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
                "FECHA_NACIMIENTO": "drop",
            }
        )

        sub = Substitution(
            column="DNI",
            replacement_char="*",
            mode="full",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            age_rules_path = os.path.join(
                tmpdir,
                "age_rules.json",
            )

            cp_rules_path = os.path.join(
                tmpdir,
                "cp_rules.json",
            )

            with open(
                age_rules_path,
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    {
                        "column": "EDAD",
                        "type": "numeric",
                        "rules": [
                            {
                                "min": 0,
                                "max": 18,
                                "value": "-18",
                            },
                            {
                                "min": 19,
                                "max": 25,
                                "value": "19-25",
                            },
                            {
                                "min": 26,
                                "max": 35,
                                "value": "26-35",
                            },
                            {
                                "min": 36,
                                "max": 50,
                                "value": "36-50",
                            },
                        ],
                    },
                    file,
                )

            with open(
                cp_rules_path,
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    {
                        "column": "CP",
                        "type": "categorical",
                        "rules": [
                            {
                                "from": "28001",
                                "to": "Madrid",
                            },
                            {
                                "from": "04070",
                                "to": "Andalucia",
                            },
                        ],
                    },
                    file,
                )

            gen_age = Generalization(
                column="EDAD",
                rules_path=age_rules_path,
            )

            gen_cp = Generalization(
                column="CP",
                rules_path=cp_rules_path,
                output_column="PROVINCIA",
            )

            pipeline = [
                supp,
                sub,
                gen_age,
                gen_cp,
            ]

            result = transformation_pipeline(
                self.df,
                pipeline,
            )

            self.assertTrue(
                hasattr(result, "select")
            )

            self.assertIn(
                "DNI",
                result.columns,
            )
            self.assertIn(
                "NOMBRE",
                result.columns,
            )
            self.assertIn(
                "EDAD",
                result.columns,
            )
            self.assertIn(
                "PROVINCIA",
                result.columns,
            )

            self.assertNotIn(
                "APELLIDOS",
                result.columns,
            )
            self.assertNotIn(
                "TELEFONO",
                result.columns,
            )
            self.assertNotIn(
                "CIUDAD",
                result.columns,
            )
            self.assertNotIn(
                "FECHA_NACIMIENTO",
                result.columns,
            )
            self.assertNotIn(
                "CP",
                result.columns,
            )

            rows = result.collect()

            self.assertEqual(
                rows[0]["DNI"],
                "*********",
            )
            self.assertIsNone(
                rows[0]["NOMBRE"]
            )
            self.assertEqual(
                rows[0]["EDAD"],
                "26-35",
            )
            self.assertEqual(
                rows[0]["PROVINCIA"],
                "Madrid",
            )

            self.assertEqual(
                rows[1]["DNI"],
                "*********",
            )
            self.assertIsNone(
                rows[1]["NOMBRE"]
            )
            self.assertEqual(
                rows[1]["EDAD"],
                "26-35",
            )
            self.assertEqual(
                rows[1]["PROVINCIA"],
                "Andalucia",
            )

    def test_pipeline_with_substitution_only(self):
        sub = Substitution(
            column="DNI",
            replacement_char="*",
            mode="full",
        )

        result = transformation_pipeline(
            self.df,
            [sub],
        )

        values = [
            row["DNI"]
            for row in result.collect()
        ]

        self.assertEqual(
            values,
            [
                "*********",
                "*********",
            ],
        )

    def test_pipeline_with_none_dataframe_raises_error(self):
        sub = Substitution(
            column="DNI",
            replacement_char="*",
            mode="full",
        )

        with self.assertRaises(ValueError):
            transformation_pipeline(
                None,
                [sub],
            )

    def test_pipeline_with_invalid_dataframe_raises_error(self):
        sub = Substitution(
            column="DNI",
            replacement_char="*",
            mode="full",
        )

        with self.assertRaises(ValueError):
            transformation_pipeline(
                "not_a_dataframe",
                [sub],
            )

    def test_pipeline_with_no_transformations_raises_error(self):
        with self.assertRaises(ValueError):
            transformation_pipeline(
                self.df,
                None,
            )

    def test_pipeline_with_empty_transformations_raises_error(self):
        with self.assertRaises(ValueError):
            transformation_pipeline(
                self.df,
                [],
            )

    def test_pipeline_with_invalid_transformations_type_raises_error(self):
        with self.assertRaises(ValueError):
            transformation_pipeline(
                self.df,
                "not_a_list",
            )

    def test_pipeline_with_none_transformation_raises_error(self):
        with self.assertRaises(ValueError):
            transformation_pipeline(
                self.df,
                [None],
            )

    def test_pipeline_with_object_without_transform_method_raises_error(self):
        with self.assertRaises(ValueError):
            transformation_pipeline(
                self.df,
                [object()],
            )

    def test_pipeline_returns_original_dataframe_when_identity_transformation(self):
        class IdentityTransformation:
            def transform(self, df):
                return df

        result = transformation_pipeline(
            self.df,
            [IdentityTransformation()],
        )

        self.assertTrue(
            hasattr(result, "select")
        )

        self.assertEqual(
            result.columns,
            self.df.columns,
        )

        self.assertEqual(
            result.collect(),
            self.df.collect(),
        )


if __name__ == "__main__":
    unittest.main()