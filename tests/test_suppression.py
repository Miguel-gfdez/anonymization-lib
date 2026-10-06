import unittest

from pyspark.sql import SparkSession

from anonymization_lib import Suppression


class TestSuppression(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder
            .appName("test_suppression")
            .master("local[1]")
            .getOrCreate()
        )

        cls.columns = [
            "dni",
            "nombre",
            "edad",
            "cp",
            "sexo",
            "fecha",
        ]

        cls.data = [
            ("11111111A", "Ana", 28, "28001", "F", "Covid"),
            ("22222222B", "Julia", 28, "28001", "F", "Hipertension"),
            ("33333333C", "Maria", 35, "28002", "F", "Covid"),
            ("44444444D", "Lucia", 35, "28002", "F", "Covid"),
            ("55555555E", "Laura", 35, "28002", "F", "Migraña"),
            ("66666666F", "Nerea", 35, "28002", "F", "Anemia"),
        ]

        cls.df = cls.spark.createDataFrame(
            cls.data,
            cls.columns,
        )

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_suppression_drop_and_null(self):
        model = Suppression(
            columns_modes={
                "nombre": "drop",
                "dni": "null",
            }
        )

        result = model.transform(self.df)

        self.assertTrue(
            hasattr(result, "select")
        )

        self.assertNotIn(
            "nombre",
            result.columns,
        )

        self.assertIn(
            "dni",
            result.columns,
        )
        self.assertIn(
            "edad",
            result.columns,
        )
        self.assertIn(
            "cp",
            result.columns,
        )
        self.assertIn(
            "sexo",
            result.columns,
        )
        self.assertIn(
            "fecha",
            result.columns,
        )

        values = [
            row["dni"]
            for row in result.select("dni").collect()
        ]

        self.assertTrue(
            all(value is None for value in values)
        )

    def test_null_suppression_preserves_column_type(self):
        original_type = self.df.schema["edad"].dataType

        model = Suppression(
            columns_modes={
                "edad": "null",
            }
        )

        result = model.transform(self.df)

        result_type = result.schema["edad"].dataType

        self.assertEqual(
            result_type,
            original_type,
        )

        values = [
            row["edad"]
            for row in result.select("edad").collect()
        ]

        self.assertTrue(
            all(value is None for value in values)
        )

    def test_drop_multiple_columns(self):
        model = Suppression(
            columns_modes={
                "nombre": "drop",
                "cp": "drop",
                "sexo": "drop",
            }
        )

        result = model.transform(self.df)

        self.assertNotIn(
            "nombre",
            result.columns,
        )
        self.assertNotIn(
            "cp",
            result.columns,
        )
        self.assertNotIn(
            "sexo",
            result.columns,
        )

        self.assertIn(
            "dni",
            result.columns,
        )
        self.assertIn(
            "edad",
            result.columns,
        )

    def test_null_multiple_columns(self):
        model = Suppression(
            columns_modes={
                "dni": "null",
                "nombre": "null",
            }
        )

        result = model.transform(self.df)

        rows = (
            result
            .select("dni", "nombre")
            .collect()
        )

        for row in rows:
            self.assertIsNone(
                row["dni"]
            )
            self.assertIsNone(
                row["nombre"]
            )

    def test_column_not_found(self):
        model = Suppression(
            columns_modes={
                "apellido": "drop",
            }
        )

        with self.assertRaises(ValueError):
            model.transform(self.df)

    def test_multiple_columns_not_found(self):
        model = Suppression(
            columns_modes={
                "apellido": "drop",
                "direccion": "null",
            }
        )

        with self.assertRaises(ValueError):
            model.transform(self.df)

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            Suppression(
                columns_modes={
                    "nombre": "invalid_mode",
                }
            )

    def test_columns_modes_not_dict(self):
        with self.assertRaises(ValueError):
            Suppression(
                columns_modes=[
                    "nombre",
                    "drop",
                ]
            )

    def test_columns_modes_none(self):
        with self.assertRaises(ValueError):
            Suppression(
                columns_modes=None
            )

    def test_columns_modes_empty(self):
        with self.assertRaises(ValueError):
            Suppression(
                columns_modes={}
            )

    def test_columns_modes_non_string_key(self):
        with self.assertRaises(ValueError):
            Suppression(
                columns_modes={
                    123: "drop",
                }
            )

    def test_columns_modes_empty_key(self):
        with self.assertRaises(ValueError):
            Suppression(
                columns_modes={
                    "": "drop",
                }
            )

    def test_columns_modes_whitespace_key(self):
        with self.assertRaises(ValueError):
            Suppression(
                columns_modes={
                    "   ": "drop",
                }
            )

    def test_columns_modes_non_string_value(self):
        with self.assertRaises(ValueError):
            Suppression(
                columns_modes={
                    "nombre": 123,
                }
            )

    def test_transform_none_dataframe(self):
        model = Suppression(
            columns_modes={
                "nombre": "drop",
            }
        )

        with self.assertRaises(ValueError):
            model.transform(None)

    def test_transform_invalid_dataframe_type(self):
        model = Suppression(
            columns_modes={
                "nombre": "drop",
            }
        )

        with self.assertRaises(ValueError):
            model.transform(
                "not_a_dataframe"
            )


if __name__ == "__main__":
    unittest.main()