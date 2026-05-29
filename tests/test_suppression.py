from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from anonymization_lib import Suppression
import unittest


class TestSuppression(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder
            .appName("test_suppression")
            .master("local[1]")
            .getOrCreate()
        )

        cls.columns = ["dni", "nombre", "edad", "cp", "sexo", "fecha"]
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

    def test_suppression_drop_and_null(self):
        model = Suppression(columns_modes={"nombre": "drop", "dni": "null"})
        result = model.transform(self.df)

        self.assertNotIn("nombre", result.columns)
        self.assertIn("dni", result.columns)
        self.assertIn("edad", result.columns)
        self.assertIn("cp", result.columns)
        self.assertIn("sexo", result.columns)
        self.assertIn("fecha", result.columns)

        non_null_count = result.select(F.count(F.col("dni"))).first()[0]
        self.assertEqual(non_null_count, 0)

    def test_column_not_found(self):
        model = Suppression(columns_modes={"apellido": "drop"})

        with self.assertRaises(ValueError):
            model.transform(self.df)

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            Suppression(columns_modes={"nombre": "invalid_mode"})

    def test_columns_modes_not_dict(self):
        with self.assertRaises(ValueError):
            Suppression(columns_modes=["nombre", "drop"])

    def test_columns_modes_empty(self):
        with self.assertRaises(ValueError):
            Suppression(columns_modes={})

    def test_columns_modes_non_string_key(self):
        with self.assertRaises(ValueError):
            Suppression(columns_modes={123: "drop"})

    def test_columns_modes_non_string_value(self):
        with self.assertRaises(ValueError):
            Suppression(columns_modes={"nombre": 123})

    def test_transform_none_dataframe(self):
        model = Suppression(columns_modes={"nombre": "drop"})

        with self.assertRaises(ValueError):
            model.transform(None)

    def test_transform_invalid_dataframe_type(self):
        model = Suppression(columns_modes={"nombre": "drop"})

        with self.assertRaises(ValueError):
            model.transform("not_a_dataframe")


if __name__ == "__main__":
    unittest.main()