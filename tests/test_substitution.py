from pyspark.sql import SparkSession
from anonymization_lib import Substitution
import unittest


class TestSubstitution(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder
            .appName("test_substitution")
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

        cls.data_with_null = [
            ("11111111A", "Ana", 28, "28001", "F", "Covid"),
            (None, "Julia", 28, "28001", "F", "Hipertension")
        ]
        cls.df_with_null = cls.spark.createDataFrame(cls.data_with_null, cls.columns)

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_substitution_full(self):
        substituter = Substitution(column="dni",replacement_char="*",mode="full")

        df_result = substituter.transform(self.df)
        result = df_result.select("dni").first()

        self.assertEqual(result["dni"], "*********")

    def test_substitution_partial(self):
        substituter = Substitution(column="dni",replacement_char="*",mode="partial",start=0,length=5)

        df_result = substituter.transform(self.df)
        result = df_result.select("dni").first()

        self.assertEqual(result["dni"], "*****111A")

    def test_substitution_partial_length_exceeds_string(self):
        substituter = Substitution(column="dni",replacement_char="*",mode="partial",start=6,length=10)

        df_result = substituter.transform(self.df)
        result = df_result.select("dni").first()

        self.assertEqual(result["dni"], "111111***")

    def test_substitution_preserves_null(self):
        substituter = Substitution(column="dni",replacement_char="*",mode="full")

        df_result = substituter.transform(self.df_with_null)
        result = df_result.select("dni").collect()

        self.assertEqual(result[0]["dni"], "*********")
        self.assertIsNone(result[1]["dni"])

    def test_substitution_nonexistent_column(self):
        substituter = Substitution(column="ciudad",replacement_char="*",mode="full")

        with self.assertRaises(ValueError):
            substituter.transform(self.df)

    def test_substitution_invalid_column_none(self):
        with self.assertRaises(ValueError):
            Substitution(column=None,replacement_char="*",mode="full")

    def test_substitution_invalid_column_empty(self):
        with self.assertRaises(ValueError):
            Substitution(column="",replacement_char="*",mode="full")

    def test_substitution_invalid_replacement_char_empty(self):
        with self.assertRaises(ValueError):
            Substitution(column="dni",replacement_char="",mode="full")

    def test_substitution_invalid_replacement_char_multiple(self):
        with self.assertRaises(ValueError):
            Substitution(column="dni",replacement_char="**",mode="full")

    def test_substitution_invalid_replacement_char_not_string(self):
        with self.assertRaises(ValueError):
            Substitution(column="dni",replacement_char=5,mode="full")

    def test_substitution_invalid_mode(self):
        with self.assertRaises(ValueError):
            Substitution(column="dni",replacement_char="*",mode="otro")

    def test_substitution_partial_without_start(self):
        with self.assertRaises(ValueError):
            Substitution(column="dni",replacement_char="*",mode="partial",start=None,length=5)

    def test_substitution_partial_without_length(self):
        with self.assertRaises(ValueError):
            Substitution(column="dni",replacement_char="*",mode="partial",start=0,length=None)

    def test_substitution_partial_invalid_start(self):
        with self.assertRaises(ValueError):
            Substitution(column="dni",replacement_char="*",mode="partial",start=-1,length=5)

    def test_substitution_partial_invalid_length(self):
        with self.assertRaises(ValueError):
            Substitution(column="dni",replacement_char="*",mode="partial",start=0,length=0)

    def test_substitution_partial_start_not_integer(self):
        with self.assertRaises(ValueError):
            Substitution(column="dni",replacement_char="*",mode="partial",start="0",length=3)

    def test_substitution_partial_length_not_integer(self):
        with self.assertRaises(ValueError):
            Substitution(column="dni",replacement_char="*",mode="partial",start=0,length="3")

    def test_substitution_full_with_start(self):
        with self.assertRaises(ValueError):
            Substitution(column="dni",replacement_char="*",mode="full",start=0,length=3)

    def test_substitution_none_dataframe(self):
        substituter = Substitution(column="dni",replacement_char="*",mode="full")

        with self.assertRaises(ValueError):
            substituter.transform(None)

    def test_substitution_invalid_dataframe_type(self):
        model = Substitution(column="dni",replacement_char="*",mode="full")

        with self.assertRaises(ValueError):
            model.transform("not_a_dataframe")


if __name__ == "__main__":
    unittest.main()