from pyspark.sql import DataFrame
from pyspark.sql import functions as F

    
class Suppression:
    """
    Applies suppression techniques to one or multiple columns.
    """

    def __init__(self, columns_modes: dict):
        """
        Initializes the suppression transformation.

        Parameters
        ----------
        columns_modes : dict
            Dictionary where keys are column names and values are suppression modes.

            Supported modes:
            - 'null': replaces all values in the column with NULL.
            - 'drop': removes the column from the dataset.

            Example:
            {"age": "null", "name": "drop"}
        """
        if not isinstance(columns_modes, dict):
            raise ValueError("'columns_modes' must be a dictionary.")
        
        if not columns_modes:
            raise ValueError("'columns_modes' must not be empty.")
        
        if not all(isinstance(col, str) for col in columns_modes.keys()):
            raise ValueError("All column names must be strings.")
        
        if not all(isinstance(mode, str) for mode in columns_modes.values()):
            raise ValueError("All modes must be strings.")
        
        valid_modes = {"null", "drop"}

        for col, mode in columns_modes.items():
            if mode not in valid_modes:
                raise ValueError(f"Invalid mode '{mode}' for column '{col}'.")

        self.columns_modes = columns_modes

    def transform(self, df: DataFrame) -> DataFrame:
        if df is None:
            raise ValueError("Input DataFrame cannot be None.")

        if not isinstance(df, DataFrame):
            raise ValueError("Input must be a pyspark.sql.DataFrame.")
        
        result_df = df

        for column, mode in self.columns_modes.items():
            if column not in result_df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame.")

            if mode == "drop":
                result_df = result_df.drop(column)

            elif mode == "null":
                # result_df = result_df.withColumn(column, F.lit(None))

                original_type = result_df.schema[column].dataType
                # print(original_type)
                result_df = result_df.withColumn(column,F.lit(None).cast(original_type))
                

        return result_df