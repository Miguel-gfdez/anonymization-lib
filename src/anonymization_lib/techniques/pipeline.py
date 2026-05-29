from pyspark.sql import DataFrame

def TransformationPipeline(df: DataFrame ,transformations: list = None) -> DataFrame:
    """
    Applies a sequence of anonymization transformations to a Spark DataFrame.

    Each transformation is applied in order, where the output of one
    transformation becomes the input of the next. All transformations
    must implement a `transform(df: DataFrame) -> DataFrame` method.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Input dataset to be anonymized.

    transformations : list
        List of transformation objects (e.g., Suppression, Substitution, Generalization).
        The order of the list determines the execution order.

    Returns
    -------
    pyspark.sql.DataFrame
        Transformed DataFrame after applying all transformations.

    """

    if transformations is None:
        raise ValueError("'transformations' must be provided and cannot be None.")

    if not isinstance(transformations, list):
        raise ValueError("'transformations' must be a list.")

    if len(transformations) == 0:
        raise ValueError("'transformations' cannot be an empty list.")
    
    transformations = list(transformations)
    final_df = df

    for transformation in transformations:
        if transformation is None:
            raise ValueError(f"Transformation '{transformation}' is None.")

        if not hasattr(transformation, "transform") or not callable(transformation.transform):
            raise ValueError(f"Transformation '{transformation}' must implement a callable 'transform' method.")

    for transformation in transformations:
        final_df = transformation.transform(final_df)

    
    return final_df



