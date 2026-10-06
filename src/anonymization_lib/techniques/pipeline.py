from pyspark.sql import DataFrame


def transformation_pipeline(
    df: DataFrame,
    transformations: list = None
) -> DataFrame:
    """
    Applies a sequence of anonymization transformations to a Spark DataFrame.

    Each transformation is applied sequentially, where the output of one
    transformation becomes the input of the next transformation.

    All transformation objects must implement a callable
    ``transform(df: DataFrame) -> DataFrame`` method.

    This allows different anonymization techniques, such as suppression,
    substitution and generalization, to be combined without coupling the
    pipeline to specific transformation classes.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Input dataset to be anonymized.

    transformations : list
        Ordered list of transformation objects to apply.

        Each object must provide a callable ``transform`` method that
        receives a Spark DataFrame and returns another Spark DataFrame.

        Examples of compatible transformations include:

        - ``Suppression``
        - ``Substitution``
        - ``Generalization``

        The order of the list determines the order in which transformations
        are applied.

    Returns
    -------
    pyspark.sql.DataFrame
        DataFrame resulting from applying all configured transformations
        sequentially.

    Raises
    ------
    ValueError
        If the input DataFrame is None or is not Spark-compatible.

        If ``transformations`` is None, is not a list, or is empty.

        If any element in ``transformations`` is None or does not implement
        a callable ``transform`` method.

    Notes
    -----
    The pipeline does not trigger Spark execution by itself. Transformations
    based on Spark DataFrame operations are lazy, so the resulting logical
    plan is evaluated only when a Spark action is executed.

    Spark's query optimizer can optimize the combined logical plan generated
    by consecutive transformations.

    The pipeline intentionally relies on the common ``transform`` interface
    instead of checking for specific transformation classes. This allows
    custom transformations to be used as long as they implement the expected
    interface.

    The order of transformations may affect the resulting dataset. For
    example, dropping a column before another transformation attempts to use
    it will cause that subsequent transformation to fail.
    """

    # ------------------------------------------------------
    # VALIDATE INPUT DATAFRAME
    # ------------------------------------------------------

    if df is None:
        raise ValueError(
            "Input DataFrame cannot be None."
        )

    # Compatible with standard Spark and Spark Connect.
    if not hasattr(df, "select"):
        raise ValueError(
            "Input must be a Spark-compatible DataFrame."
        )

    # ------------------------------------------------------
    # VALIDATE TRANSFORMATIONS
    # ------------------------------------------------------

    if transformations is None:
        raise ValueError(
            "'transformations' must be provided and cannot be None."
        )

    if not isinstance(transformations, list):
        raise ValueError(
            "'transformations' must be a list."
        )

    if not transformations:
        raise ValueError(
            "'transformations' cannot be an empty list."
        )

    for index, transformation in enumerate(transformations):
        if transformation is None:
            raise ValueError(
                f"Transformation at index {index} cannot be None."
            )

        if not callable(
            getattr(transformation, "transform", None)
        ):
            raise ValueError(
                f"Transformation at index {index} "
                f"('{type(transformation).__name__}') must implement "
                "a callable 'transform' method."
            )

    # ------------------------------------------------------
    # APPLY TRANSFORMATIONS
    # ------------------------------------------------------

    final_df = df

    for transformation in transformations:
        final_df = transformation.transform(final_df)

    return final_df