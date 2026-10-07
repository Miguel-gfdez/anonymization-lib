# Data Anonymization Library for Big Data

[![CI](https://github.com/Miguel-gfdez/anonymization-lib/actions/workflows/CI.yml/badge.svg)](https://github.com/Miguel-gfdez/anonymization-lib/actions)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=Miguel-gfdez_anonymization-lib&metric=alert_status)](https://sonarcloud.io/project/overview?id=Miguel-gfdez_anonymization-lib)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=Miguel-gfdez_anonymization-lib&metric=coverage)](https://sonarcloud.io/project/overview?id=Miguel-gfdez_anonymization-lib)

Data anonymization library developed with Python and Apache Spark for processing large volumes of sensitive information in a scalable, efficient, and modular way.

The main goal of this project is to provide privacy-preserving mechanisms that reduce the risk of re-identification while maintaining the analytical utility of the data.

---

## Overview

This project focuses on the design and development of a data anonymization library capable of working with large datasets in Big Data environments.

The library integrates privacy metrics, anonymization techniques, recommendation tools, and visualization capabilities using distributed data processing with Apache Spark.

The implementation is designed to work with standard PySpark environments as well as Spark Connect-based environments such as Databricks.

---

## Features

The library includes the following functionalities:

- Privacy metrics:
  - k-anonymity
  - l-diversity
  - t-closeness

- Anonymization techniques:
  - Suppression
  - Substitution
  - Generalization
  - Sequential transformation pipelines

- Anonymization advisor:
  - Analysis of quasi-identifier cardinality
  - Analysis of equivalence-group reduction
  - Risk-level estimation
  - Suggested anonymization actions

- Data visualization tools:
  - Exploratory analysis before anonymization
  - Support for assessing privacy impact and data utility

- Data management:
  - Import utilities for common data formats
  - Export utilities for common data formats

- Big Data support:
  - Distributed processing with Apache Spark
  - Spark Connect compatibility
  - Scalable processing of large datasets

---

## Technologies

The project uses the following technologies:

- Python
- Apache Spark
- PySpark
- Plotly
- GitHub Actions
- SonarCloud

---

## Installation

### Install from PyPI

If Spark is not already provided by your environment, install the library with the optional PySpark dependency:

```bash
pip install "anonymization-lib[local]"
```

In managed Spark environments, such as Databricks, install the library without the optional PySpark dependency:

```bash
pip install anonymization-lib
```

This prevents the package from replacing the PySpark version managed by the platform.

### Install from source

Clone the repository:

```bash
git clone https://github.com/Miguel-gfdez/anonymization-lib.git
cd anonymization-lib
```

For local development:

```bash
pip install -e ".[local]"
```

In managed Spark environments, such as Databricks:

```bash
pip install -e .
```

The editable installation (`-e`) allows changes to the source code to be reflected without reinstalling the package.

## Quick Start

```python
# 1. Create a Spark session
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("AnonymizationExample")
    .getOrCreate()
)


# 2. Load a dataset
from anonymization_lib import DataImporter

df = DataImporter.import_data(
    spark=spark,
    path="data.parquet",
    file_format="parquet",
)


# 3. Visualize data
from anonymization_lib import Visualization

viz = Visualization(column="AGE").transform(df)
viz.show()


# 4. Apply anonymization techniques
from anonymization_lib import Generalization, Substitution, Suppression
from anonymization_lib.techniques import TransformationPipeline

supp = Suppression(
    columns_modes={
        "NAME": "null",
        "CITY": "drop",
    }
)

sub = Substitution(
    column="NATIONAL_ID",
    replacement_char="*",
    mode="full",
)

gen_age = Generalization(
    column="AGE",
    rules_path="data/gen_age.json",
)

gen_postal_code = Generalization(
    column="POSTAL_CODE",
    rules_path="data/gen_postal_code_region.json",
    output_column="REGION",
)

pipeline = [
    supp,
    sub,
    gen_age,
    gen_postal_code,
]

df_anonymized = TransformationPipeline(
    df,
    pipeline,
)


# 5. Evaluate privacy metrics
from anonymization_lib import KAnonymity

k_metric = KAnonymity(
    quasi_identifiers=["AGE", "REGION"]
)

result = k_metric.summary(df_anonymized)

result.show_summary()
result.show_violating_groups()


# 6. Get anonymization recommendations
from anonymization_lib import AnonymizationAdvisor

advisor = AnonymizationAdvisor(
    quasi_identifiers=["AGE", "REGION"],
    k=3,
)

recommendations = advisor.suggest(df_anonymized)

recommendations.get_summary_df().show()
recommendations.get_suggestions_df().show()
```

---

## Project Status

Current stable version: **v1.3.0**

Implemented modules:

- Privacy metrics
- Anonymization techniques
- Transformation pipelines
- Visualization tools
- Import/export utilities
- Anonymization advisor
- Spark Connect compatibility

---

## Author

Miguel Galán Fernández

---

## License

This project is licensed under the MIT License.

See the LICENSE file for more details.