# Data Anonymization Library for Big Data

[![CI](https://github.com/Miguel-gfdez/anonymization-lib/actions/workflows/CI.yml/badge.svg)](https://github.com/Miguel-gfdez/anonymization-lib/actions)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=Miguel-gfdez_anonymization-lib&metric=alert_status)](https://sonarcloud.io/project/overview?id=Miguel-gfdez_anonymization-lib)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=Miguel-gfdez_anonymization-lib&metric=coverage)](https://sonarcloud.io/project/overview?id=Miguel-gfdez_anonymization-lib)

A Python library for **scalable data anonymization using Apache Spark**, designed to reduce re-identification risks while preserving the analytical utility of large datasets.

The library combines privacy metrics, anonymization techniques, automated recommendations, and data visualization in a modular framework compatible with standard PySpark and Spark Connect environments.

## Features

| Category | Functionality |
|---|---|
| Privacy metrics | k-anonymity, l-diversity, t-closeness |
| Anonymization | Suppression, substitution, generalization |
| Transformation pipelines | Sequential application of anonymization techniques |
| Anonymization advisor | Quasi-identifier analysis, risk estimation, transformation recommendations |
| Visualization | Exploratory analysis and privacy assessment |
| Data management | Import and export utilities |
| Distributed processing | Apache Spark and Spark Connect compatibility |

### Privacy metrics

The library provides three privacy metrics for assessing disclosure risks:

- **k-anonymity:** evaluates whether each equivalence group contains at least *k* records.
- **l-diversity:** evaluates the diversity of sensitive attribute values within equivalence groups.
- **t-closeness:** measures the distance between sensitive attribute distributions within equivalence groups and the overall dataset.

Each metric provides aggregated results and information about groups that violate the configured privacy threshold.

### Anonymization techniques

Three anonymization techniques are supported:

- **Suppression:** removes or nullifies selected attributes.
- **Substitution:** replaces attribute values using configurable replacement rules.
- **Generalization:** reduces data granularity through predefined rules.

Techniques can be applied individually or combined into sequential transformation pipelines.

### Anonymization advisor

The `AnonymizationAdvisor` provides recommendations based on the selected quasi-identifiers and the characteristics of the dataset.

Its functionality includes cardinality analysis, equivalence-group reduction, risk-level estimation, and suggested anonymization actions.

## Installation

The library supports installation from PyPI, Wheel distributions, or source code.

### From PyPI

Once the package is published on PyPI, install it with:

```bash
pip install anonymization-lib
```

For local environments that require PySpark as an additional dependency:

```bash
pip install "anonymization-lib[local]"
```

### From a Wheel

Download the appropriate `.whl` file from a project distribution and install it using:

```bash
pip install path/to/anonymization_lib-<version>-py3-none-any.whl
```

For local environments requiring the optional Spark dependencies:

```bash
pip install "path/to/anonymization_lib-<version>-py3-none-any.whl[local]"
```

Replace `<version>` with the actual package version.

### From source

Clone the repository:

```bash
git clone https://github.com/Miguel-gfdez/anonymization-lib.git
cd anonymization-lib
```

Install the library in editable mode:

```bash
pip install -e ".[local]"
```

For environments where Spark is already provided, such as Databricks:

```bash
pip install -e .
```

Editable installations allow source code changes to take effect without reinstalling the package.

**Note:** In managed Spark environments, avoid installing the optional local PySpark dependency unless necessary, to prevent conflicts with the Spark version provided by the platform.

## Quick Start

The following example demonstrates how to evaluate k-anonymity on a Spark DataFrame.

```python
from pyspark.sql import SparkSession
from anonymization_lib import KAnonymity

# Create a Spark session
spark = (
    SparkSession.builder
    .appName("AnonymizationExample")
    .getOrCreate()
)

# Load a dataset
df = spark.read.parquet("data.parquet")

# Evaluate k-anonymity
metric = KAnonymity(
    quasi_identifiers=["AGE", "CITY"],
    k_threshold=3,
)

result = metric.summary(df)

# Display results
result.show_summary()
result.show_violating_groups()
```

The resulting summary provides information about equivalence groups, privacy threshold compliance, and potentially violating groups.

## Usage Examples

### Data visualization

Generate an exploratory visualization for a selected column:

```python
from anonymization_lib import Visualization

fig = Visualization(
    column="AGE"
).transform(df)

fig.show()
```

### Anonymization pipeline

Combine multiple anonymization techniques into a single workflow:

```python
from anonymization_lib import (
    Generalization,
    Substitution,
    Suppression,
)
from anonymization_lib.techniques import TransformationPipeline

pipeline = [
    Suppression(
        columns_modes={
            "NAME": "null",
            "CITY": "drop",
        }
    ),
    Substitution(
        column="NATIONAL_ID",
        replacement_char="*",
        mode="full",
    ),
    Generalization(
        column="AGE",
        rules_path="data/gen_age.json",
    ),
    Generalization(
        column="POSTAL_CODE",
        rules_path="data/gen_postal_code_region.json",
        output_column="REGION",
    ),
]

df_anonymized = TransformationPipeline(
    df,
    pipeline,
)
```

Transformations are applied sequentially, with the output of each transformation serving as the input to the next.

### Anonymization recommendations

Analyze quasi-identifiers and obtain anonymization recommendations:

```python
from anonymization_lib import AnonymizationAdvisor

advisor = AnonymizationAdvisor(
    quasi_identifiers=["AGE", "REGION"],
    k=3,
)

recommendations = advisor.suggest(df_anonymized)

recommendations.get_summary_df().show()
recommendations.get_suggestions_df().show()
```

### Data export

Export an anonymized DataFrame using Spark:

```python
df_anonymized.write.mode("overwrite").parquet(
    "output/anonymized"
)
```

## Development and Testing

The project uses Python's `unittest` framework for automated testing and `coverage` for measuring code coverage.

### Install development dependencies

```bash
pip install -e ".[local]"
pip install coverage
```

### Run tests

```bash
python -m unittest discover -s tests -p "test_*.py" -t .
```

### Generate a coverage report

```bash
coverage run --source=src/anonymization_lib -m unittest discover -s tests -p "test_*.py" -t .
coverage report -m
coverage html
```

The HTML coverage report is generated in `htmlcov/`.

### Continuous Integration and Delivery

The project uses GitHub Actions for automated workflows.

**Continuous Integration (CI)** performs unit testing, code coverage analysis, static analysis with SonarCloud, and package building.

**Continuous Delivery (CD)** builds Wheel and source distributions when a GitHub Release is published. PyPI publishing is planned but is not currently enabled in the release workflow.

## Technologies

- **Python:** library development and testing.
- **Apache Spark / PySpark:** distributed data processing.
- **Spark Connect:** compatibility with remote Spark environments.
- **Plotly:** interactive data visualization.
- **GitHub Actions:** CI/CD automation.
- **SonarCloud:** static code analysis and quality monitoring.

## Project Status

**Current stable version:** v1.4.0

The library includes implemented privacy metrics, anonymization techniques, transformation pipelines, visualization utilities, data import/export, and an anonymization advisor.

## Author

Miguel Galán Fernández

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.