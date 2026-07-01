# Data Anonymization Library for Big Data

[![CI](https://github.com/Miguel-gfdez/anonymization-lib/actions/workflows/test.yml/badge.svg)](https://github.com/Miguel-gfdez/anonymization-lib/actions) 
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=Miguel-gfdez_anonymization-lib&metric=alert_status)](https://sonarcloud.io/project/overview?id=Miguel-gfdez_anonymization-lib)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=Miguel-gfdez_anonymization-lib&metric=coverage)](https://sonarcloud.io/project/overview?id=Miguel-gfdez_anonymization-lib)

Data anonymization library developed with Python and Apache Spark for processing large volumes of sensitive information in a scalable, efficient and modular way.

The main goal of this project is to provide privacy-preserving mechanisms that reduce the risk of re-identification while maintaining the analytical utility of the data.

<!--
Contents
    - Overview
    - Features
    - Technologies
    - Repository Structure
    - Installation
    - Quick Start
    - Project Status
    - Author
    - License
-->
---

## Overview

This project focuses on the design and development of a data anonymization library capable of working with large datasets in Big Data environments.

The library integrates privacy metrics, anonymization techniques and visualization tools using distributed data processing with Apache Spark.

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
      
- Data visualization tools:
    - Exploratory analysis before anonymization
    - Support for assessing privacy impact and data utility

- Additional utilities
    - Distributed processing with Apache Spark.
    - Scalable processing of large datasets.
    - Import/export utilities for working with common data formats.

---

## Technologies

The project uses the following technologies:

- Python
- Apache Spark
- PySpark
- Plotly
- GitHub for version control
- SonarCloud

---

## Installation

Clone the repository:

- git clone https://github.com/Miguel-gfdez/anonymization-lib.git
- cd anonymization-lib

Install the package:

- pip install .

---

## Quick Start

```python
## 1. Create a Spark session
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("AnonymizationExample") \
    .getOrCreate()



## 2. Load a dataset
from anonymization_lib import DataImporter

df = DataImporter.import_data(
    spark=spark,
    path="data.parquet",
    file_format="parquet"
)


## 3. Visualize data
from anonymization_lib import Visualization

viz = Visualization(column="AGE").transform(df)
viz.show()


## 4. Apply anonymization techniques
from anonymization_lib import Suppression, Substitution, Generalization

supp = Suppression(columns_modes={ "Name": "null", "City": "drop"} )

sub = Substitution(column="NATIONAL_ID", replacement_char="*", mode="full" )

gen_age = Generalization(column="AGE", rules_path="data/gen_age.json" )

gen_pc = Generalization(column="POSTAL_CODE",rules_path="data/gen_cp_region.json",output_column="REGION")

pipeline = [supp, sub, gen_age, gen_pc] 

from anonymization_lib.techniques import TransformationPipeline
df_anonymized = TransformationPipeline(df, pipeline)


## 5. Evaluate privacy metrics
from anonymization_lib import KAnonymity

k_metric = KAnonymity(quasi_identifiers=["AGE", "REGION"])

result = k_metric.summary(df_anonymized)

result.show_summary()
result.show_violating_groups()
```

---

Project Status

Current stable version: v1.2.0

Implemented modules:

Privacy metrics
Anonymization techniques
Visualization tools
Import/Export utilities
Anonymity advisor

---

## Author

Miguel Galán Fernández

---

## License

This project is licensed under the MIT License.

See the LICENSE file for more details.















