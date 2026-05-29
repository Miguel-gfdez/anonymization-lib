# Data Anonymization Library for Big Data

## Overview
This project focuses on the design and development of a data anonymization library capable of processing large volumes of sensitive information in an efficient, scalable, and secure way. The goal is to provide mechanisms to reduce the risk of re-identification while maintaining the utility of the data.

The library is designed to operate in Big Data environments and integrates anonymization techniques with distributed data processing frameworks.

---

## Objectives

The main objectives of this project are:

- Develop a modular library for data anonymization.
- Implement common anonymization techniques.
- Evaluate privacy risks such as re-identification.
- Maintain a balance between privacy protection and data utility.
- Support large datasets using distributed processing technologies.

---

## Features

The library includes the following functionalities:

- Implementation of anonymization metrics such as:
  - k-anonymity
  - l-diversity
  - t-closeness
- Implementation of anonymization techniques such as:
  - suppression
  - substitution
  - generalization
- Data visualization tools to analyze datasets before and after anonymization, allowing the assessment of data utility and privacy impact.
- Evaluation of re-identification risk
- Distributed processing with Apache Spark
- Scalable processing of large datasets.

---

## Technologies

The project uses the following technologies:

- Python
- Apache Spark
- GitHub for version control
<!--
---

## Repository Structure
project-root

│

├── src

│ ├── anonymization

│ ├── metrics

│ └── utils

│

├── datasets

│

├── tests

│

├── docs

│

└── README.md

---

## Installation

Clone the repository:

- git clone https://github.com/yourusername/yourrepository.git

Navigate to the project folder:

- cd yourrepository

Install dependencies:

- pip install -r requirements.txt

---

## Usage

Example of how the anonymization library may be used:

from anonymization import kanonymity

dataset = load_dataset("data.csv")

anonymized_data = kanonymity.apply(dataset, k=5)

---

## Future Work

Possible improvements include:

- Additional anonymization techniques

- Integration with more Big Data frameworks

---
-->
## Author

Miguel Galán Fernández

University of Oviedo

---

## License

This project was developed for academic purposes as part of a Final Degree Project (TFG).















