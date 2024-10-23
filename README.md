# Heart-Disease-Prediction
## Overview

This project analyzes a dataset related to heart disease and builds a predictive model to determine the likelihood of heart disease in individuals based on various health features. The dataset contains several health indicators, including age, sex, cholesterol levels, and more. The goal is to perform exploratory data analysis (EDA), visualize relationships, and apply machine learning techniques to predict heart disease outcomes.

## Table of Contents

- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Modeling](#modeling)
- [Results](#results)
- [Conclusion](#conclusion)

## Technologies Used

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Dataset

The dataset used in this project is the **Heart Disease Dataset** (`heart.csv`). It contains 303 rows and 14 columns with the following features:

| Feature                                     | Description                                   |
|---------------------------------------------|-----------------------------------------------|
| age                                         | Age of the individual                          |
| sex                                         | Gender (1 = male, 0 = female)                |
| chest_pain_types                            | Type of chest pain (0-3)                      |
| resting_blood_pressure                      | Resting blood pressure (mm Hg)                |
| cholesterol                                 | Serum cholesterol (mg/dl)                     |
| fasting_blood_sugar                         | Fasting blood sugar (1 = true, 0 = false)    |
| resting_electrocardiographic_results       | Resting electrocardiographic results (0-2)    |
| max_heart_rate                              | Maximum heart rate achieved                    |
| exercise_induced_angina                     | Exercise induced angina (1 = yes, 0 = no)    |
| oldpeak                                     | ST depression induced by exercise              |
| slope                                       | Slope of the peak exercise ST segment         |
| caa                                         | Number of major vessels (0-3)                 |
| thalassemia                                 | Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect) |
| heart_disease                               | Heart disease diagnosis (1 = yes, 0 = no)     |

## Installation

To run this project, you will need to have Python installed on your machine. You can install the required packages using `pip`:

```bash
pip install pandas matplotlib seaborn scikit-learn
