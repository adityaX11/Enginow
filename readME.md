# Customer Purchase Prediction Using Classification Algorithms

## Project Overview

This project focuses on predicting whether a customer will purchase a product or service based on demographic and behavioral data using Machine Learning Classification Algorithms.

The project follows a complete Machine Learning workflow including:

- Data Loading & Understanding
- Data Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Building
- Model Evaluation
- Model Comparison
- Business Insights

The objective is to help businesses identify potential customers and improve marketing strategies using data-driven decision making.

Project requirements were followed according to the internship task document. :contentReference[oaicite:0]{index=0}

---

# Project Objective

Build a classification model capable of predicting customer purchase behavior using:

- Age
- Gender
- Annual Income
- Spending Score
- Previous Purchases
- Engagement Score

Target Variable:

- Purchase = 1 (Customer Purchases)
- Purchase = 0 (Customer Does Not Purchase)

The goal is to determine which machine learning model performs best for customer purchase prediction. :contentReference[oaicite:1]{index=1}

---

# Dataset Information

Dataset Size:

- Records: 1000 Customers
- Features: 7 Original Features
- Target Variable: Purchase

### Features

| Feature | Description |
|----------|-------------|
| Customer_ID | Unique Customer Identifier |
| Age | Customer Age |
| Gender | Customer Gender |
| Annual_Income | Annual Income |
| Spending_Score | Spending Behavior Score |
| Previous_Purchases | Number of Previous Purchases |
| Engagement_Score | Customer Engagement Level |
| Purchase | Target Variable |

Dataset exploration confirmed:

- 1000 rows
- No missing values
- Mixed numerical and categorical features

:contentReference[oaicite:2]{index=2}

---

# Technologies Used

## Programming Language

- Python 3

## Libraries

### Data Manipulation

```python
import pandas as pd
import numpy as np
