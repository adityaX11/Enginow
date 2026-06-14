# Customer Purchase Prediction Using Classification Algorithms

## Project Description

Customer Purchase Prediction is a Machine Learning classification project that aims to predict whether a customer is likely to purchase a product or service based on demographic and behavioral information.

The project follows a complete machine learning pipeline, starting from data understanding and preprocessing to model building, evaluation, and business insight generation. The main objective is to identify the factors that influence customer purchasing behavior and build a predictive model that can help businesses make data-driven decisions.

---

# Project Objective

The primary goal of this project is to classify customers into two categories:

- Purchase (1)
- Not Purchase (0)

By analyzing customer characteristics and past behavior, the model learns patterns that help predict future purchasing decisions.

This type of prediction is commonly used in:

- Customer Analytics
- Marketing Campaign Optimization
- Customer Segmentation
- Sales Forecasting
- Recommendation Systems

---

# Dataset Overview

The dataset contains customer demographic information and behavioral attributes that help determine purchasing behavior.

## Dataset Features

### 1. Customer_ID
A unique identifier assigned to each customer.

### 2. Age
Represents the customer's age.

### 3. Gender
Represents the gender category of the customer.

Possible values:

- Male
- Female
- Other

### 4. Annual_Income
Represents the yearly income of the customer.

### 5. Spending_Score
Measures how actively a customer spends money.

A higher spending score indicates a stronger purchasing tendency.

### 6. Previous_Purchases
Represents the number of purchases made previously by the customer.

This feature helps identify returning customers.

### 7. Engagement_Score
Measures customer interaction and engagement with products, services, or marketing activities.

Higher engagement generally indicates greater purchase potential.

### 8. Purchase (Target Variable)

Target variable used for prediction.

| Value | Meaning |
|---------|---------|
| 0 | Customer did not purchase |
| 1 | Customer purchased |

---

# Data Understanding

Before building machine learning models, it is important to understand the dataset.

The following checks were performed:

- Dataset shape analysis
- Column identification
- Data type inspection
- Statistical summary
- Target variable analysis

These steps help understand the quality and structure of the data before preprocessing.

---

# Missing Value Analysis

One of the most important preprocessing steps is checking for missing values.

Missing values can negatively affect machine learning models and lead to inaccurate predictions.

The dataset was inspected for:

- Null values
- Empty records
- Incomplete entries

### Observation

No missing values were found in the dataset.

This indicates that the dataset was already clean and ready for preprocessing.

---

# Feature Engineering and Target Identification

The project involves identifying the target variable that the model must predict.

### Target Feature

Purchase

The target variable represents whether a customer purchases a product or not.

The remaining columns become input features that help the model learn purchasing behavior.

---

# Categorical Feature Encoding

Machine learning algorithms cannot directly process text-based categorical data.

The Gender feature contains categorical values:

- Male
- Female
- Other

To convert these categories into numerical form, One-Hot Encoding was applied.

### Generated Features

- Gender_Female
- Gender_Male
- Gender_Other

### Why One-Hot Encoding?

One-Hot Encoding:

- Converts categories into numerical format
- Prevents false ordering between categories
- Improves model performance
- Makes data suitable for machine learning algorithms

---

# Feature Scaling

Different numerical features may exist on different ranges.

For example:

- Age may range from 18–70
- Annual Income may range from thousands to lakhs
- Spending Score may range from 1–100

Such differences can affect model performance.

To solve this issue, StandardScaler was applied.

### Features Scaled

- Age
- Annual_Income
- Spending_Score
- Previous_Purchases
- Engagement_Score

### Purpose of Scaling

Feature scaling:

- Standardizes feature values
- Improves model learning
- Reduces bias caused by large numerical ranges
- Helps optimization algorithms converge faster

After scaling:

- Mean becomes approximately 0
- Standard deviation becomes approximately 1

---

# Train-Test Splitting

To evaluate model performance fairly, the dataset was divided into:

### Training Dataset

75% of data

Used to train machine learning models.

### Testing Dataset

25% of data

Used to evaluate how well the trained model performs on unseen data.

This helps measure the model's generalization ability.

---

# Exploratory Data Analysis (EDA)

EDA was performed to understand relationships, trends, and customer behavior patterns within the dataset.

EDA helps answer questions such as:

- Which features influence purchases?
- Which customers are more likely to buy?
- What relationships exist among features?

---

## Distribution Analysis

Feature distributions were analyzed using histograms.

The following features were explored:

- Age
- Annual Income
- Spending Score
- Previous Purchases
- Engagement Score

### Findings

- Customers belong to various age groups.
- Income levels vary significantly.
- Spending behavior differs across customers.
- Customer engagement varies from low to high.
- Previous purchase behavior is diverse.

---

## Correlation Analysis

A correlation heatmap was created to understand how features relate to one another.

Correlation values range between:

- +1 → Strong Positive Relationship
- 0 → No Relationship
- -1 → Strong Negative Relationship

### Key Findings

The strongest positive correlations with Purchase were:

#### Spending Score

Strongest relationship with customer purchases.

Customers who spend more are more likely to purchase again.

#### Engagement Score

Moderate positive relationship.

Highly engaged customers tend to purchase more frequently.

#### Previous Purchases

Customers with previous purchase history show a greater tendency to buy again.

### Weak Relationships

The following features showed minimal impact:

- Age
- Gender
- Annual Income

This indicates that behavioral features are more important than demographic features for predicting purchases.

---

# Customer Behavior Insights

EDA revealed several important business insights.

### High Spending Customers

Customers with higher spending scores are more likely to purchase products.

### Returning Customers

Customers with previous purchase history tend to purchase again.

### Engaged Customers

Highly engaged customers demonstrate stronger purchasing behavior.

### Demographic Influence

Age and gender showed relatively low impact on purchase decisions.

---

# Machine Learning Models Used

Two classification algorithms were implemented and compared.

---

# Logistic Regression

## Purpose

Logistic Regression is a supervised machine learning algorithm used for binary classification problems.

Since the target variable contains only two classes:

- Purchase
- Not Purchase

Logistic Regression is an appropriate baseline model.

## Characteristics

- Simple and interpretable
- Fast training
- Easy to understand
- Produces probability-based predictions

## Behavior

The model attempts to find a relationship between customer features and purchasing outcomes.

It predicts the probability that a customer belongs to the purchase class.

---

# Decision Tree Classifier

## Purpose

Decision Tree is a rule-based machine learning algorithm.

It learns decision patterns by repeatedly splitting the dataset based on feature values.

## Characteristics

- Easy to interpret
- Captures nonlinear relationships
- Handles complex decision boundaries
- Provides clear decision-making logic

## Behavior

The model learns a hierarchy of decision rules such as:

- Spending Score threshold
- Engagement threshold
- Previous Purchase threshold

These rules help classify customers into purchase and non-purchase groups.

---

# Model Evaluation

To compare model performance, several evaluation metrics were used.

---

## Accuracy

Measures overall prediction correctness.

Accuracy indicates how many predictions were classified correctly.

Higher accuracy means better overall performance.

---

## Precision

Measures the quality of positive predictions.

It answers:

"Among all predicted purchasers, how many actually purchased?"

Higher precision means fewer false positive predictions.

---

## Recall

Measures the model's ability to identify actual purchasers.

It answers:

"Among all real purchasers, how many were correctly identified?"

Higher recall means fewer missed customers.

---

## F1 Score

Combines Precision and Recall into a single metric.

Useful when balancing both metrics is important.

A higher F1 score indicates better classification performance.

---

# Model Performance Summary

| Model | Accuracy | Precision | Recall | F1 Score |
|---------|---------|---------|---------|---------|
| Logistic Regression | 84.4% | 80.0% | 75.6% | 77.7% |
| Decision Tree | 100% | 100% | 100% | 100% |

---

# Best Performing Model

Based on all evaluation metrics, the Decision Tree Classifier achieved the highest performance.

### Reasons

- Highest Accuracy
- Highest Precision
- Highest Recall
- Highest F1 Score

The model successfully identified purchasing behavior more effectively than Logistic Regression.

---

# Project Insights

The project revealed several important findings:

### Insight 1

Customer behavior is more important than customer demographics.

### Insight 2

Spending Score is the strongest predictor of purchase behavior.

### Insight 3

Engagement Score significantly affects customer purchasing decisions.

### Insight 4

Previous Purchases strongly influence future purchasing behavior.

### Insight 5

Age and Gender contribute very little to purchase prediction.

### Insight 6

Behavioral features provide the most valuable information for customer targeting and marketing strategies.

---

# Business Applications

This project can help organizations:

- Identify potential buyers
- Improve customer targeting
- Increase marketing efficiency
- Enhance customer retention
- Optimize promotional campaigns
- Support data-driven decision making

---

# Conclusion

This project successfully implemented an end-to-end Machine Learning classification pipeline for customer purchase prediction.

The workflow included:

- Dataset Understanding
- Missing Value Analysis
- Target Identification
- One-Hot Encoding
- Feature Scaling
- Exploratory Data Analysis
- Model Training
- Model Evaluation
- Model Comparison
- Business Insight Generation

Among the evaluated algorithms, the Decision Tree Classifier achieved the best performance and was selected as the final model.

The analysis demonstrated that Spending Score, Engagement Score, and Previous Purchases are the most influential factors affecting customer purchasing behavior, making them valuable indicators for future customer targeting and business growth strategies.