# Customer Churn Analysis Project

## Overview
This project analyzes customer churn patterns to help businesses identify at-risk customers and develop retention strategies. The analysis includes SQL queries, Python machine learning models, and Power BI dashboards.

## Problem Statement
Businesses lose revenue when customers stop using their services. This project aims to:
- Identify at-risk customers through predictive modeling
- Profile customer segments to understand churn patterns
- Provide actionable insights for retention strategies

## Data Sources
- Kaggle's Telco Customer Churn Dataset
- Bank/telecom/customer behavior datasets

## Project Structure
```
churn/
├── data/                   # Raw and processed datasets
├── sql/                    # SQL queries for data analysis
├── notebooks/              # Jupyter notebooks for EDA and modeling
├── src/                    # Python source code
├── powerbi/               # Power BI dashboard files
├── results/               # Model outputs and analysis results
└── docs/                  # Documentation and reports
```

## Workflow

### 1. SQL Analysis
- Query customer demographics and subscription types
- Analyze usage behavior patterns
- Create aggregated tables for segment analysis

### 2. Python Analysis
- Exploratory Data Analysis (EDA)
- Churn prediction using Logistic Regression and Random Forest
- Customer segmentation using KMeans clustering
- Feature importance analysis

### 3. Power BI Dashboard
- Churn Rate visualization
- Churn analysis by customer segments
- At-risk customer identification
- Retention strategy insights

## Deliverables
1. **Power BI Dashboard**: "Customer Retention Dashboard"
2. **Python Notebook**: Churn prediction model with feature importance
3. **SQL Queries**: Customer segmentation and analysis queries
4. **Documentation**: Analysis insights and recommendations

## Getting Started
1. Install required Python packages: `pip install -r requirements.txt`
2. Download the dataset and place it in the `data/` directory
3. Run the Jupyter notebooks in the `notebooks/` directory
4. Execute SQL queries for data preparation
5. Open the Power BI dashboard for visualization

## Key Insights
- Analysis of high-paying vs low-paying customer churn patterns
- Customer segment profiling by age group, region, and plan type
- Predictive model for identifying at-risk customers
- Actionable retention strategies based on data insights
