# Customer Retention Dashboard - Power BI Setup Guide

## Overview
This guide provides instructions for creating a comprehensive Customer Retention Dashboard in Power BI using the analyzed customer churn data.

## Data Sources Required
1. **customers_with_clusters.csv** - Customer data with cluster assignments
2. **cluster_analysis.csv** - Cluster statistics and analysis
3. **churn_analysis.db** - SQLite database with customer data (optional)

## Dashboard Structure

### Page 1: Executive Summary
**Key Metrics Cards:**
- Total Customers
- Overall Churn Rate %
- Monthly Revenue at Risk
- Average Customer Lifetime Value

**Main Visualizations:**
1. **Churn Rate Trend** (Line Chart)
   - X-axis: Tenure Cohorts (0-6 months, 7-12 months, etc.)
   - Y-axis: Churn Rate %
   - Shows churn patterns over customer lifecycle

2. **Revenue Impact** (Donut Chart)
   - Shows revenue split between churned vs retained customers
   - Include total revenue values

### Page 2: Customer Segmentation Analysis
**Visualizations:**
1. **Customer Segments Overview** (Clustered Column Chart)
   - X-axis: Customer Clusters
   - Y-axis: Number of Customers
   - Color by: Churn Status

2. **Segment Profiling** (Table/Matrix)
   - Rows: Customer Clusters
   - Columns: Avg Tenure, Avg Monthly Charges, Churn Rate, Customer Count
   - Conditional formatting for high-risk segments

3. **Segment Scatter Plot** (Scatter Chart)
   - X-axis: Average Tenure
   - Y-axis: Average Monthly Charges
   - Bubble Size: Customer Count
   - Color: Churn Rate

### Page 3: Churn Analysis by Demographics
**Visualizations:**
1. **Churn by Age Group** (Bar Chart)
   - Senior Citizens vs Non-Senior Citizens
   - Show both count and percentage

2. **Churn by Contract Type** (Stacked Bar Chart)
   - X-axis: Contract Type (Month-to-month, One year, Two year)
   - Y-axis: Customer Count
   - Stack by: Churn Status

3. **Churn by Payment Method** (Horizontal Bar Chart)
   - Y-axis: Payment Methods
   - X-axis: Churn Rate %

4. **Family Status Impact** (Grouped Bar Chart)
   - Groups: Partner (Yes/No) and Dependents (Yes/No)
   - Y-axis: Churn Rate %

### Page 4: Service Usage and Churn
**Visualizations:**
1. **Internet Service Analysis** (Pie Charts)
   - Three pie charts for DSL, Fiber Optic, No Internet
   - Each showing churn vs retention

2. **Add-on Services Impact** (Heatmap)
   - Rows: Services (Online Security, Tech Support, Streaming, etc.)
   - Columns: Service Usage (Yes/No)
   - Values: Churn Rate %

3. **Service Bundle Analysis** (Tree Map)
   - Size: Customer Count
   - Color: Churn Rate
   - Hierarchy: Internet Service > Contract Type

### Page 5: At-Risk Customer Identification
**Visualizations:**
1. **Risk Score Distribution** (Histogram)
   - X-axis: Risk Score (0-5)
   - Y-axis: Number of Customers
   - Color by actual churn status for validation

2. **High-Risk Customers Table** (Table)
   - Columns: Customer ID, Risk Score, Monthly Charges, Tenure, Contract Type
   - Filter for top 100 highest risk customers
   - Conditional formatting for risk levels

3. **Predicted vs Actual Churn** (Confusion Matrix Visual)
   - Shows model performance
   - True Positives, False Positives, etc.

## DAX Measures to Create

### Basic Metrics
```dax
Total Customers = COUNTROWS(customers)

Churned Customers = CALCULATE(COUNTROWS(customers), customers[Churn] = 1)

Churn Rate = DIVIDE([Churned Customers], [Total Customers], 0)

Active Customers = CALCULATE(COUNTROWS(customers), customers[Churn] = 0)

Total Monthly Revenue = SUMX(customers, customers[MonthlyCharges])

Revenue at Risk = CALCULATE([Total Monthly Revenue], customers[Churn] = 1)

Avg Customer Lifetime Value = AVERAGE(customers[TotalCharges])
```

### Advanced Metrics
```dax
High Risk Customers = 
CALCULATE(
    COUNTROWS(customers),
    customers[Churn] = 0,
    customers[Contract] = "Month-to-month",
    customers[tenure] < 12
)

Retention Rate = 1 - [Churn Rate]

Monthly Revenue Loss = [Revenue at Risk]

Customer Acquisition Cost Impact = [Churned Customers] * 50  -- Assume $50 CAC
```

## Color Scheme
- **Primary**: #1f77b4 (Blue)
- **Secondary**: #ff7f0e (Orange)
- **Success**: #2ca02c (Green)
- **Warning**: #d62728 (Red)
- **Neutral**: #7f7f7f (Gray)

## Filters and Slicers
1. **Customer Segment** (Cluster)
2. **Contract Type**
3. **Internet Service Type**
4. **Tenure Range**
5. **Monthly Charges Range**
6. **Senior Citizen** (Yes/No)

## Key Insights to Highlight
1. Month-to-month contracts have significantly higher churn rates
2. Fiber optic customers churn more than DSL customers
3. Customers without online security are more likely to churn
4. New customers (< 12 months tenure) are at highest risk
5. Electronic check payment method correlates with higher churn

## Recommended Actions Panel
Include a text box with actionable recommendations:
- Target month-to-month customers with annual contract incentives
- Improve fiber optic service quality and support
- Promote online security add-ons
- Implement new customer onboarding programs
- Encourage automatic payment methods
