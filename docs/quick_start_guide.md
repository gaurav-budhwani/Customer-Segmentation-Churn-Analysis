# Quick Start Guide - Customer Churn Analysis

## Prerequisites
- Python 3.8+ installed
- Power BI Desktop (for dashboard creation)
- Basic familiarity with Jupyter notebooks

## Step-by-Step Setup

### 1. Environment Setup
```bash
# Navigate to project directory
cd /path/to/churn

# Install required packages
pip install -r requirements.txt

# Create Jupyter kernel (optional)
python -m ipykernel install --user --name churn-analysis
```

### 2. Data Preparation
```bash
# Download/create sample dataset
python src/download_data.py
```

This will either:
- Use existing Kaggle dataset if available in `data/` folder
- Create a realistic sample dataset for demonstration

### 3. Run Complete Analysis
```bash
# Execute full analysis pipeline
python src/main_analysis.py
```

This script will:
- Load and preprocess data
- Perform exploratory data analysis
- Train machine learning models
- Execute customer segmentation
- Export results for Power BI

### 4. Jupyter Notebook Analysis
```bash
# Start Jupyter notebook
jupyter notebook notebooks/churn_analysis.ipynb
```

Run all cells to see:
- Detailed EDA visualizations
- Model training and evaluation
- Feature importance analysis
- Customer segmentation results

### 5. SQL Analysis (Optional)
If you have a SQL database:
```sql
-- Load data into your database
-- Then run queries from sql/ folder

-- Basic demographics
\i sql/customer_demographics.sql

-- Segmentation analysis
\i sql/customer_segmentation.sql

-- Aggregated insights
\i sql/aggregated_analysis.sql
```

### 6. Power BI Dashboard Creation

1. **Open Power BI Desktop**

2. **Import Data Sources:**
   - `results/customers_with_clusters.csv`
   - `results/cluster_analysis.csv`
   - `results/high_risk_customers.csv`
   - `results/spending_segment_analysis.csv`

3. **Follow Dashboard Setup:**
   - Open `powerbi/dashboard_setup.md`
   - Follow the detailed instructions for each page
   - Implement the recommended DAX measures
   - Apply the suggested color scheme and filters

4. **Create Dashboard Pages:**
   - Executive Summary
   - Customer Segmentation Analysis
   - Churn Analysis by Demographics
   - Service Usage and Churn
   - At-Risk Customer Identification

## Expected Outputs

### Files Generated
```
results/
├── customers_with_clusters.csv      # Customer data with segments
├── cluster_analysis.csv             # Segment statistics
├── high_risk_customers.csv          # Top 500 at-risk customers
├── spending_segment_analysis.csv    # Spending behavior analysis
├── contract_analysis.csv            # Contract type analysis
├── feature_importance.csv           # ML model feature rankings
├── key_insights.txt                 # Summary of findings
└── models/                          # Trained ML models
    ├── logistic_regression_model.pkl
    ├── random_forest_model.pkl
    ├── scalers.pkl
    └── encoders.pkl
```

### Key Insights You'll Discover
- Which customer segments have highest churn risk
- Most important factors driving customer churn
- Revenue impact of customer churn by segment
- Specific customers requiring immediate retention focus

## Troubleshooting

### Common Issues

**1. Missing Data File**
```bash
# If Kaggle dataset not available, the script creates sample data
# To use real Kaggle data:
# 1. Download from: https://www.kaggle.com/blastchar/telco-customer-churn
# 2. Place WA_Fn-UseC_-Telco-Customer-Churn.csv in data/ folder
```

**2. Package Installation Issues**
```bash
# Create virtual environment (recommended)
python -m venv churn_env
source churn_env/bin/activate  # On Windows: churn_env\Scripts\activate
pip install -r requirements.txt
```

**3. Jupyter Notebook Issues**
```bash
# Install Jupyter if not available
pip install jupyter

# Start notebook server
jupyter notebook
```

**4. Power BI Data Import Issues**
- Ensure CSV files are in UTF-8 encoding
- Check that file paths are correct
- Verify data types are properly detected

## Next Steps After Setup

1. **Review Analysis Results**
   - Check `results/key_insights.txt` for summary
   - Review model performance metrics
   - Examine customer segmentation results

2. **Customize for Your Business**
   - Adjust risk scoring criteria
   - Modify customer segments based on business needs
   - Update retention strategy recommendations

3. **Deploy Insights**
   - Share Power BI dashboard with stakeholders
   - Implement retention campaigns for high-risk customers
   - Monitor churn rate improvements

4. **Continuous Improvement**
   - Retrain models with new data monthly
   - Update customer segments quarterly
   - Refine retention strategies based on results

## Support
For questions or issues:
- Review the detailed documentation in `docs/analysis_report.md`
- Check the Power BI setup guide in `powerbi/dashboard_setup.md`
- Examine the code comments in the Python modules
