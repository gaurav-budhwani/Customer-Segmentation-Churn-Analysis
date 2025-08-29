-- Aggregated Analysis Queries
-- Create summary tables for Power BI dashboard

-- Monthly churn trends (if date data available, otherwise by tenure cohorts)
CREATE VIEW monthly_churn_trends AS
SELECT 
    CASE 
        WHEN tenure <= 6 THEN '0-6 months'
        WHEN tenure <= 12 THEN '7-12 months'
        WHEN tenure <= 24 THEN '13-24 months'
        WHEN tenure <= 36 THEN '25-36 months'
        WHEN tenure <= 48 THEN '37-48 months'
        ELSE '48+ months'
    END as tenure_cohort,
    COUNT(*) as total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned_customers,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as churn_rate_pct,
    AVG(CAST(MonthlyCharges AS FLOAT)) as avg_monthly_charges,
    AVG(CAST(TotalCharges AS FLOAT)) as avg_total_charges
FROM customers
WHERE TotalCharges != ' '
GROUP BY 
    CASE 
        WHEN tenure <= 6 THEN '0-6 months'
        WHEN tenure <= 12 THEN '7-12 months'
        WHEN tenure <= 24 THEN '13-24 months'
        WHEN tenure <= 36 THEN '25-36 months'
        WHEN tenure <= 48 THEN '37-48 months'
        ELSE '48+ months'
    END
ORDER BY MIN(tenure);

-- Service adoption and churn correlation
CREATE VIEW service_churn_analysis AS
SELECT 
    'PhoneService' as service_type,
    PhoneService as service_value,
    COUNT(*) as customer_count,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned_count,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as churn_rate_pct
FROM customers
GROUP BY PhoneService

UNION ALL

SELECT 
    'InternetService' as service_type,
    InternetService as service_value,
    COUNT(*) as customer_count,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned_count,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as churn_rate_pct
FROM customers
GROUP BY InternetService

UNION ALL

SELECT 
    'Contract' as service_type,
    Contract as service_value,
    COUNT(*) as customer_count,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned_count,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as churn_rate_pct
FROM customers
GROUP BY Contract

UNION ALL

SELECT 
    'PaymentMethod' as service_type,
    PaymentMethod as service_value,
    COUNT(*) as customer_count,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned_count,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as churn_rate_pct
FROM customers
GROUP BY PaymentMethod;

-- Revenue impact analysis
CREATE VIEW revenue_impact_analysis AS
SELECT 
    Churn,
    COUNT(*) as customer_count,
    SUM(CAST(MonthlyCharges AS FLOAT)) as total_monthly_revenue,
    AVG(CAST(MonthlyCharges AS FLOAT)) as avg_monthly_charges,
    SUM(CAST(TotalCharges AS FLOAT)) as total_lifetime_revenue,
    AVG(CAST(TotalCharges AS FLOAT)) as avg_lifetime_value
FROM customers
WHERE TotalCharges != ' '
GROUP BY Churn;

-- High-risk customer identification
CREATE VIEW high_risk_customers AS
SELECT 
    customerID,
    tenure,
    Contract,
    PaymentMethod,
    MonthlyCharges,
    TotalCharges,
    InternetService,
    -- Risk factors
    CASE WHEN Contract = 'Month-to-month' THEN 1 ELSE 0 END as month_to_month_risk,
    CASE WHEN PaymentMethod = 'Electronic check' THEN 1 ELSE 0 END as payment_risk,
    CASE WHEN tenure < 12 THEN 1 ELSE 0 END as tenure_risk,
    CASE WHEN CAST(MonthlyCharges AS FLOAT) > 70 THEN 1 ELSE 0 END as high_charges_risk,
    CASE WHEN OnlineSecurity = 'No' AND InternetService != 'No' THEN 1 ELSE 0 END as security_risk,
    
    -- Calculate risk score
    (CASE WHEN Contract = 'Month-to-month' THEN 1 ELSE 0 END +
     CASE WHEN PaymentMethod = 'Electronic check' THEN 1 ELSE 0 END +
     CASE WHEN tenure < 12 THEN 1 ELSE 0 END +
     CASE WHEN CAST(MonthlyCharges AS FLOAT) > 70 THEN 1 ELSE 0 END +
     CASE WHEN OnlineSecurity = 'No' AND InternetService != 'No' THEN 1 ELSE 0 END) as risk_score
FROM customers
WHERE Churn = 'No'  -- Only current customers
ORDER BY risk_score DESC, MonthlyCharges DESC;
