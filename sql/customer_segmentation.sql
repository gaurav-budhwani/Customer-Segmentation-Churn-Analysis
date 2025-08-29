-- Customer Segmentation Analysis
-- Create aggregated tables for customer segments

-- Customer segments by tenure and spending
CREATE VIEW customer_segments AS
SELECT 
    customerID,
    CASE 
        WHEN tenure <= 12 THEN 'New (0-12 months)'
        WHEN tenure <= 24 THEN 'Growing (13-24 months)'
        WHEN tenure <= 48 THEN 'Established (25-48 months)'
        ELSE 'Loyal (48+ months)'
    END as tenure_segment,
    
    CASE 
        WHEN CAST(MonthlyCharges AS FLOAT) < 35 THEN 'Low Spender'
        WHEN CAST(MonthlyCharges AS FLOAT) < 65 THEN 'Medium Spender'
        ELSE 'High Spender'
    END as spending_segment,
    
    CASE 
        WHEN SeniorCitizen = 1 THEN 'Senior'
        ELSE 'Non-Senior'
    END as age_segment,
    
    Contract,
    InternetService,
    Churn,
    tenure,
    CAST(MonthlyCharges AS FLOAT) as monthly_charges,
    CAST(TotalCharges AS FLOAT) as total_charges
FROM customers
WHERE TotalCharges != ' ';

-- Segment analysis with churn rates
SELECT 
    tenure_segment,
    spending_segment,
    age_segment,
    COUNT(*) as total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned_customers,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as churn_rate_pct,
    AVG(monthly_charges) as avg_monthly_charges,
    AVG(total_charges) as avg_total_charges,
    AVG(tenure) as avg_tenure
FROM customer_segments
GROUP BY tenure_segment, spending_segment, age_segment
ORDER BY churn_rate_pct DESC;

-- Service usage patterns by segment
SELECT 
    tenure_segment,
    spending_segment,
    COUNT(*) as customers,
    SUM(CASE WHEN PhoneService = 'Yes' THEN 1 ELSE 0 END) as phone_users,
    SUM(CASE WHEN InternetService != 'No' THEN 1 ELSE 0 END) as internet_users,
    SUM(CASE WHEN StreamingTV = 'Yes' THEN 1 ELSE 0 END) as streaming_tv_users,
    SUM(CASE WHEN StreamingMovies = 'Yes' THEN 1 ELSE 0 END) as streaming_movie_users,
    SUM(CASE WHEN TechSupport = 'Yes' THEN 1 ELSE 0 END) as tech_support_users,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as churn_rate_pct
FROM customer_segments cs
JOIN customers c ON cs.customerID = c.customerID
GROUP BY tenure_segment, spending_segment
ORDER BY churn_rate_pct DESC;
