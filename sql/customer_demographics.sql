-- Customer Demographics Analysis
-- Query customer demographics, subscription type, and usage behavior

-- Basic customer demographics with churn status
SELECT 
    customerID,
    gender,
    SeniorCitizen,
    Partner,
    Dependents,
    tenure,
    PhoneService,
    MultipleLines,
    InternetService,
    OnlineSecurity,
    OnlineBackup,
    DeviceProtection,
    TechSupport,
    StreamingTV,
    StreamingMovies,
    Contract,
    PaperlessBilling,
    PaymentMethod,
    MonthlyCharges,
    TotalCharges,
    Churn
FROM customers;

-- Customer demographics summary by churn status
SELECT 
    Churn,
    COUNT(*) as customer_count,
    AVG(CAST(tenure AS FLOAT)) as avg_tenure,
    AVG(CAST(MonthlyCharges AS FLOAT)) as avg_monthly_charges,
    AVG(CAST(TotalCharges AS FLOAT)) as avg_total_charges,
    COUNT(CASE WHEN gender = 'Male' THEN 1 END) as male_count,
    COUNT(CASE WHEN gender = 'Female' THEN 1 END) as female_count,
    COUNT(CASE WHEN SeniorCitizen = 1 THEN 1 END) as senior_count,
    COUNT(CASE WHEN Partner = 'Yes' THEN 1 END) as with_partner_count,
    COUNT(CASE WHEN Dependents = 'Yes' THEN 1 END) as with_dependents_count
FROM customers
GROUP BY Churn;

-- Contract type analysis
SELECT 
    Contract,
    Churn,
    COUNT(*) as customer_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY Contract), 2) as churn_rate_pct
FROM customers
GROUP BY Contract, Churn
ORDER BY Contract, Churn;

-- Payment method analysis
SELECT 
    PaymentMethod,
    Churn,
    COUNT(*) as customer_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY PaymentMethod), 2) as churn_rate_pct
FROM customers
GROUP BY PaymentMethod, Churn
ORDER BY PaymentMethod, Churn;
