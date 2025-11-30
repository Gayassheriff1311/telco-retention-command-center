-- ==========================================
-- 1. INFRASTRUCTURE CLEANUP
-- ==========================================
DROP VIEW IF EXISTS analytics_master_view;
DROP TABLE IF EXISTS predictions CASCADE;
DROP TABLE IF EXISTS model_metrics CASCADE;
DROP TABLE IF EXISTS feature_importance CASCADE;
DROP TABLE IF EXISTS billing CASCADE;
DROP TABLE IF EXISTS services CASCADE;
DROP TABLE IF EXISTS customers CASCADE;

-- ==========================================
-- 2. CORE DATA WAREHOUSE
-- ==========================================

-- A. Customers Table
CREATE TABLE customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    gender VARCHAR(10),
    senior_citizen INT,
    partner VARCHAR(5),
    dependents VARCHAR(5),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- B. Services Table (Matches Python: lowercase, includes Contract/Payment)
CREATE TABLE services (
    customer_id VARCHAR(50) REFERENCES customers(customer_id),
    phoneservice VARCHAR(5),       
    multiplelines VARCHAR(20),
    internetservice VARCHAR(20),
    onlinesecurity VARCHAR(20),
    onlinebackup VARCHAR(20),
    deviceprotection VARCHAR(20),
    techsupport VARCHAR(20),
    streamingtv VARCHAR(20),
    streamingmovies VARCHAR(20),
    contract VARCHAR(20),          -- Python sends this here
    paperlessbilling VARCHAR(5),
    paymentmethod VARCHAR(30),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- C. Billing Table (Matches Python: churn_label)
CREATE TABLE billing (
    customer_id VARCHAR(50) REFERENCES customers(customer_id),
    monthly_charges FLOAT,
    total_charges FLOAT,
    churn_label VARCHAR(5),        
    tenure INT,                    
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ==========================================
-- 3. INTELLIGENCE LAYER
-- ==========================================
CREATE TABLE predictions (
    customer_id VARCHAR(50) REFERENCES customers(customer_id),
    churn_probability FLOAT,
    predicted_churn INT,
    risk_level VARCHAR(20),
    cluster_id INT,               
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Stores model performance metrics (Accuracy, AUC) & System Logs
CREATE TABLE model_metrics (
    metric_id SERIAL PRIMARY KEY,
    metric VARCHAR(50),
    value VARCHAR(50),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Stores Global Feature Importance (Explainable AI)
CREATE TABLE feature_importance (
    feat_id SERIAL PRIMARY KEY,
    "Feature" VARCHAR(100),
    "Importance" FLOAT
);