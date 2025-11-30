{{ config(materialized='view', alias='analytics_master_view') }}

WITH customers AS (
    SELECT * FROM {{ source('telco_db', 'customers') }}
),

services AS (
    SELECT * FROM {{ source('telco_db', 'services') }}
),

billing AS (
    SELECT * FROM {{ source('telco_db', 'billing') }}
),

predictions AS (
    SELECT * FROM {{ source('telco_db', 'predictions') }}
)

SELECT 
    c.customer_id,
    c.gender,
    c.senior_citizen,
    c.partner,
    c.dependents,
    -- Service Details
    s.phoneservice,
    s.multiplelines,
    s.internetservice AS internet_service,
    s.onlinesecurity,
    s.onlinebackup,
    s.deviceprotection,
    s.techsupport,
    s.streamingtv,
    s.streamingmovies,
    s.contract,
    s.paperlessbilling,
    s.paymentmethod AS payment_method,
    -- Billing
    b.tenure,
    b.monthly_charges,
    b.total_charges,
    b.churn_label AS churn,
    -- Predictive
    p.churn_probability,
    p.risk_level,
    p.prediction_date,
    p.cluster_id,
    (b.monthly_charges * 12) AS "LTV_Projection"

FROM customers c
JOIN services s ON c.customer_id = s.customer_id
JOIN billing b ON c.customer_id = b.customer_id
LEFT JOIN predictions p ON c.customer_id = p.customer_id