import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
import os

# Airflow Imports
from airflow import DAG
from airflow.decorators import task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.operators.bash import BashOperator  
from sqlalchemy import text  

# --- CONFIGURATION ---
DATA_PATH = "/opt/airflow/data/Telco-Customer-Churn.csv"
SQL_INIT_PATH = "sql/init_db.sql"  
DB_CONN_ID = "telco_db"

# --- HELPER: Get Database Engine ---
def get_sqlalchemy_engine():
    hook = PostgresHook(postgres_conn_id=DB_CONN_ID)
    return hook.get_sqlalchemy_engine()

# --- DEFINE THE DAG ---
default_args = {
    'owner': 'data_engineer',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='telco_churn_prediction_pipeline',
    default_args=default_args,
    start_date=datetime(2023, 1, 1),
    schedule_interval='@daily',
    catchup=False,
    template_searchpath=['/opt/airflow/include'],
    tags=['telco', 'churn', 'production'],
) as dag:

    # TASK 1: Initialize Database
    init_db = PostgresOperator(
        task_id='initialize_schema',
        postgres_conn_id=DB_CONN_ID,
        sql=SQL_INIT_PATH
    )

    # TASK 2: ETL - Ingest Raw Data
    @task
    def extract_and_load_data():
        print(f"Reading data from {DATA_PATH}...")
        
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"File not found at {DATA_PATH}")
        
        df = pd.read_csv(DATA_PATH)
        
        # Basic Cleaning
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
        
        # Data Quality Gate
        if df['customerID'].duplicated().any():
            raise ValueError("Data Quality Failed: Duplicate Customer IDs found!")
        
        # Prepare Tables
        customers = df[['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents']].copy()
        customers.columns = ['customer_id', 'gender', 'senior_citizen', 'partner', 'dependents']
        
        # Updated Services (Mapped to lowercase to match SQL)
        services = df[['customerID', 'PhoneService', 'MultipleLines', 'InternetService', 
                       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']].copy()
        services.columns = ['customer_id', 'phoneservice', 'multiplelines', 'internetservice',
                            'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
                            'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling', 'paymentmethod']

        # Updated Billing (Included tenure)
        billing = df[['customerID', 'MonthlyCharges', 'TotalCharges', 'Churn', 'tenure']].copy()
        billing.columns = ['customer_id', 'monthly_charges', 'total_charges', 'churn_label', 'tenure']

        engine = get_sqlalchemy_engine()
        
        print("Loading tables to Data Warehouse...")
        # Use 'append' because 'initialize_schema' already recreated the empty tables
        customers.to_sql('customers', engine, if_exists='append', index=False)
        services.to_sql('services', engine, if_exists='append', index=False)
        billing.to_sql('billing', engine, if_exists='append', index=False)
        
        print("ETL Data Load Complete.")

    # TASK 3: Machine Learning (Clustering + XGBoost + Metrics)
    @task
    def train_and_predict():
        engine = get_sqlalchemy_engine()
        
        # 1. Fetch Data
        query = """
        SELECT c.customer_id, c.gender, c.senior_citizen, c.partner, c.dependents,
               s.phoneservice, s.multiplelines, s.internetservice, s.contract,
               b.monthly_charges, b.total_charges, b.churn_label, b.tenure
        FROM customers c
        JOIN services s ON c.customer_id = s.customer_id
        JOIN billing b ON c.customer_id = b.customer_id
        """
        df = pd.read_sql(query, engine)
        
        # 2. Preprocessing
        df['churn_target'] = df['churn_label'].apply(lambda x: 1 if x == 'Yes' else 0)
        X = df.drop(['customer_id', 'churn_label', 'churn_target'], axis=1)
        y = df['churn_target']
        
        # Encode
        X_encoded = pd.get_dummies(X, drop_first=True)
        
        # --- UNSUPERVISED LEARNING (Clustering) ---
        kmeans = KMeans(n_clusters=4, random_state=42)
        cluster_ids = kmeans.fit_predict(X_encoded)
        
        # 3. Train XGBoost Model
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)
        
        # 4. Generate Predictions
        predictions = model.predict_proba(X_encoded)[:, 1]
        pred_labels = model.predict(X_encoded)
        
        # 5. Save Predictions
        results_df = pd.DataFrame({
            'customer_id': df['customer_id'],
            'churn_probability': predictions,
            'predicted_churn': pred_labels,
            'cluster_id': cluster_ids,
            'risk_level': ['High Risk' if x > 0.7 else ('Medium Risk' if x > 0.4 else 'Low Risk') for x in predictions],
            'prediction_date': datetime.now()
        })
        
        print("Saving Predictions & Clusters to DB...")
        results_df.to_sql('predictions', engine, if_exists='replace', index=False)
        
        # --- SAVE FEATURE IMPORTANCE ---
        importance_df = pd.DataFrame({
            'Feature': X_encoded.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        print("Saving Feature Importance...")
        importance_df.to_sql('feature_importance', engine, if_exists='replace', index=False)
        
        # --- SAVE MODEL METRICS (Fixed for SQLAlchemy 1.4/2.0) ---
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM model_metrics"))
            # Auto-commit happens here
            
        metrics_df = pd.DataFrame([
            {'metric': 'last_updated', 'value': datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
            {'metric': 'accuracy', 'value': str(accuracy_score(y_test, model.predict(X_test)))}
        ])
        
        print("Saving System Metrics...")
        metrics_df.to_sql('model_metrics', engine, if_exists='append', index=False)
        print("Training Logic Complete.")

    # TASK 4: Transformation (dbt)
    dbt_run = BashOperator(
        task_id='dbt_transform',
        # FIX: Using the absolute path to the dbt executable
        bash_command='cd /opt/airflow/dbt_project && /home/airflow/.local/bin/dbt run --profiles-dir .',
        env={
            'DBT_HOST': 'postgres-dw',
            'DBT_USER': 'admin',
            'DBT_PASSWORD': 'password123'
        }
    )
    
    # TASK 5: Data Quality Tests (dbt)
    dbt_test = BashOperator(
        task_id='dbt_test',
        # FIX: Using the absolute path to the dbt executable
        bash_command='cd /opt/airflow/dbt_project && /home/airflow/.local/bin/dbt test --profiles-dir .',
        env={
            'DBT_HOST': 'postgres-dw',
            'DBT_USER': 'admin',
            'DBT_PASSWORD': 'password123'
        }
    )

    # --- EXECUTION FLOW ---
    init_db >> extract_and_load_data() >> train_and_predict() >> dbt_run >> dbt_test