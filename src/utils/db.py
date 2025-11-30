import os
import sys
from sqlalchemy import create_engine

def get_db_uri():
    """
    Constructs the PostgreSQL connection URI from environment variables.
    Exits the application if critical secrets are missing.
    """
    user = os.getenv('POSTGRES_USER')
    password = os.getenv('POSTGRES_PASSWORD')
    host = os.getenv('POSTGRES_HOST')
    port = os.getenv('POSTGRES_PORT', '5432') 
    db = os.getenv('POSTGRES_DB')

    # Security Gate: Fail fast if secrets are missing
    if not all([user, password, host, db]):
        print(" CRITICAL ERROR: Database credentials missing from environment.")
        print("   -> Check your .env file or Docker configuration.")
        sys.exit(1)

    return f"postgresql://{user}:{password}@{host}:{port}/{db}"

def get_db_engine():
    """Returns a SQLAlchemy engine instance."""
    uri = get_db_uri()
    try:
        engine = create_engine(uri)
        return engine
    except Exception as e:
        print(f" Database Connection Failed: {e}")
        sys.exit(1)