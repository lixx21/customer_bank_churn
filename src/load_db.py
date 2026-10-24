from dotenv import load_dotenv
import os
import pandas as pd
from sqlalchemy import create_engine

load_dotenv()
    
def read_data(table_name):
    # Connect to PostgreSQL
    DB_NAME = os.getenv("DB_NAME")
    DB_SCHEMA = os.getenv("DB_SCHEMA")
    DB_TABLE_NAME = table_name
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = int(os.getenv("DB_PORT", 5432))
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_USER = os.getenv("DB_USER")

    db_engine = create_engine(
        f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    
    df = pd.read_sql(f"SELECT * FROM {DB_SCHEMA}.{DB_TABLE_NAME};", db_engine)
    db_engine.dispose()

    return df