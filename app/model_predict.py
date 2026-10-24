import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)
# from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
from upload_db import upload_data_to_postgres
import mlflow
import mlflow.xgboost

load_dotenv()

def read_data():
    # Connect to PostgreSQL
    DB_NAME = os.getenv("DB_NAME")
    DB_SCHEMA = os.getenv("DB_SCHEMA")
    DB_TABLE_NAME = os.getenv("DB_TABLE_NAME")
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

def preprocess_data(df):
    # Preprocess the data
    columns = [
        'CreditScore',
        'Tenure',
        'Balance',
        'NumOfProducts',
        'IsActiveMember',
        'EstimatedSalary',
        'Exited'
    ]

    df.dropna(inplace=True)
    new_df = df[columns].drop_duplicates()
    df_features = new_df.drop('Exited', axis=1)
    df_target = new_df['Exited']

    return df_features, df_target   

def split_data(df_features, df_target):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df_features, df_target, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test

def train_model(params, X_train, y_train):
    xgb_model = XGBClassifier(
        **params
    )
    xgb_model.fit(X_train, y_train)

    return xgb_model

# Evaluation
def evaluate_model(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc_score = roc_auc_score(y_test, y_proba)
    # report = classification_report(y_test, y_pred)
    # cm = confusion_matrix(y_test, y_pred)
    evaluate_data = []   
    evaluate_data.append({
        "Model": 'XGBoost',
        "AUC Score": auc_score
    }) 
    # print(f"AUC Score: {auc_score}")
    # print("Evaluation results:")
    df = pd.DataFrame(evaluate_data).sort_values(by="AUC Score", ascending=False)
    df["load_date"] = pd.to_datetime('today').normalize()

    print("Uploading evaluation results to PostgreSQL...")
    upload_data_to_postgres(df, "churn_model_evaluation_results")

    print("Successfully evaluated the model and uploaded the results to PostgreSQL")

    metrics = {
        "roc_auc": auc_score,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0)
    }

    evaluation =   {
        "AUC Score": auc_score
    }

    return evaluation, metrics
    


xgb_params = {
    "n_estimators":500,
    "learning_rate":0.05,
    "max_depth":6,
    "subsample":0.8,
    "colsample_bytree":0.8,
    "objective":"binary:logistic",
    "eval_metric":"auc",
    "random_state":42,
    "n_jobs":-1
    }

print("Reading data from PostgreSQL...")
df = read_data()

print("Preprocessing data...")
df_features, df_target = preprocess_data(df)

print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = split_data(df_features, df_target)

print("Training the model...")
trained_model = train_model(xgb_params,X_train, y_train)

print("Evaluating the model...")
evaluation, metrics = evaluate_model(trained_model, X_test, y_test)

## MLFLOW
# Create an experiment
mlflow.set_experiment("Customer Churn Model")

# Track experiment
with mlflow.start_run(run_name="xgboost_baseline"):

    mlflow.log_params(xgb_params)
    mlflow.log_metrics(metrics)

    mlflow.set_tag("model_type", "XGBoost")
    mlflow.set_tag("dataset", "customeer_churn")

    # Save model to MLflow
    model_info = mlflow.xgboost.log_model(
        xgb_model=trained_model,
        name="xgboost_model"
    )

    print("Metrics:", metrics)
    print("Model URI:", model_info.model_uri)