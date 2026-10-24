import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)
from scipy.stats import randint, uniform, loguniform
from sklearn.model_selection import RandomizedSearchCV
# from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
from upload_db import upload_data_to_postgres
from load_db import read_data
import mlflow
import mlflow.xgboost


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

def train_model(baseline_params, X_train, y_train):

    xgb_model = XGBClassifier(
        **baseline_params
    )

    #Define the search space
    param_distributions = {
        "n_estimators": randint(100, 1001),
        "max_depth": randint(3, 11),
        "learning_rate": loguniform(0.01, 0.3),
        "min_child_weight": loguniform(1, 20),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "gamma": uniform(0, 5),
        "reg_alpha": loguniform(1e-4, 10),
        "reg_lambda": loguniform(0.1, 10)
    }

    # Configure automatic tuning
    search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_distributions,
        n_iter=50,             # Number of parameter combinations
        scoring="roc_auc",
        cv=3,                   # 3-fold cross-validation
        random_state=42,
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    # Run tuning
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    best_model.fit(X_train, y_train)

    return best_model, search.best_params_, search.best_score_

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
    
def log_model_to_mlflow(model, metrics, params, best_score):
    print('logging experiment to MLflow...')
    mlflow.set_experiment("Customer Churn Model")
    # Track experiment
    with mlflow.start_run(run_name="xgboost_baseline") as run:

        mlflow.log_params(params)
        mlflow.log_param("n_iter", 50)
        mlflow.log_param("cv_folds", 3)

        mlflow.log_metric("best_cv_roc_auc", best_score)
        mlflow.log_metrics(metrics)

        mlflow.set_tag("model_type", "XGBoost")
        mlflow.set_tag("dataset", "customeer_churn")

        # Save model to MLflow
        model_info = mlflow.xgboost.log_model(
            xgb_model=model,
            name="xgboost_model"
        )

    return model_info

baseline_params = {
    "objective":"binary:logistic",
    "eval_metric":"auc",
    "random_state":42,
    "n_jobs":-1
    }

print("Reading data from PostgreSQL...")
df = read_data('customer_data')

print("\nreprocessing data...")
df_features, df_target = preprocess_data(df)

print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = split_data(df_features, df_target)

print("\nTraining the model...")
trained_model, best_params, best_score = train_model(baseline_params,X_train, y_train)

print("Evaluating the model...")
evaluation, metrics = evaluate_model(trained_model, X_test, y_test)

print("\nLogging the model and metrics to MLflow...")
log_model_to_mlflow(trained_model, metrics, best_params, best_score)
