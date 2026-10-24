import mlflow
from load_db import read_data
from upload_db import upload_data_to_postgres

def load_model_from_best_run():
    # Get experiment
    experiment = mlflow.get_experiment_by_name(
        "Customer Churn Model"
    )

    # Retrieve runs and sort by accuracy
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.roc_auc DESC"]
    )

    run_id = runs.iloc[0]["run_id"]

    model_uri = f"runs:/{run_id}/xgboost_model"

    print(model_uri)
    loaded_model = mlflow.xgboost.load_model(model_uri)
    print("Loaded run:", run_id)

    return loaded_model

def predict_churn(model, input_data):
    columns = [
        'CreditScore',
        'Tenure',
        'Balance',
        'NumOfProducts',
        'IsActiveMember',
        'EstimatedSalary'
    ]

    input_data = input_data[columns]
    y_proba = model.predict_proba(input_data)[:, 1]
    y_pred = model.predict(input_data)
    return y_proba, y_pred

def prediction_result(df, y_proba, y_pred):
    print("\n\nUploading predictions to PostgreSQL...")
    df['predicted_churn_probability'] = y_proba
    df['predicted_churn'] = y_pred
    upload_data_to_postgres(df, table_name="predicted_churn")

    return "Predictions uploaded successfully to PostgreSQL!"

df = read_data('customer_data')
model = load_model_from_best_run()
y_proba, y_pred = predict_churn(model, df)
prediction_result(df, y_proba, y_pred)
# print("Predictions:", y_pred)