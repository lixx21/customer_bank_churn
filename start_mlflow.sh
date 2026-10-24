
#!/bin/bash
set -e

echo "Checking MLflow database..."

if ! psql -h postgres -U postgres -d postgres -tAc \
    "SELECT 1 FROM pg_database WHERE datname = 'mlflow'" | grep -q 1; then
    psql -h postgres -U postgres -d postgres \
        -v ON_ERROR_STOP=1 -c "CREATE DATABASE mlflow;"
    echo "MLflow database created."
else
    echo "MLflow database already exists."
fi

echo "Starting MLflow server..."


exec mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --allowed-hosts "mlflow:5000,localhost:5000,127.0.0.1:5000" \
    --backend-store-uri "postgresql+psycopg://postgres:admin@postgres:5432/mlflow" \
    --artifacts-destination /mlartifacts
