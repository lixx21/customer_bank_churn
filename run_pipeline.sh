#!/bin/sh
set -e

echo "Step 1: Upload data to PostgreSQL..."
python src/upload_db.py

echo "Step 2: Train model..."
python src/train.py

echo "Step 3: Generate predictions..."
python src/predict.py

echo "Pipeline completed successfully!"