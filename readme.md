# Customer Churn Early-Warning System

An end-to-end machine learning project that predicts which bank customers are likely to leave the bank within the next 90 days, so a retention team can reach out to the right customers first.

The pipeline covers data ingestion, model training with experiment tracking, a results database, and an interactive dashboard, all containerized with Docker.

# Business problem

A retention team cannot call or message every customer. If a customer quietly becomes inactive and moves their money elsewhere, the bank usually finds out too late.

**Question:** which customers are most at risk of leaving in the next 90 days, and what is driving that risk?

- Target: 1 if the customer closes their accounts or has no transactions in the 90 days after a snapshot date, otherwise 0.
- Business metric: share of true leavers (and balance at risk) captured when the team contacts only the top 10% highest-risk customers.

# Pipeline Flow:
![pipeline](image/pipeline.png)

# Tech stack
Language = Python 3.11
Database = PostgreSQL 16
ML = scikit-learn, XGBoost
Experiment tracking	= MLflow
UI = Streamlit
Containers = Docker, Docker Compose
Data handling = pandas, SQLAlchemy, psycopg2

# Quick Start

```
# 1. Clone the repository
git clone
cd customer_bank_churn

# 2. Build and start everything
docker compose up --build
```

Then open:

- Dashboard: http://localhost:8501
- MLflow UI: http://localhost:5000

# App Overview
![app](image/app.png)