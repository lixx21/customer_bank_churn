import streamlit as st
from load_db import read_data
import mlflow

def main(df):
    st.set_page_config(
        page_title="Customer Churn"
    )

    st.title("Customer Churn Early Warning")

    col1, col2, col3, col4 = st.columns(4)
    total_customers = df['CustomerId'].nunique()
    churned_customers = df[df['predicted_churn'] == 1]['CustomerId'].nunique()
    df['predicted_churn_probability'] = df['predicted_churn_probability'] * 100
    churned_rate = churned_customers / total_customers * 100

    a, b = st.columns(2)
    # c, d = st.columns(2)
    a.metric("Total Customers", total_customers, border=True)
    b.metric("Churned Rate", f"{churned_rate:.1f}%", border=True)
    # c.metric("Model Accuracy", "87.2%", border=True)
    # d.metric("ROC-AUC", "0.91", border=True)

    st.subheader("Customers Risk of Churning")

    view = df
    # df.sort_values(by='predicted_churn_probability', ascending=False).head(10)

    st.dataframe(
        view[['Surname', 'Gender', 'Geography', 'predicted_churn_probability']],
        hide_index=True,
        selection_mode="single-row",
        column_config={
            "Surname": "Customer",
            "predicted_churn_probability": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%.1f%%"),
        }

    )
data = read_data('predicted_churn')

if __name__== "__main__":
    main(data)