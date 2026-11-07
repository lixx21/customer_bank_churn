import streamlit as st
from load_db import read_data
from llm import llm_inference

def main(df, roc_data):
    st.set_page_config(
        page_title="Customer Churn"
    )

    st.title("Customer Churn Early Warning")

    total_customers = df['CustomerId'].nunique()
    churned_customers = df[df['predicted_churn'] == 1]['CustomerId'].nunique()
    df['predicted_churn_probability'] = df['predicted_churn_probability'] * 100
    churned_rate = churned_customers / total_customers * 100
    roc_auc = roc_data['AUC Score'].max()*100

    a, b,c = st.columns(3)
    
    a.metric("Total Customers", total_customers, border=True)
    b.metric("Churned Rate", f"{churned_rate:.1f}%", border=True)
    c.metric("Model ROC-AUC", f"{roc_auc:.1f}%", border=True)
    # d.metric("ROC-AUC", "0.91", border=True)

    st.subheader("Customers Risk of Churning")

    view = df
    # df.sort_values(by='predicted_churn_probability', ascending=False).head(10)

    event = st.dataframe(
        view[['Surname', 'Gender', 'Geography', 'predicted_churn_probability']],
        hide_index=True,
        selection_mode="single-row",
        column_config={
            "Surname": "Customer",
            "predicted_churn_probability": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%.1f%%"),
        },
        use_container_width=True,
        on_select="rerun"

    )
    # Get selected row
    if event.selection.rows:
        selected_index = event.selection.rows[0] # check selected row indices
        selected_customer = df.iloc[selected_index] # get the selected customer data based on the index
        with st.spinner("Loading customer analysis..."):
            llm_response = llm_inference(selected_customer)

        st.subheader("Customer Churn Analysis")
        st.write(f"**Customer:** {selected_customer['Surname']}")
        st.write(f"**Gender**: {selected_customer['Gender']}")
        st.write(f"**Age**: {selected_customer['Age']}")
        st.write(llm_response)


predicted_data = read_data('predicted_churn')
roc_auc = read_data('churn_model_evaluation_results')
if __name__== "__main__":
    main(predicted_data, roc_auc)