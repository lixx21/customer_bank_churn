import ollama

def llm_inference(customer):

    columns = [
        'Surname',
        'CreditScore',
        'Tenure',
        'Balance',
        'Age',
        'Geography',
        'NumOfProducts',
        'IsActiveMember',
        'EstimatedSalary',
        'predicted_churn_probability'
    ]

    prompt = f"""
    Context: 
    You are a senior data analyst. You have been given a dataset containing customer information and their predicted churn status. 
    Your task is to analyze the dataset and provide insights on customer churn patterns, 
    including any significant features that may contribute to churn, and suggest potential strategies to reduce churn based on the analysis.

    Customer Data:
    {customer[columns]} 

    Format Output:
    # why the customer is likely to churn
    # how to prevent churn for this customer

    Important Notes: do not provide any code in your response. Only provide analysis and recommendations based on the customer data provided.
    """
    messages = [
        {
            "role": "user",
            "content": prompt,
        },
    ]
    client = ollama.Client(
        host="http://ollama:11434"
    )
    response = client.chat(model="llama3.2", messages=messages)
    return response.message.content