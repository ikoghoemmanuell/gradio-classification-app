import gradio as gr
import pandas as pd
import numpy as np
import pickle

# Load the model and encoder and scaler
model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Load the data
data = pd.read_csv('data.csv')

# Define the input and output interfaces for the Gradio app
input_components = [
    gr.Dropdown(choices=list(data['gender'].unique()), label="Gender"),
    gr.Radio(choices=list(data['Partner'].unique()), label="Partner"),
    gr.Radio(choices=list(data['Dependents'].unique()), label="Dependents"),
    gr.Radio(choices=['None', 'SingleLine', 'MultipleLines'], label="Phone Service"),
    gr.Radio(choices=["DSL", "Fiber optic", "No"], label="Internet Service"),
    gr.Radio(choices=["Yes", "No"], label="Online Backup"),
    gr.Radio(choices=["Yes", "No"], label="Tech Support"),
    gr.Dropdown(choices=["Month-to-month", "One year", "Two year"], label="Contract"),
    gr.Radio(choices=list(data['PaperlessBilling'].unique()), label="Paperless Billing"),
    gr.Dropdown(choices=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], label="Payment Method"),
    gr.Radio(choices=['FullService', 'TV', 'Movies'], label="Streaming Service"),
    gr.Radio(choices=['FullSecurity', 'OnlineSecurity', 'DeviceProtection'], label="Security Service"),
    gr.Radio(choices=list(data['SeniorCitizen'].unique()), label="Senior Citizen"),
    gr.Slider(minimum=1, maximum=data['tenure'].max(), step=1, label="Tenure (Month)"),
    gr.Slider(maximum=data['MonthlyCharges'].max(), step=0.5, label="Monthly Charges"),
    gr.Slider(maximum=data['TotalCharges'].max(), step=0.5, label="Total Charges")
]

output_components = [
    gr.Label(label="Churn Prediction"),
]

# Convert the input values to a pandas DataFrame with the appropriate column names
def input_df_creator(gender, partner, dependents, phone_service, internet_service, online_backup, tech_support, 
                     contract, paperless_billing, payment_method, streaming_service, security_service, senior_citizen, 
                     tenure, monthly_charges, total_charges):
    input_data = pd.DataFrame({
        "gender": [gender],
        "Partner": [partner],
        "Dependents": [dependents],
        "PhoneService": [phone_service],
        "InternetService": [internet_service],
        "OnlineBackup": [online_backup],
        "TechSupport": [tech_support],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "StreamingService": [streaming_service],
        "SecurityService": [security_service],
        "SeniorCitizen": [int(senior_citizen)],
        "tenure": [int(tenure)],
        "MonthlyCharges": [float(monthly_charges)],
        "TotalCharges": [float(total_charges)],
    }) 
    return input_data



# Define the function to be called when the Gradio app is run
def predict_churn(gender, partner, dependents, phone_service, internet_service, online_backup, tech_support, 
                     contract, paperless_billing, payment_method, streaming_service, security_service, senior_citizen, 
                     tenure, monthly_charges, total_charges):
    input_df = input_df_creator()
    # Encode categorical variables
    cat_cols = input_df.select_dtypes(include=['object']).columns
    print(input_df[cat_cols])
    print(len(input_df[cat_cols].columns))
    print(len(input_df[cat_cols].keys()))
    print(len(input_df[cat_cols].values))
    input_df[cat_cols] = encoder.transform(input_df[cat_cols].fillna('NaN'))

    # Scale numerical variables
    num_cols = input_df.select_dtypes(include=['int64', 'float64']).columns
    input_df[num_cols] = scaler.transform(input_df[num_cols])
    # Make prediction
    prediction = model.predict(input_df)
    return "Churn" if prediction[0] == 1 else "No Churn"

# Launch the Gradio app
iface = gr.Interface(predict_churn, inputs=input_components, outputs=output_components)
iface.launch()