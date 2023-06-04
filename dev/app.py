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
def create_gradio_inputs(data):
    input_components = []
    for column in data.columns:
        if data[column].dtype == 'object' and len(data[column].unique()) > 3:
            input_components.append(gr.Dropdown(choices=list(data[column].unique()), label=column))
        elif data[column].dtype == 'object' and len(data[column].unique()) <= 3:
            input_components.append(gr.Radio(choices=list(data[column].unique()), label=column))
        elif data[column].dtype in ['int64', 'float64']:
            if data[column].min() == 1:
                input_components.append(gr.Slider(minimum=1, maximum=data[column].max(), step=1, label=column))
            else:
                input_components.append(gr.Slider(maximum=data[column].max(), step=0.5, label=column))
    return input_components

input_components = create_gradio_inputs(data)

output_components = [
    gr.Label(label="Churn Prediction"),
]

# Convert the input values to a pandas DataFrame with the appropriate column names
def input_df_creator(gender, SeniorCitizen, Partner, Dependents, tenure,
       PhoneService, InternetService, OnlineBackup, TechSupport,
       Contract, PaperlessBilling, PaymentMethod, MonthlyCharges,
       TotalCharges, StreamingService, SecurityService):
    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [SeniorCitizen],
        "Partner": [Partner],
        "Dependents": [Dependents],
        "tenure": [int(tenure)],
        "PhoneService": [PhoneService],
        "InternetService": [InternetService],
        "OnlineBackup": [OnlineBackup],
        "TechSupport": [TechSupport],
        "Contract": [Contract],
        "PaperlessBilling": [PaperlessBilling],
        "PaymentMethod": [PaymentMethod],
        "StreamingService": [StreamingService],
        "SecurityService": [SecurityService],
        "MonthlyCharges": [float(MonthlyCharges)],
        "TotalCharges": [float(TotalCharges)],
    }) 
    return input_data

# Define the function to be called when the Gradio app is run
def predict_churn(gender, SeniorCitizen, Partner, Dependents, tenure,
       PhoneService, InternetService, OnlineBackup, TechSupport,
       Contract, PaperlessBilling, PaymentMethod, MonthlyCharges,
       TotalCharges, StreamingService, SecurityService):
    input_df = input_df_creator(gender, SeniorCitizen, Partner, Dependents, tenure,
       PhoneService, InternetService, OnlineBackup, TechSupport,
       Contract, PaperlessBilling, PaymentMethod, MonthlyCharges,
       TotalCharges, StreamingService, SecurityService)
    
    # Encode categorical variables
    cat_cols = data.select_dtypes(include=['object']).columns
    cat_encoded = encoder.transform(input_df[cat_cols])

    # Scale numerical variables
    num_cols = data.select_dtypes(include=['int64', 'float64']).columns
    num_scaled = scaler.transform(input_df[num_cols])

    # joining encoded and scaled columns back together
    processed_df = pd.concat([num_scaled, cat_encoded], axis=1)

    # Make prediction
    prediction = model.predict(processed_df)
    return "Churn" if prediction[0] == 1 else "No Churn"

# Launch the Gradio app
iface = gr.Interface(predict_churn, inputs=input_components, outputs=output_components)
iface.launch(inbrowser= True, show_error= True)
