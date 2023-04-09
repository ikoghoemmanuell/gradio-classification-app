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


# separate categorical and numeric columns
categoric_columns = data.select_dtypes(include=['object']).columns.tolist()
numeric_columns = data.select_dtypes(include=['int', 'float']).columns.tolist()

# Define the input and output interfaces for the Gradio app
input_components = []

# Create Gradio inputs for each column
for col in data.columns:
    if col in categoric_columns:
        categories = data[col].unique()
        if len(categories) < 4:
            input = gr.CheckboxGroup(choices=list(categories), label=col)
        else:
            input = gr.Dropdown(choices=list(categories), label=col)
    elif col in numeric_columns:
        min_val = data[col].min()
        max_val = data[col].max()
        if min_val-max_val < 30:
            input = gr.Slider(minimum=min_val, maximum=max_val, label=col)
        else:
            input = gr.Number(label=col)
    input_components.append(input)

output_components = [
    gr.Label(label="Churn Prediction"),
]

# Convert the input values to a pandas DataFrame with the appropriate column names
def input_df_creator():
    input_labels = [input.label for input in input_components]
    input_values = [input.value for input in input_components]
    input_data = pd.DataFrame([input_values], columns=input_labels)
    return input_data

# Define the function to be called when the Gradio app is run
def predict_churn():
    input_df = input_df_creator()
    
    # Cast any int64 values to int32
    input_df = input_df.astype({'column_name': 'int32'})
    
    # Scale numeric columns and encode categorical columns
    scaled_num = scaler.fit_transform(input_df[numeric_columns])
    encoded_cat = encoder.transform(input_df[categoric_columns])
    input_data = pd.concat([scaled_num, encoded_cat], axis=1)
    
    # Use the pre-trained model to make a prediction
    prediction = model.predict(input_data)[0]
    
    # Convert the prediction to a human-readable format
    if prediction == 1:
        churn_prediction = "This customer is likely to churn."
    else:
        churn_prediction = "This customer is not likely to churn."
    
    return churn_prediction

# Create the Gradio interface
gr.Interface(predict_churn, inputs=input_components, outputs=output_components, title="Churn Prediction", description="Enter the customer's demographic and service usage information to predict whether they are likely to churn or not.").launch()