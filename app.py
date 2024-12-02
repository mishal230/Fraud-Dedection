import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from transformers import pipeline
import gradio as gr

# Load the creditcard.csv dataset from your local directory
file_path = 'creditcard.csv'  

# Load the dataset
df = pd.read_csv(file_path)

# Display basic information
print("Columns in the dataset:", df.columns)
print(df.head())

# Preprocessing: Selecting relevant columns
time_col = 'Time'
amount_col = 'Amount'
class_col = 'Class'
feature_cols = [col for col in df.columns if col not in [class_col, time_col]]

# Handle missing values
df = df.fillna(df.mean())

# Downsample the majority class to handle class imbalance
df_majority = df[df[class_col] == 0]
df_minority = df[df[class_col] == 1]
df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(df_minority))
df_balanced = pd.concat([df_majority_downsampled, df_minority])

# Feature scaling
X = df_balanced[feature_cols]
y = df_balanced[class_col]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Balancing the dataset using SMOTE
smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)

# Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Initialize the retrieval pipeline with a lightweight model 
retrieval_pipeline = pipeline("feature-extraction", model="distilbert-base-uncased")

def retrieve_explanation(prediction):
    if prediction == 1:
        return "The transaction is classified as fraudulent based on the provided features."
    return "The transaction is classified as non-fraudulent based on the provided features."

# Gradio prediction function
def fraud_detection_predictor(V1, V2, V3, Amount):
    # Prepare input features
    input_features = [0] * len(feature_cols)
    input_features[feature_cols.index('V1')] = V1
    input_features[feature_cols.index('V2')] = V2
    input_features[feature_cols.index('V3')] = V3
    input_features[feature_cols.index('Amount')] = Amount

    # Scale input data
    input_data = scaler.transform([input_features])
    
    # Make a prediction
    prediction = model.predict(input_data)[0]
    fraud_status = "Fraudulent" if prediction == 1 else "Non-Fraudulent"
    explanation = retrieve_explanation(prediction)
    return fraud_status, explanation

# Define Gradio Interface
interface = gr.Interface(
    fn=fraud_detection_predictor,
    inputs=[
        gr.Number(label="V1"),
        gr.Number(label="V2"),
        gr.Number(label="V3"),
        gr.Number(label="Amount")
    ],
    outputs=[
        gr.Textbox(label="Fraud Status"),
        gr.Textbox(label="Explanation")
    ],
    title="Credit Card Fraud Detection",
    description="Enter transaction features (V1, V2, V3, Amount) to predict fraud status."
)

# Launch Gradio Interface
interface.launch()
