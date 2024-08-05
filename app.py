import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
from torchvision import models
import requests
from io import BytesIO

# Define the model class as it was in the training script
class RetinalModel(nn.Module):
    def __init__(self, num_parameters):
        super(RetinalModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_parameters)
    
    def forward(self, x):
        return self.resnet(x)

# Function to download the model from Google Drive
def download_model(model_url):
    # Convert the Google Drive URL to a direct download link
    file_id = model_url.split('/')[-2]
    direct_link = f"https://drive.google.com/uc?id={file_id}"

    response = requests.get(direct_link)
    if response.status_code == 200:
        return BytesIO(response.content)
    else:
        st.error(f"Failed to download the model: {response.status_code}")
        return None

# Load the model
def load_model(model_url, num_parameters):
    model = RetinalModel(num_parameters)
    model_data = download_model(model_url)
    
    if model_data:
        try:
            model.load_state_dict(torch.load(model_data, map_location=torch.device('cpu')))
            model.eval()
            return model
        except RuntimeError as e:
            st.error(f"Failed to load the model: {e}")
            return None
    return None

# Define the transformation for the input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define the healthy ranges for men and women
healthy_ranges = {
    'Male': {
        'Total Cholesterol': (125, 200),
        'LDL': (0, 100),
        'HDL': (40, 100),
        'Triglycerides': (0, 150),
        'Mean Arterial Blood Pressure': (70, 105),
        'eGFR': (90, 150),
        'Albumin': (3.5, 5.0),
        'Fasting Glucose Level': (70, 99),
        'Normal HbA1c': (0, 5.7),
        'Postprandial Glucose Level': (0, 140),
        'Sodium': (135, 145),
        'Potassium': (3.5, 5.0),
        'Red Blood Cells Count': (4.7, 6.1),
        'White Blood Cells Count': (4500, 11000),
        'Packed Cell Volume': (40.7, 50.3),
        'Magnesium': (1.7, 2.2),
        'Uric Acid': (3.5, 7.2),
        'C-Reactive Protein (CRP)': (0.1, 1.0),
        'Body Mass Index (BMI)': (18.5, 24.9),
        'Vitamin D': (20, 50),
        'Systolic Blood Pressure': (90, 120),
        'Diastolic Blood Pressure': (60, 80)
    },
    'Female': {
        'Total Cholesterol': (125, 200),
        'LDL': (0, 100),
        'HDL': (50, 100),
        'Triglycerides': (0, 150),
        'Mean Arterial Blood Pressure': (70, 105),
        'eGFR': (90, 150),
        'Albumin': (3.5, 5.0),
        'Fasting Glucose Level': (70, 99),
        'Normal HbA1c': (0, 5.7),
        'Postprandial Glucose Level': (0, 140),
        'Sodium': (135, 145),
        'Potassium': (3.5, 5.0),
        'Red Blood Cells Count': (4.2, 5.4),
        'White Blood Cells Count': (4500, 11000),
        'Packed Cell Volume': (36.1, 44.3),
        'Magnesium': (1.7, 2.2),
        'Uric Acid': (2.6, 6.0),
        'C-Reactive Protein (CRP)': (0.1, 1.0),
        'Body Mass Index (BMI)': (18.5, 24.9),
        'Vitamin D': (20, 50),
        'Systolic Blood Pressure': (90, 120),
        'Diastolic Blood Pressure': (60, 80)
    }
}

# Function to clip predicted values within gender-specific healthy ranges
def clip_predictions(predictions, gender, ranges):
    gender_ranges = ranges[gender]
    clipped_predictions = np.clip(predictions, [v[0] for v in gender_ranges.values()], [v[1] for v in gender_ranges.values()])
    return clipped_predictions

# Function to handle the image and parameter prediction
def predict_parameters(image_path_left, image_path_right, model, gender):
    image_left = Image.open(image_path_left).convert('RGB')
    image_right = Image.open(image_path_right).convert('RGB')

    image_left = transform(image_left).unsqueeze(0)
    image_right = transform(image_right).unsqueeze(0)

    with torch.no_grad():
        outputs_left = model(image_left)
        outputs_right = model(image_right)

    predictions_left = outputs_left.squeeze().numpy()
    predictions_right = outputs_right.squeeze().numpy()

    # Average predictions from both images
    average_predictions = (predictions_left + predictions_right) / 2
    clipped_predictions = clip_predictions(average_predictions, gender, healthy_ranges)

    return clipped_predictions

def main():
    st.title("Retinal Parameter Prediction")

    st.sidebar.header("Upload Retinal Images")
    image_file_left = st.sidebar.file_uploader("Upload Left Eye Image", type=["jpg", "jpeg", "png"])
    image_file_right = st.sidebar.file_uploader("Upload Right Eye Image", type=["jpg", "jpeg", "png"])

    st.sidebar.header("Patient Information")
    name = st.sidebar.text_input("Name")
    age = st.sidebar.number_input("Age", min_value=0, max_value=120)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

    if st.sidebar.button("Predict"):
        if image_file_left is not None and image_file_right is not None and name and age and gender:
            model_url = 'https://drive.google.com/file/d/1nbJUE_P74egDQLfTb4qIdY6AtyqkTadM/view?usp=drive_link'
            model = load_model(model_url, num_parameters=22)
            
            if model:
                # Predict parameters
                predictions = predict_parameters(image_file_left, image_file_right, model, gender)

                # Prepare the results
                result_df = pd.DataFrame([predictions], columns=list(healthy_ranges[gender].keys()))
                result_df.insert(0, 'Name', name)
                result_df.insert(1, 'Age', age)
                result_df.insert(2, 'Gender', gender)

                # Save the results to a CSV file
                result_csv = result_df.to_csv(index=False)
                st.download_button(
                    label="Download Results",
                    data=result_csv,
                    file_name=f"{name}_parameters.csv",
                    mime="text/csv"
                )
        else:
            st.error("Please upload both images and fill out all fields.")

if __name__ == "__main__":
    main()
