import requests
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
import os
import streamlit as st

MODEL_URL = "https://drive.google.com/file/d/1nbJUE_P74egDQLfTb4qIdY6AtyqkTadM"  # Replace with your model link
MODEL_PATH = "best_model_parameters.pth"

# Function to download model file
def download_model(url, path):
    response = requests.get(url, stream=True)
    with open(path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

# Download model if not already present
if not os.path.isfile(MODEL_PATH):
    download_model(MODEL_URL, MODEL_PATH)

# Load the trained model
class RetinalModel(nn.Module):
    def __init__(self, num_parameters):
        super(RetinalModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_parameters)
    
    def forward(self, x):
        return self.resnet(x)

def load_model(model_path, num_parameters):
    model = RetinalModel(num_parameters)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Healthy ranges dictionary for gender-specific ranges
healthy_ranges = {
    'Total Cholesterol': (125, 200),
    'LDL': (0, 100),
    'HDL': {'Men': (40, 100), 'Women': (50, 100)},
    'Triglycerides': (0, 150),
    'Mean Arterial Blood Pressure': (70, 105),
    'eGFR': (90, 150),
    'Albumin': (3.5, 5.0),
    'Fasting Glucose Level': (70, 99),
    'Normal HbA1c': (0, 5.7),
    'Postprandial Glucose Level': (0, 140),
    'Sodium': (135, 145),
    'Potassium': (3.5, 5.0),
    'Red Blood Cells Count': {'Men': (4.7, 6.1), 'Women': (4.2, 5.4)},
    'White Blood Cells Count': (4500, 11000),
    'Packed Cell Volume': {'Men': (40.7, 50.3), 'Women': (36.1, 44.3)},
    'Magnesium': (1.7, 2.2),
    'Uric Acid': (2.6, 7.2),
    'C-Reactive Protein (CRP)': (0.1, 1.0),
    'Body Mass Index (BMI)': (18.5, 24.9),
    'Vitamin D': (20, 50),
    'Systolic Blood Pressure': (90, 120),
    'Diastolic Blood Pressure': (60, 80)
}

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to clip predictions
def clip_predictions(predictions, ranges, gender):
    clipped_predictions = []
    for i, param in enumerate(predictions):
        param_name = list(ranges.keys())[i]
        if param_name in ['HDL', 'Red Blood Cells Count', 'Packed Cell Volume']:
            range_for_gender = ranges[param_name].get(gender, ranges[param_name])
        else:
            range_for_gender = ranges[param_name]
        clipped_predictions.append(np.clip(param, range_for_gender[0], range_for_gender[1]))
    return clipped_predictions

# Streamlit app
def main():
    st.title("Retinal Image Parameter Prediction")

    # Upload images
    left_image = st.file_uploader("Upload Left Retinal Image", type=["jpg", "jpeg", "png"])
    right_image = st.file_uploader("Upload Right Retinal Image", type=["jpg", "jpeg", "png"])
    name = st.text_input("Name")
    age = st.number_input("Age", min_value=0, max_value=120)
    gender = st.selectbox("Gender", ["Male", "Female"])

    if st.button("Predict"):
        if left_image and right_image and name and age and gender:
            # Load and preprocess images
            left_image = Image.open(left_image).convert('RGB')
            right_image = Image.open(right_image).convert('RGB')

            left_image = transform(left_image).unsqueeze(0)
            right_image = transform(right_image).unsqueeze(0)

            # Download model if not exists
            model_path = 'best_model.pth'
            if not os.path.exists(model_path):
                file_id = '1nbJUE_P74egDQLfTb4qIdY6AtyqkTadM'  # Your Google Drive file ID
                download_file_from_google_drive(file_id, model_path)

            # Load model and make predictions
            model = load_model(model_path, num_parameters=22)  # Replace 21 with the actual number of parameters

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.to(device)

            left_image = left_image.to(device)
            right_image = right_image.to(device)

            with torch.no_grad():
                left_pred = model(left_image).cpu().numpy()[0]
                right_pred = model(right_image).cpu().numpy()[0]

            # Average the predictions
            avg_predictions = (left_pred + right_pred) / 2
            avg_predictions = clip_predictions(avg_predictions, healthy_ranges, gender)

            # Prepare output
            results = {
                "Name": [name],
                "Age": [age],
                "Gender": [gender]
            }
            results.update({param: [value] for param, value in zip(healthy_ranges.keys(), avg_predictions)})

            df = pd.DataFrame(results)
            output_csv = 'predictions.csv'
            df.to_csv(output_csv, index=False)

            st.write("Predictions saved to:", output_csv)
            st.download_button(label="Download CSV", data=df.to_csv(index=False), file_name="predictions.csv", mime="text/csv")
        else:
            st.error("Please upload both images and fill out all fields")

if __name__ == "__main__":
    main()
