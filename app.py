import requests
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
import os
import streamlit as st
import gdown

# Constants
MODEL_URL = "https://drive.google.com/uc?id=1nbJUE_P74egDQLfTb4qIdY6AtyqkTadM"  # Google Drive file ID
MODEL_PATH = "best_model_parameters.pth"

# Define the model architecture
class RetinalModel(nn.Module):
    def __init__(self, num_parameters):
        super(RetinalModel, self).__init__()
        self.resnet = models.resnet50(weights='DEFAULT')
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_parameters)
    
    def forward(self, x):
        return self.resnet(x)

def download_model(url, path):
    try:
        gdown.download(url, path, quiet=False)
        st.success("Model downloaded successfully.")
    except Exception as e:
        st.error(f"Error downloading model: {e}")
        raise

def load_model(model_path, num_parameters):
    model = RetinalModel(num_parameters)
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except RuntimeError as e:
        print(f"Error loading model state dict: {e}")
        return None


# Define healthy ranges
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
            left_image = Image.open(left_image).convert('RGB')
            right_image = Image.open(right_image).convert('RGB')

            left_image = transform(left_image).unsqueeze(0)
            right_image = transform(right_image).unsqueeze(0)

            if not os.path.isfile(MODEL_PATH):
                try:
                    download_model(MODEL_URL, MODEL_PATH)
                except Exception as e:
                    st.error(f"Failed to download the model: {e}")
                    return

            model = load_model(MODEL_PATH, num_parameters=22)  # Adjust num_parameters as needed

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.to(device)

            left_image = left_image.to(device)
            right_image = right_image.to(device)

            try:
                with torch.no_grad():
                    left_pred = model(left_image).cpu().numpy()[0]
                    right_pred = model(right_image).cpu().numpy()[0]

                avg_predictions = (left_pred + right_pred) / 2
                avg_predictions = clip_predictions(avg_predictions, healthy_ranges, gender)

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
            except Exception as e:
                st.error(f"Error making predictions: {e}")
        else:
            st.error("Please upload both images and fill out all fields")

if __name__ == "__main__":
    main()
