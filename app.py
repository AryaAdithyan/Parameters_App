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

# Define the model class
class RetinalModel(nn.Module):
    def __init__(self, num_parameters):
        super(RetinalModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_parameters)
    
    def forward(self, x):
        return self.resnet(x)

# Function to download the model from Google Drive
def download_model(model_url):
    file_id = '1nbJUE_P74egDQLfTb4qIdY6AtyqkTadM'
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

# Define the healthy ranges
healthy_ranges = {
    'Male': {
        # Add ranges here
    },
    'Female': {
        # Add ranges here
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
                predictions = predict_parameters(image_file_left, image_file_right, model, gender)
                result_df = pd.DataFrame([predictions], columns=list(healthy_ranges[gender].keys()))
                result_df.insert(0, 'Name', name)
                result_df.insert(1, 'Age', age)
                result_df.insert(2, 'Gender', gender)

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
