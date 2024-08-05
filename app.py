import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import pandas as pd
import gdown

# Define the model class
class RetinalModel(nn.Module):
    def __init__(self, num_parameters):
        super(RetinalModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_parameters)
    
    def forward(self, x):
        return self.resnet(x)

# Download model function
def download_model():
    url = "https://drive.google.com/uc?export=download&id=1nbJUE_P74egDQLfTb4qIdY6AtyqkTadM"
    output = "/mount/src/parameters_app/best_model_parameters.pth"
    gdown.download(url, output, quiet=False)

# Load the model
def load_model(model_path, num_parameters):
    model = RetinalModel(num_parameters)
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True), strict=False)
        model.eval()
        return model
    except RuntimeError as e:
        print(f"Error loading model state dict: {e}")
        return None

# Preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Clip predictions
def clip_predictions(predictions, ranges):
    return np.clip(predictions, [v[0] for v in ranges.values()], [v[1] for v in ranges.values()])

# Define healthy ranges
healthy_ranges = {
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
    'Red Blood Cells Count': (4.2, 6.1),
    'White Blood Cells Count': (4500, 11000),
    'Packed Cell Volume': (36.1, 50.3),
    'Magnesium': (1.7, 2.2),
    'Uric Acid': (2.6, 7.2),
    'C-Reactive Protein (CRP)': (0.1, 1.0),
    'Body Mass Index (BMI)': (18.5, 24.9),
    'Vitamin D': (20, 50),
    'Systolic Blood Pressure': (90, 120),
    'Diastolic Blood Pressure': (60, 80)
}

def main():
    st.title("Parameters Prediction Using Retinal Image")

    # Download and load the model
    download_model()
    model_path = "/mount/src/parameters_app/best_model_parameters.pth"
    model = load_model(model_path, num_parameters=len(healthy_ranges))
    if model is None:
        st.error("Failed to load model.")
        return
    
    # User input
    name = st.text_input("Enter Patient Name")
    age = st.number_input("Enter Patient Age", min_value=0)
    gender = st.selectbox("Select Gender", ["Male", "Female"])
    
    # Upload images
    uploaded_left_image = st.file_uploader("Upload Left Retinal Image", type=["jpg", "jpeg", "png"])
    uploaded_right_image = st.file_uploader("Upload Right Retinal Image", type=["jpg", "jpeg", "png"])

    # Predict button
    if st.button("Predict"):
        if uploaded_left_image and uploaded_right_image and name and age and gender:
            # Process images
            left_image = Image.open(uploaded_left_image).convert("RGB")
            right_image = Image.open(uploaded_right_image).convert("RGB")
            
            left_image_tensor = preprocess_image(left_image)
            right_image_tensor = preprocess_image(right_image)
            
            # Predict
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.to(device)
            
            left_image_tensor = left_image_tensor.to(device)
            right_image_tensor = right_image_tensor.to(device)
            
            with torch.no_grad():
                left_prediction = model(left_image_tensor).cpu().numpy().flatten()
                right_prediction = model(right_image_tensor).cpu().numpy().flatten()
            
            # Average predictions
            average_prediction = (left_prediction + right_prediction) / 2
            averaged_prediction = clip_predictions(average_prediction, healthy_ranges)
            
            # Output results
            result_df = pd.DataFrame([averaged_prediction], columns=list(healthy_ranges.keys()))
            result_df.insert(0, "Name", [name])
            result_df.insert(1, "Age", [age])
            result_df.insert(2, "Gender", [gender])
            result_df.to_csv('predicted_parameters.csv', index=False)
            
            st.write("Predicted Parameters:")
            st.dataframe(result_df)
            
            st.download_button(
                label="Download CSV",
                data=result_df.to_csv(index=False).encode('utf-8'),
                file_name='predicted_parameters.csv',
                mime='text/csv'
            )
        else:
            st.error("Please upload both images and fill out all fields.")

if __name__ == "__main__":
    main()
