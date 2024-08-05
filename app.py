pip install streamlit
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

# Load the pre-trained model
model_path = 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(model_path, map_location=device)
model.to(device)
model.eval()

# Define the transform
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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

def clip_predictions(predictions, healthy_ranges):
    clipped_preds = []
    for pred, (min_val, max_val) in zip(predictions[0], healthy_ranges.values()):
        clipped_preds.append(min(max(pred, min_val), max_val))
    return clipped_preds

def predict_average_values(left_img_path, right_img_path):
    left_image = Image.open(left_img_path).convert('RGB')
    right_image = Image.open(right_img_path).convert('RGB')

    left_image = transform(left_image).unsqueeze(0).to(device)
    right_image = transform(right_image).unsqueeze(0).to(device)

    with torch.no_grad():
        left_pred = model(left_image)
        right_pred = model(right_image)

    left_pred = clip_predictions(left_pred.cpu().numpy(), healthy_ranges)
    right_pred = clip_predictions(right_pred.cpu().numpy(), healthy_ranges)

    avg_pred = (left_pred + right_pred) / 2.0

    return avg_pred

# Streamlit app
st.title('Retinal Image Parameter Prediction')

# Input fields
patient_name = st.text_input('Patient Name')
age = st.number_input('Age', min_value=0, max_value=120, step=1)
gender = st.selectbox('Gender', ['Male', 'Female'])
left_image_file = st.file_uploader('Upload Left Retinal Image', type=['jpg', 'jpeg', 'png'])
right_image_file = st.file_uploader('Upload Right Retinal Image', type=['jpg', 'jpeg', 'png'])

if st.button('Predict and Download CSV'):
    if not left_image_file or not right_image_file:
        st.error('Please upload both left and right retinal images.')
    else:
        left_img_path = f'/tmp/{left_image_file.name}'
        right_img_path = f'/tmp/{right_image_file.name}'

        with open(left_img_path, 'wb') as f:
            f.write(left_image_file.getbuffer())

        with open(right_img_path, 'wb') as f:
            f.write(right_image_file.getbuffer())

        avg_values = predict_average_values(left_img_path, right_img_path)

        output = pd.DataFrame({
            'Patient Name': [patient_name],
            'Age': [age],
            'Gender': [gender],
            **dict(zip(healthy_ranges.keys(), avg_values))
        })

        csv = output.to_csv(index=False)
        st.download_button(
            label='Download CSV',
            data=csv,
            file_name='predicted_values.csv',
            mime='text/csv'
        )

        st.success('CSV file is ready for download.')
