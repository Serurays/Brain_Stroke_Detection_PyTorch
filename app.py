import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from stroke_detection_model import StrokeClassifierCNN

model = StrokeClassifierCNN()
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

# Function to preprocess image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)
    return image

# Function to perform inference
def predict_image(image):
    model.eval()
    with torch.no_grad():
        output = model(image)
        prediction = torch.round(torch.sigmoid(output))
    return prediction.item()

# Streamlit app
def main():
    st.title("MRI Stroke Detection")

    uploaded_file = st.file_uploader("Upload an MRI image", type=["png", "jpg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded MRI Image', use_column_width=True)

        if st.button("Predict"):
            # Preprocess image
            processed_image = preprocess_image(image)

            # Make prediction
            prediction = predict_image(processed_image)

            if prediction == 1:
                st.write("Prediction: Stroke")
            else:
                st.write("Prediction: No Stroke")

if __name__ == '__main__':
    main()