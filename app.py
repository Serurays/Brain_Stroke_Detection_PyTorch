import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from torch import nn

class StrokeClassifierCNN(nn.Module):
   def __init__(self):
       super(StrokeClassifierCNN, self).__init__()
       self.conv_block_1 = nn.Sequential(
           nn.Conv2d(1, 32, kernel_size=3, padding=1),
           nn.BatchNorm2d(32),
           nn.ReLU(inplace=True),
           nn.Conv2d(32, 64, kernel_size=3, padding=1),
           nn.BatchNorm2d(64),
           nn.ReLU(inplace=True),
           nn.MaxPool2d(kernel_size=2, stride=2),
       )

       self.conv_block_2 = nn.Sequential(
           nn.Conv2d(64, 64, kernel_size=3, padding=1),
           nn.BatchNorm2d(64),
           nn.ReLU(inplace=True),
           nn.Conv2d(64, 128, kernel_size=3, padding=1),
           nn.BatchNorm2d(128),
           nn.ReLU(inplace=True),
           nn.MaxPool2d(kernel_size=2, stride=2),
       )

       self.classifier = nn.Sequential(
           nn.Flatten(),
           nn.Linear(128*56*56, 128),
           nn.ReLU(inplace=True),
           nn.Dropout(p=0.2),
           nn.Linear(128, 1)
       )

   def forward(self, x):
       # x = self.conv_block_1(x)
       # print(x.shape)
       # x = self.conv_block_2(x)
       # print(x.shape)
       # x = self.classifier(x)
       # print(x.shape)
       # return x
       return self.classifier(self.conv_block_2(self.conv_block_1(x)))

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
