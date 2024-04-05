import streamlit as st
import numpy as np

import pickle

# Importing the CNN model
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.cnn_model = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
        torch.nn.Tanh(),
        torch.nn.AvgPool2d(kernel_size=2, stride=5),
        torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
        torch.nn.Tanh(),
        torch.nn.AvgPool2d(kernel_size=2, stride=5))
        
        self.fc_model = torch.nn.Sequential(
        torch.nn.Linear(in_features=256, out_features=120),
        torch.nn.Tanh(),
        torch.nn.Linear(in_features=120, out_features=84),
        torch.nn.Tanh(),
        torch.nn.Linear(in_features=84, out_features=1))
        
    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        x = torch.sigmoid(x)
        
        return x

# Streamlit App
def main():
    st.title("Brain Tumor Detection")

    # Sidebar to upload image
    uploaded_file = st.sidebar.file_uploader("Choose a brain MRI image...", type=["jpg", "jpeg"])

    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        image = cv2.resize(image, (128, 128))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        if st.sidebar.button('Detect Tumor'):
            model = load_model()  # Load the trained model
            result = predict(image, model)
            st.write("Prediction:", result)

# Function to load the trained model
def load_model():
    with open("trained_model.sav", 'rb') as f:
        model = pickle.load(f)
    return model

# Function to preprocess and predict
def predict(image, model):
    # Preprocess the image
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    # Convert to Torch Tensor
    image_tensor = torch.from_numpy(image)

    # Make prediction
    model.eval()
    with torch.no_grad():
        prediction = model(image_tensor)
    
    return "Tumor Detected" if prediction.item() > 0.5 else "No Tumor Detected"

if __name__ == "__main__":
    main()
