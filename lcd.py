
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import base64

# Load pre-trained model (ensure the path is correct)
MODEL_PATH = "C:\\Users\\Dell\\Downloads\\lung cancer detection system\\lung_cancer_model.h5"
model = load_model(MODEL_PATH)

# Verify the input shape of the model
input_shape = model.input_shape
print("Model input shape:", input_shape)

# Class labels
class_labels = ['lung_aca', 'lung_scc', 'lung_n']

# Function to preprocess the uploaded image
def preprocess_image(image):
    target_size = (input_shape[1], input_shape[2])  # Resize image to match the input size of the model
    image = image.resize(target_size)  # Resize image to the target size
    image = np.array(image) / 255.0   # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to make predictions
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100  # Confidence as a percentage
    return predicted_class, confidence, prediction

# Function to set background image
def set_background_image(image_url):
    page_bg_img = f'''
    <style>
    body {{
        background-image: url({image_url});
        background-size: cover;
        background-position: center;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Main function to run the app
def main():
    st.set_page_config(page_title="Lung Cancer Detection System", layout="wide")

    # Set background image
    set_background_image('https://path-to-your-image.jpg')  # Replace with your image URL

    st.title("Lung Cancer Detection System")
    st.write("""
        Welcome to the Lung Cancer Detection System. This application uses a deep learning model 
        to predict whether an uploaded image of a lung sample indicates normal tissue, adenocarcinoma, 
        or squamous cell carcinoma. Please log in to proceed.
    """)

    # Initialize session state variables
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    # Login page
    if not st.session_state.logged_in:
        with st.form("login_form"):
            st.subheader("Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_button = st.form_submit_button("Login")

            if login_button:
                if username == "admin" and password == "password":
                    st.session_state.logged_in = True
                    st.success("Login successful")
                else:
                    st.error("Invalid username or password")

    # Home page
    if st.session_state.logged_in:
        st.sidebar.header("Navigation")
        pages = ["Home", "About", "Contact"]
        choice = st.sidebar.selectbox("Select a page:", pages)

        if choice == "Home":
            st.header("Home Page")
            st.write("Upload an image for lung cancer detection")

            # File uploader
            uploaded_file = st.file_uploader("Choose an image...", type="jpg")

            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image.', use_column_width=True)
                st.write("")
                st.write("Classifying...")

                predicted_class, confidence, prediction = predict(image)
                
                st.write(f"Prediction: {predicted_class}")
                st.write(f"Confidence: {confidence:.2f}%")
                st.write(f"Raw Prediction: {prediction}")

                # Manual verification (optional)
                if predicted_class not in class_labels:
                    st.warning("The uploaded image may not be a valid lung sample. Please upload a relevant image.")

                # Additional Information
                st.write("---")
                st.write("### Prediction Breakdown")
                for label, score in zip(class_labels, prediction[0]):
                    st.write(f"{label}: {score*100:.2f}%")

        elif choice == "About":
            st.header("About")
            st.write("""
                The Lung Cancer Detection System is designed to assist in the early detection and diagnosis
                of lung cancer by analyzing histopathological images. This application leverages a deep learning model 
                that has been trained on a comprehensive dataset to classify images into three categories:
                
                - **Normal tissue (lung_n)**: Indicates healthy lung tissue without signs of cancer.
                - **Adenocarcinoma (lung_aca)**: A type of cancer that forms in mucus-secreting glands.
                - **Squamous cell carcinoma (lung_scc)**: A type of cancer that forms in the squamous cells, which are flat cells that line the airways in the lungs.
                
                ### Features:
                - **User-friendly Interface**: Easy-to-use interface for uploading images and viewing predictions.
                - **High Accuracy**: The model has been trained to achieve high accuracy in classifying lung tissue images.
                - **Real-time Predictions**: Get instant predictions with confidence scores for the uploaded images.
                - **Prediction Breakdown**: Detailed breakdown of prediction scores for each class.
                
                ### Creators:
                This application was developed by:
                - Aditi Madan
                - Diksha Khandelwal
                - Halija Vahaj
                
                We hope this tool helps in the early detection and treatment of lung cancer.
            """)

        elif choice == "Contact":
            st.header("Contact")
            st.write("""
                For any inquiries or support, please contact us at:
                - Email: support@lungcancerdetector.com
                - Phone: +1 (800) 123-4567
            """)

if __name__ == '__main__':
    main()
