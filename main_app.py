# Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
from PIL import Image

# Loading the Model
model = load_model('plant_disease_model.h5')

# Name of Classes
CLASS_NAMES = ['Tomato-Bacterial_spot', 'Potato-Barly blight', 'Corn-Common_rust']

# Function to check if image looks like a plant leaf
def is_likely_leaf(image):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define green color range (typical for leaves)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    # Create mask for green areas
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Calculate percentage of green pixels
    green_percentage = np.sum(mask > 0) / (image.shape[0] * image.shape[1])
    
    # Consider it a leaf if at least 20% green
    return green_percentage > 0.2

# Setting Title of App
st.title("Plant Disease Detection")
st.markdown("Upload an image of the plant leaf or capture one using your camera")

# Function to preprocess and predict
def predict_disease(image):
    try:
        # Convert to OpenCV format
        if isinstance(image, np.ndarray):
            opencv_image = image
        else:
            file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
        
        # First check if this looks like a leaf
        if not is_likely_leaf(opencv_image):
            return opencv_image, None, None, "This doesn't appear to be a plant leaf. Please upload a clear image of a leaf."
        
        # Resizing and preprocessing
        opencv_image = cv2.resize(opencv_image, (256, 256))
        processed_image = np.expand_dims(opencv_image, axis=0)
        
        # Make Prediction
        Y_pred = model.predict(processed_image)
        confidence = np.max(Y_pred) * 100
        
        # Only accept predictions with high confidence
        if confidence < 70:
            return opencv_image, None, None, "The image doesn't clearly match any known plant disease. Please try with a clearer leaf image."
        
        result = CLASS_NAMES[np.argmax(Y_pred)]
        return opencv_image, result, confidence, None
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None, None, None

# Create tabs for different input methods
tab1, tab2 = st.tabs(["Upload Image", "Camera Capture"])

with tab1:
    st.header("Upload Plant Leaf Image")
    plant_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="uploader")
    submit_upload = st.button('Predict Disease from Upload')
    
    if submit_upload and plant_image is not None:
        # Check if the uploaded file is an image
        try:
            img = Image.open(plant_image)
            img.verify()
            plant_image.seek(0)
            
            # Process and predict
            opencv_image, result, confidence, error = predict_disease(plant_image)
            
            if error:
                st.warning(error)
            elif result is not None:
                st.image(opencv_image, channels="BGR", caption="Uploaded Leaf Image")
                st.success(f"Prediction Confidence: {confidence:.2f}%")
                plant_type = result.split('-')[0]
                disease = result.split('-')[1]
                st.title(f"This is {plant_type} leaf with {disease}")
        except:
            st.error("The uploaded file is not a valid image. Please upload an image file.")

with tab2:
    st.header("Capture Plant Leaf Image")
    use_camera = st.checkbox('Use Camera', key='camera_checkbox')
    
    if use_camera:
        picture = st.camera_input("Take a picture of the plant leaf")
        submit_camera = st.button('Predict Disease from Camera')
        
        if submit_camera and picture is not None:
            # Process and predict
            opencv_image, result, confidence, error = predict_disease(picture)
            
            if error:
                st.warning(error)
            elif result is not None:
                st.image(opencv_image, channels="BGR", caption="Captured Leaf Image")
                st.success(f"Prediction Confidence: {confidence:.2f}%")
                plant_type = result.split('-')[0]
                disease = result.split('-')[1]
                st.title(f"This is {plant_type} leaf with {disease}")

# Add information about supported plants
st.sidebar.header("Supported Plants")
st.sidebar.info(
    """
    This model can detect diseases in:
    - Tomato leaves (Bacterial spot)
    - Potato leaves (Early blight)
    - Corn leaves (Common rust)
    """
)

# Add troubleshooting tips
st.sidebar.header("Tips for Best Results")
st.sidebar.info(
    """
    1. Capture only the leaf against a plain background
    2. Ensure good lighting
    3. Fill most of the frame with the leaf
    4. Avoid shadows on the leaf
    5. Use clear, undamaged leaves when possible
    """
)