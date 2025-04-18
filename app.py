import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('animal_classifier.h5')

model = load_model()

# Class names (must match training order)
class_names = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 
               'elephant', 'horse', 'sheep', 'spider', 'squirrel']

# App title
st.title(" Animal Classifier")
st.write("Upload an image to classify the animal")

# Image uploader
uploaded_file = st.file_uploader(
    "Choose an animal image...", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', width=300)
    
    # Preprocess
    img_array = np.array(image.resize((224, 224))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    with st.spinner('Classifying...'):
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions) * 100
    
    # Results
    st.success(f"**Prediction:** {predicted_class}")
    st.metric("Confidence", f"{confidence:.2f}%")
    
    # Show all probabilities
    st.subheader("All Predictions:")
    for i, (class_name, prob) in enumerate(zip(class_names, predictions[0])):
        st.progress(float(prob), text=f"{class_name}: {prob*100:.2f}%")