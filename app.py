import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the trained model (once)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("/home/developer/Coding/Kaggle/BioNonBioDegradable/BioOrNonBioModel.h5")

model = load_model()

# Preprocessing function for images
def preprocess_image(image, target_size=(64, 64)):
    image = image.resize(target_size)
    #image = np.array(image) / 255.0  # normalize pixel values
    # if image.shape[-1] == 4:  # drop alpha channel if present
    #     image = image[..., :3]
    image = np.expand_dims(image, axis=0)  # add batch dimension
    return image

# UI
st.title("Biodegradable Material Classifier 🌱")

uploaded_file = st.file_uploader("Upload an image of the item", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Predicting...")
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0][0]  # sigmoid output

    # Threshold at 0.5
    if prediction >= 0.5:
        st.error(f"Prediction: **Non-Biodegradable** 🧴 ({prediction:.2f})")
    else:
        st.success(f"Prediction: **Biodegradable** 🌿 ({prediction:.2f})")
