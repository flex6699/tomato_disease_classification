import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load the TensorFlow model
model = tf.keras.models.load_model('model.h5')

# Define class labels (replace these with your own)
class_labels=['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']

# Function to preprocess the image
def preprocess_image(image):
    image = np.array(image)
    image = tf.image.resize(image, (128, 128))  # Assuming your model requires input size 224x224
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image

# Function to make predictions
def predict(image):
    image = preprocess_image(image)
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    return predictions

# Streamlit app interface
def main():
    st.title('Tomato Disease  Classification')
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Perform classification when the 'Classify' button is clicked
        if st.button('Classify'):
            with st.spinner('Classifying...'):
                predictions = predict(image)
                st.write("Prediction:", class_labels[np.argmax(predictions)])

if __name__ == '__main__':
    main()
