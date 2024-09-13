import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

# Define image dimensions
img_height, img_width = 150, 150

# Function to load and preprocess an image
def load_and_preprocess_image(img):
    img = img.resize((img_height, img_width))  # Resize image
    img_array = np.array(img)  # Convert image to array
    img_array = img_array.astype('float32')  # Convert to float32 for normalization
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize image to range [0, 1]
    return img_array

# Function to define a simple CNN model
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')  # 2 output classes (cat, dog)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Load a pre-trained model or create a new one
model = create_model()
# Uncomment the following line and provide the path to your pre-trained weights if available
# model.load_weights('path_to_your_weights.h5')

# Streamlit app
st.title('Image Classification with CNN')
st.write('Upload an image file to classify it.')

# Create a file uploader in Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and preprocess the image
    img = Image.open(uploaded_file)  # Open the uploaded image
    img_array = load_and_preprocess_image(img)  # Preprocess the image

    # Predict the class
    predictions = model.predict(img_array)  # Get predictions
    class_names = ['cat', 'dog']  # Define class names
    predicted_class = class_names[np.argmax(predictions)]  # Get the predicted class
    confidence = np.max(predictions)  # Get the confidence of the prediction

    # Display the image and the prediction result
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)  # Show uploaded image
    st.write(f"Predicted Class: {predicted_class}")  # Show predicted class
    st.write(f"Confidence: {confidence:.2f}")  # Show confidence
