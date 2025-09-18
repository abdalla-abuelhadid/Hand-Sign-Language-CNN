import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

# --- Load Your Trained Model ---
# Make sure your model file is in the same folder as this script.
try:
    model = tf.keras.models.load_model('asl_mnist_model.h5')
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# --- App Title and Description ---
st.title("🤟 Hand Sign Language Classifier")
st.write("Upload an image of a hand sign, and the model will predict which letter it represents.")

# --- Class Labels ---
# This list maps the model's output index to the corresponding letter.
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# --- Image Upload Widget ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# --- Image Preprocessing Function ---
def preprocess_image(image):
    """
    Correctly processes an image into the format the model expects,
    handling different input image formats (e.g., RGBA, Grayscale, RGB).
    """
    # 1. Convert the uploaded PIL image to a NumPy array.
    image_array = np.array(image)

    # 2. Handle different image formats before processing.
    # First, check if the image is already grayscale (2 dimensions).
    if image_array.ndim == 2:
        # It's already grayscale, assign it directly.
        gray_image = image_array
    # Next, check if it has an alpha channel (4 channels).
    elif image_array.shape[2] == 4:
        # Convert from RGBA to RGB to drop the alpha channel.
        rgb_image = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        # Now convert from RGB to grayscale.
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    # Otherwise, assume it's a standard 3-channel RGB image.
    else:
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # 3. Resize the image to the model's expected size (28x28).
    resized_image = cv2.resize(gray_image, (28, 28))

    # 4. Convert the 1-channel grayscale image back to a 3-channel image.
    # This step is necessary if your model was trained on 3-channel images.
    three_channel_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)

    # 5. Reshape for the model to add the batch dimension.
    # The final shape will be (1, 28, 28, 3).
    reshaped_image = np.expand_dims(three_channel_image, axis=0)

    # 6. Normalize the pixel values.
    normalized_image = reshaped_image / 255.0
    
    return normalized_image

# --- Main App Logic ---
# This part runs when a user uploads a file and a model is successfully loaded.
if uploaded_file is not None and model is not None:
    # Display the user's uploaded image.
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image to the correct format.
    processed_image = preprocess_image(image)

    # Make a prediction.
    prediction = model.predict(processed_image)
    
    # Get the top prediction's index and confidence score.
    predicted_class_index = np.argmax(prediction)
    confidence = np.max(prediction)
    
    # Safely get the predicted label.
    if 0 <= predicted_class_index < len(class_labels):
        predicted_class_label = class_labels[predicted_class_index]
        # Display the final result.
        st.success(f"**Prediction:** {predicted_class_label}")
        st.write(f"**Confidence:** {confidence:.2f}")
    else:
        st.error(f"Error: The model produced an invalid prediction index ({predicted_class_index}).")
