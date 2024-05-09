import sys
import json
import numpy as np
from PIL import Image
import base64
import cv2
import io
from tensorflow.keras.models import load_model

# Load the saved model
loaded_model = load_model("model.keras")

# Function to preprocess an image before passing it to the model
def preprocess_image(image):
    # Convert the image to a numpy array
    image_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    # Expand the dimensions of the image array to match the model's input shape
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Get the image data and IDs from the command line arguments
images_data = json.loads(sys.argv[1])
class_labels = {0: 'class1', 1: 'class2', 2: 'class3', 3: 'class4'}
# Initialize lists to store predictions and IDs
predictions = []
ids = []

# Iterate over the images and their corresponding IDs
for image_data in images_data:
    image_bytes = base64.b64decode(image_data["image"])
    image = Image.open(io.BytesIO(image_bytes))
    preprocessed_image = preprocess_image(image)
    prediction = loaded_model.predict(preprocessed_image)
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = class_labels[predicted_class_index]                                 
    predictions.append(predicted_class_index)
    ids.append(image_data["id"])




# Print the predicted probabilities and IDs
print("Predicted probabilities:", predictions)
print("Image IDs:", ids)
with open("predictions.txt", "w") as f:
    for i in range(len(ids)):
        f.write(f"Image ID: {ids[i]}, Predicted Probabilities: {predictions[i]}\n")

print("Predictions saved to predictions.txt")
