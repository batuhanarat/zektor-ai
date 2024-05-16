import sys
import json
import numpy as np
from PIL import Image
import requests
import base64
import cv2
import io
from tensorflow.keras.models import load_model

predictions = []
image_ids = []
plant_ids = []

# Load the saved model
#loaded_model = load_model("model.keras")
loaded_model = load_model("model2.keras")

# Function to preprocess an image before passing it to the model
def preprocess_image(image):
    # Convert the image to a numpy array
    image_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    # Expand the dimensions of the image array to match the model's input shape
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def send_predictions(predictions, image_ids, plant_ids):
    url = "http://54.208.55.232:5004/developmentPhaseOutput"
    data = {
        'predictions':  [int(prediction) for prediction in predictions],
        'imageIds': image_ids,
        'plantIds' : plant_ids
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json=data, headers=headers)
    print("Status Code:", response.status_code)  # Debug: Print status code
    print("Response Content:", response.text)
    return response


# Get the image data and IDs from the command line arguments
images_data = json.loads(sys.argv[1])
class_labels = {0: 'class1', 1: 'class2', 2: 'class3', 3: 'class4'}
# Initialize lists to store predictions and IDs

# Iterate over the images and their corresponding IDs
for image_data in images_data:
    image_bytes = base64.b64decode(image_data["image"])
    image = Image.open(io.BytesIO(image_bytes))
    preprocessed_image = preprocess_image(image)
    prediction = loaded_model.predict(preprocessed_image)
    predicted_class_index = np.argmax(prediction)
    predictions.append(predicted_class_index+1)
    image_ids.append(image_data["image_id"])
    plant_ids.append(image_data["plant_id"])
    
print("Predicted probabilities:", predictions)
print("Image IDs:", image_ids)
print("Plant IDs:", plant_ids)

send_predictions(predictions,image_ids,plant_ids)


# Print the predicted probabilities and IDs

with open("predictions.txt", "w") as f:
    for i in range(len(image_ids)):
        f.write(f" Plant ID: {plant_ids[i]},Image ID: {image_ids[i]}, Predicted Probabilities: {predictions[i]}\n")

print("Predictions saved to predictions.txt")
