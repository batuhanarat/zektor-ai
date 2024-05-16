import sys
import json
import numpy as np
from PIL import Image
import requests
import base64
import io
from tensorflow.keras.models import load_model

dev_predictions = []
health_predictions = []

image_ids = []
plant_ids = []

# Load the saved model
development_phase_model = load_model("model2.keras")
health_status_model = load_model("model_health.keras")

# Function to preprocess an image before passing it to the model
def preprocess_image(image):
    # Convert the image to a numpy array
    image_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    # Expand the dimensions of the image array to match the model's input shape
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def send_dev_predictions(predictions, image_ids, plant_ids):
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

def send_health_predictions(predictions, image_ids, plant_ids):
    url = "http://54.208.55.232:5004/healthStatusOutput"
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
    dev_pred = development_phase_model.predict(preprocessed_image)
    health_pred = health_status_model.predict(preprocessed_image)

    predicted_dev_class_index = np.argmax(dev_pred)
    predicted_health_class_index = np.argmax(health_pred)

    dev_predictions.append(predicted_dev_class_index + 1)
    health_predictions.append(predicted_health_class_index)

    image_ids.append(image_data["image_id"])
    plant_ids.append(image_data["plant_id"])

print("Predicted dev probabilities:", dev_predictions)
print("Predicted health probabilities:", health_predictions)
print("Image IDs:", image_ids)
print("Plant IDs:", plant_ids)

send_dev_predictions(dev_predictions, image_ids, plant_ids)
send_health_predictions(health_predictions, image_ids, plant_ids)

# Print the predicted probabilities and IDs
with open("predictions.txt", "w") as f:
    for i in range(len(image_ids)):
        f.write(f" Plant ID: {plant_ids[i]}, Image ID: {image_ids[i]}, Predicted Dev Probabilities: {dev_predictions[i]}\n")
        f.write(f" Plant ID: {plant_ids[i]}, Image ID: {image_ids[i]}, Predicted Health Probabilities: {health_predictions[i]}\n")

print("Predictions saved to predictions.txt")
