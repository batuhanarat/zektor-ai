import sys
import json
import numpy as np
from PIL import Image
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

# Initialize lists to store predictions and IDs
predictions = []
ids = []

# Iterate over the images and their corresponding IDs
for image_data in images_data:
    image_bytes = np.frombuffer(base64.b64decode(image_data["image"]), dtype=np.uint8)
    
    # Decode the image bytes into an image array
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Make predictions on the preprocessed image
    prediction = loaded_model.predict(preprocessed_image)

    # Assuming your model outputs probabilities for different classes,
    # you can append the predicted probabilities to the predictions list
    predictions.append(prediction.tolist()[0])

    # Append the ID of the image to the IDs list
    ids.append(image_data["id"])

# Print the predicted probabilities and IDs
print("Predicted probabilities:", predictions)
print("Image IDs:", ids)
with open("predictions.txt", "w") as f:
    for i in range(len(ids)):
        f.write(f"Image ID: {ids[i]}, Predicted Probabilities: {predictions[i]}\n")

print("Predictions saved to predictions.txt")
