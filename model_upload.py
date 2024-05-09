import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os


# Load the saved model
loaded_model = load_model("model.keras")

# Function to preprocess an image before passing it to the model
def preprocess_image(image):
    # Convert the image to a numpy array
    image_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    # Expand the dimensions of the image array to match the model's input shape
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Path to the new image you want to make predictions on
new_image_path = '1-5.jpg'

if os.path.exists(new_image_path):
    print(f"Image file found at path: {new_image_path}")
else:
    print(f"Image file not found at path: {new_image_path}")


try:
    image = Image.open(new_image_path)
    print("Image opened successfully.")
    # You can add further processing here, such as displaying the image or checking its properties
except Exception as e:
    print(f"Error opening image: {e}")
# Preprocess the new image
new_image = preprocess_image(new_image_path)

# Make predictions on the new image
predictions = loaded_model.predict(new_image)

# Assuming your model outputs probabilities for different classes, you can print the predicted probabilities
print("Predicted probabilities:", predictions)

# If you want to get the index of the class with the highest probability, you can use argmax
predicted_class_index = np.argmax(predictions)
print("Predicted class index:", predicted_class_index)

# You might have a dictionary mapping class indices to class labels
class_labels = {0: 'class1', 1: 'class2', 2: 'class3', 3: 'class4'}  # Example dictionary
predicted_class_label = class_labels[predicted_class_index]
print("Predicted class label:", predicted_class_label)
