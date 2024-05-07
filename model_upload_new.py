import numpy as np
from PIL import Image

from keras.saving import load_model

# Load the saved model
#loaded_model = load_model('C:/Users/volka/Desktop/server/saved_model.pb')
loaded_model = load_model("model.keras")

# Function to preprocess an image before passing it to the model
def preprocess_image(image):
    # Convert the image to a numpy array
    image_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    # Expand the dimensions of the image array to match the model's input shape
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Path to the new image you want to make predictions on
new_image_path = 'C:/Users/volka/Desktop/server/aug_0_132.jpeg'

# Open the image using PIL
image = Image.open(new_image_path)


# Preprocess the new image
new_image = preprocess_image(image)

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
