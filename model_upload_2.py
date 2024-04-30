import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf

test_data_dir = "C:/Users/volka\PycharmProjects\pythonProject1\imagesAll/test/augmented"

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

# Generate batches of test data
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(128, 128),
    batch_size=8,
    class_mode='categorical',  # Use 'categorical' for one-hot encoded labels
    shuffle=False  # Do not shuffle test data
)

# Load your trained model
model = tf.keras.models.load_model("mymodel")  # Load your saved model

# Make predictions on the test dataset
predictions = model.predict(test_generator)

# Convert predictions and ground truth labels to class indices
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

# Generate confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)

# Plot confusion matrix as heatmap
class_labels = list(test_generator.class_indices.keys())  # Get class labels
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_labels)
disp.plot(cmap='Blues')  # Adjust the colormap as needed
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
