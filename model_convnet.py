import tensorflow as tf
import matplotlib.pyplot as plt

tf.keras.utils.set_random_seed(2)
tf.config.experimental.enable_op_determinism()

# Define data directories
train_data_dir = "C:/Users/volka/PycharmProjects/pythonProject1/imagesAll/train/augmented"  # Update with your train data directory path
test_data_dir = "C:/Users/volka/PycharmProjects/pythonProject1/imagesAll/test/augmented"  # Update with your test data directory path
batch_size = 128
epoc_num= 15

# Define image dimensions
image_height = 128
image_width = 128

def create_convnet(input_shape, num_classes):
    model = tf.keras.Sequential([
        # Convolutional layers
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Flatten layer
        tf.keras.layers.Flatten(),

        # Fully connected layers
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')  # Output layer
    ])

    return model

# Define image data generator for training data
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

# Generate batches of training data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical',  # Use 'categorical' for one-hot encoded labels
    shuffle=True,
    subset='training',  # Use only training data
    seed=2
)

# Generate batches of validation data
validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical',  # Use 'categorical' for one-hot encoded labels
    shuffle=True,
    subset='validation',  # Use only validation data
    seed=2
)

# Define image data generator for test data
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

# Generate batches of test data
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical',  # Use 'categorical' for one-hot encoded labels
    shuffle=False  # Do not shuffle test data
)

# Get the number of classes
num_classes = len(train_generator.class_indices)

# Create the model
model = create_convnet((image_height, image_width, 3), num_classes)

# Compile the model
model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=epoc_num, validation_data=validation_generator)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_generator)
print("Test Accuracy:", test_accuracy)

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

#model.save("mymodel")