import tensorflow as tf
import matplotlib.pyplot as plt
tf.keras.utils.set_random_seed(12)

# Define data directories
train_data_dir = "C:/Users/volka\PycharmProjects\pythonProject1\imagesAll/train/augmented"  # Update with your train data directory path
test_data_dir = "C:/Users/volka\PycharmProjects\pythonProject1\imagesAll/test/augmented"  # Update with your test data directory path
batch_size = 4
epoc_num = 10

# Define image dimensions
image_height = 128
image_width = 128



# Define the U-Net architecture
def conv_block(inputs, num_filters):
    x = tf.keras.layers.Conv2D(num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        inputs)
    x = tf.keras.layers.Conv2D(num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    return x


def unet(input_shape, num_classes):
    inputs = tf.keras.layers.Input(input_shape)

    # Encoder
    conv1 = conv_block(inputs, 64)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_block(pool1, 128)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_block(pool2, 256)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)


    conv5 = conv_block(pool3, 512)
    drop5 = tf.keras.layers.Dropout(0.5)(conv5)

    # Decoder

    up7 = tf.keras.layers.Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        tf.keras.layers.UpSampling2D(size=(2, 2))(drop5))
    merge7 = tf.keras.layers.concatenate([conv3, up7], axis=3)
    conv7 = conv_block(merge7, 256)

    up8 = tf.keras.layers.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        tf.keras.layers.UpSampling2D(size=(2, 2))(conv7))
    merge8 = tf.keras.layers.concatenate([conv2, up8], axis=3)
    conv8 = conv_block(merge8, 128)

    up9 = tf.keras.layers.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        tf.keras.layers.UpSampling2D(size=(2, 2))(conv8))
    merge9 = tf.keras.layers.concatenate([conv1, up9], axis=3)
    conv9 = conv_block(merge9, 64)

    flat = tf.keras.layers.Flatten()(conv9)

    # Fully connected layers
    dense = tf.keras.layers.Dense(128, activation='relu')(flat)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(dense)

    # # Output layer
    # outputs = tf.keras.layers.Conv2D(num_classes, 1, activation='softmax')(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
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

# Create the U-Net model
model = unet((image_height, image_width, 3), num_classes)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=epoc_num, validation_data=validation_generator)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_generator)
print("Test Accuracy:", test_accuracy)

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

model.save("my_unet_model")