import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# Define your ImageDataGenerator with the desired augmentations
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='wrap')

# Specify the directory where your original images are located
original_dataset_dir = 'C:/Users/volka/PycharmProjects/pythonProject1/images'

# Specify the directory where you want to save the augmented images
augmented_dataset_dir = 'C:/Users/volka/PycharmProjects/pythonProject1/augmented'

# Loop through each folder (class) in the original dataset directory
for class_name in os.listdir(original_dataset_dir):
    class_dir = os.path.join(original_dataset_dir, class_name)
    augmented_class_dir = os.path.join(augmented_dataset_dir, class_name)
    if not os.path.exists(augmented_class_dir):
        os.makedirs(augmented_class_dir)

    # Loop through each image in the class folder
    for image_file in os.listdir(class_dir):
        img_path = os.path.join(class_dir, image_file)
        img = load_img(img_path)  # Load image
        x = img_to_array(img)  # Convert to numpy array
        x = x.reshape((1,) + x.shape)  # Reshape to (1, height, width, channels)

        # Generate and save augmented images
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=augmented_class_dir, save_prefix='aug',
                                  save_format='jpeg'):
            i += 1
            if i > 20:  # Save 20 augmented images per original image
                break
