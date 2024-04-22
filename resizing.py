from PIL import Image
import os

def resize_images_in_folder(folder_path,target_path, target_size):
    """
    Resize all images in a folder to the target size.

    Args:
    - folder_path (str): Path to the folder containing images.
    - target_size (tuple): Target size in the format (width, height).
    """
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            try:
                with Image.open(image_path) as img:
                    resized_img = img.resize(target_size)
                    new_path = os.path.join(target_path, filename)
                    resized_img.save(new_path)
            except Exception as e:
                print(f"Error resizing image {filename}: {str(e)}")

# Example usage
for i in range(1,5):
    for mode in ["train","test"]:
        folder_path = f"C:/Users/volka\PycharmProjects\pythonProject1\imagesAll/{mode}/images/{i}"
        target_path = f"C:/Users/volka\PycharmProjects\pythonProject1\imagesAll/{mode}/resized/{i}"
        target_size = (128, 128)  # Specify the target size here
        resize_images_in_folder(folder_path,target_path, target_size)
