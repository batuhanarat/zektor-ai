import requests
import base64
from io import BytesIO
from PIL import Image


def create_user():
    url = 'http://172.20.10.3:5002/user'

    # Prepare headers and payload
    headers = {'Content-Type': 'application/json'}
    payload = {
        'device_id': "device-2"
    }

    #
    # Send the POST request with JSON body
    response = requests.post(url, json=payload, headers=headers)
    print(response.text)


def create_plant(user_id, plant_type):
    url = 'http://172.20.10.3:5002/plant'
    headers = {'Content-Type': 'application/json'}
    payload = {
        'userId': user_id,  # Corrected to pass the user_id as an object with a 'userId' property
        'type': plant_type  # Corrected variable name from 'type' to 'plant_type' to avoid conflict
    }
    response = requests.post(url, json=payload, headers=headers)
    print(response.text)


userId = "66264c11c15bb16b60f83eaa"
plant_type = "lettuce"  # Corrected variable name from 'type' to 'plant_type' to avoid conflict
#create_plant("66264c11c15bb16b60f83eaa", "lettuce")

def create_plant_image(plantImage,plantOrder,userId):
    url = 'http://172.20.10.3:5002/plantImage'
    headers = {'Content-Type': 'application/json'}
    payload = {
        'base64': plantImage,  # Corrected to pass the user_id as an object with a 'userId' property
        'order': plantOrder,
        "userId": userId # Corrected variable name from 'type' to 'plant_type' to avoid conflict
    }
    response = requests.post(url, json=payload, headers=headers)
    print(response.text)
def encode_file_to_base64(image):
    img_bytes = BytesIO()
    image.save(img_bytes, format='PNG')
    #image.save(img_bytes, format='JPEG')
    return base64.b64encode(img_bytes.getvalue()).decode('utf-8')

#file_path = "2-74.jpg"
#image = Image.open(file_path)
#image= image.resize((64, 64))
#encoded_image = encode_file_to_base64(image)
#create_plant_image(encoded_image,"6626558002a3ff369595c5d9",0)
# Setup the API endpoint
# url = 'http://192.168.43.43:5001/images'

# Prepare headers and payload
# headers = {'Content-Type': 'application/json'}
# payload = {'base64': encoded_image}

# Send the POST request with JSON body
# response = requests.post(url, json=payload, headers=headers)
# print(response.text)

def test_user_endpoint():
    create_user()

def test_plant_endpoint():
    userId = "66266df42fc41b0d13849d3e"
    plant_type = "lettuce"  # Corrected variable name from 'type' to 'plant_type' to avoid conflict
    create_plant(userId, "lettuce")

def test_image_endpoint():
    file_path = "1-25.jpg"
    image = Image.open(file_path)
    image= image.resize((64, 64))
    encoded_image = encode_file_to_base64(image)
    create_plant_image(encoded_image,0,"66266df42fc41b0d13849d3e")

test_image_endpoint()
