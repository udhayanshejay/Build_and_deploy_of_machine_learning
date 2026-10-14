
import requests

image_path = input("Enter the path to a tomato leaf image: ").strip().strip('"')

with open(image_path, "rb") as image_file:
    response = requests.post(
        "http://127.0.0.1:5000/predict",
        files={"image": image_file}
    )

print("Status code:", response.status_code)
print("Prediction:", response.json())
