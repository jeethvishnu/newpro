# import torch
# from torchvision import transforms
# from PIL import Image
# import matplotlib.pyplot as plt
# from model import PretrainedCNNModel
#
# race_mapping = {
#     0: "White",
#     1: "Black",
#     2: "Asian",
#     3: "Indian",
#     4: "Others"
# }
#
# # Define transformations for the input image
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Resize images to (224, 224)
#     transforms.ToTensor(),  # Convert images to tensors
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize images
# ])
#
#
# # Function to predict the race of an input image
# def predict_race(image_path, model_path):
#     # Load the image
#     image = Image.open(image_path).convert('RGB')
#
#     # Apply transformations
#     image = transform(image)
#
#     # Add batch dimension
#     image = image.unsqueeze(0)
#
#     # Load the model
#     model = PretrainedCNNModel(num_classes=5)
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
#
#     # Predict the race
#     with torch.no_grad():
#         outputs = model(image)
#         _, predicted = torch.max(outputs, 1)
#
#     # Get the predicted race
#     race = race_mapping[predicted.item()]
#     return race
#
#
# # Test the function with an example image
# if __name__ == "__main__":
#     model_path = "../models/utkface_model.pth"
#     test_image_path = "../test/1_0_2_20161219140530307.jpg"  # Replace with your test image path
#
#     predicted_race = predict_race(test_image_path, model_path)
#     print(f"The predicted race is: {predicted_race}")
#
#     # Optionally display the image
#     image = Image.open(test_image_path)
#     plt.imshow(image)
#     plt.title(f"Predicted Race: {predicted_race}")
#     plt.axis('off')
#     plt.show()

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import PretrainedCNNModel


# Define transformations for the input image



import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import PretrainedCNNModel




race_mapping = {
    0: "White",
    1: "Black",
    2: "Asian",
    3: "Indian",
    4: "Others"
}
# Define transformations for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to (224, 224)
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize images
])


# Function to predict the race of input images
def predict_races(image_paths, model_path):
    images = []

    # Load and transform the images
    for image_path in image_paths:
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        images.append(image)

    # Stack the images into a batch
    images = torch.stack(images)

    # Load the model
    model = PretrainedCNNModel(num_classes=len(race_mapping))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Predict the races
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    # Get the predicted races
    predicted_races = [race_mapping[p.item()] for p in predicted]
    return predicted_races


# Test the function with example images
if __name__ == "__main__":
    model_path = "../models/utkface_model.pth"
    test_image_paths = ['../test/ab.jpg', '../test/african.jpg', '../test/img.png', '../test/white.jpg','../test/20200525042959_IMG_0882.JPG']  # Replace with your test image paths

    predicted_races = predict_races(test_image_paths, model_path)

    # Display the images and their predictions
    fig, axes = plt.subplots(1, len(test_image_paths), figsize=(15, 5))
    for idx, image_path in enumerate(test_image_paths):
        image = Image.open(image_path)
        axes[idx].imshow(image)
        axes[idx].set_title(f"Predicted Race: {predicted_races[idx]}")
        axes[idx].axis('off')

    plt.show()
