import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

from digit_recognition import DigitClassifier  

model = DigitClassifier()
model.load_state_dict(torch.load('digit_model.pth'))
model.eval()

# Preprocess your image (28x28, grayscale)
# convert to grey scale before doing the training
def process_image(img_path):
    image = Image.open(img_path).convert('L')  
    image = image.resize((28, 28))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = transform(image).unsqueeze(0)
    return image

# Predict
img_tensor = process_image("paperpic.jpg")  
output = model(img_tensor)
_, predicted = torch.max(output, 1)

print("Predicted Digit:", predicted.item())
