import torch
import torch.nn as nn
from torchvision import transforms
from net import Net
from PIL import Image

# Load the trained model
def load_model(model_path):
    model = Net()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

# Function to preprocess input image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Function to make predictions
def predict(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output.data, 1)
    return predicted.item()

# Main deployment function
def deploy_model(model_path, image_path):
    model = load_model(model_path)
    input_tensor = preprocess_image(image_path)
    prediction = predict(model, input_tensor)
    print(f'Predicted class: {prediction}')

if __name__ == "__main__":
    # Example usage
    deploy_model('model/mnist_model.pth', 'path/to/sample_image.png')  # Replace with your image path