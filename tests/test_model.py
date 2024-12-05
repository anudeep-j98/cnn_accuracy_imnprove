import torch
import torch.nn as nn
from torchvision import datasets, transforms
from model.net import Net
import time

# Load MNIST dataset for testing
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ])

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=2, shuffle=False)

def test_model():
    model = Net()
    model.load_state_dict(torch.load('model/mnist_model.pth'))
    model.eval()

    # Test 1: Check number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print('\nChecking number of params')
    assert num_params < 20000, f"Model has {num_params} parameters, exceeds limit."

    # Test 2: Check accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = correct / total
    print(f'\nChecking trained model accuracy on test data which is {accuracy}')
    assert accuracy > 0.994, f"Model accuracy is {accuracy}, expected > 0.994."

    # Test 3: check if BN used or not
    has_batch_norm = any(isinstance(module, torch.nn.BatchNorm2d) 
                        for module in model.modules())
    print('\nBatch Normalization Test:')
    print('Batch Normalization layers found:', has_batch_norm)
    assert has_batch_norm, "Model must use Batch Normalization"

    # Test 4: check if Dropout is used or not
    dropout_layers = [module for module in model.modules() 
                     if isinstance(module, torch.nn.Dropout)]
    print('\nDropout Test:')
    print(f'Number of Dropout layers found: {len(dropout_layers)}')
    print(f'Dropout rates: {[layer.p for layer in dropout_layers]}')
    assert len(dropout_layers) > 0, "Model must use Dropout"

    #Test 5: Check if GAP AND ANN is used or not
    has_gap = any(isinstance(module, torch.nn.AdaptiveAvgPool2d) 
                  for module in model.modules())
    has_fc = any(isinstance(module, torch.nn.Linear) 
                 for module in model.modules())
    print('\nGAP/FC Layer Test:')
    print(f'Global Average Pooling found: {has_gap}')
    print(f'Fully Connected Layer found: {has_fc}')
    assert has_gap or has_fc, "Model must use either Global Average Pooling or Fully Connected Layer"

if __name__ == "__main__":
    test_model()