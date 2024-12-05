import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from net import Net
from eval import train,  test

# Set device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

torch.manual_seed(1)
batch_size = 128

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.RandomAffine(
                           degrees=15,            # Rotate by ±15 degrees
                           translate=(0.1, 0.1), # Translate by ±10%
                           scale=(0.9, 1.1)      # Scale between 90% to 110%
                       ),
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)

# Initialize the model, loss function, and optimizer
model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5, verbose=True)


for epoch in range(1, 21):
  train(model, device, train_loader, optimizer, epoch)
  test(model, device, test_loader)
  scheduler.step()
  print(f"epoch {epoch} Done")

# Save the model in the specified path
torch.save(model.state_dict(), 'model/mnist_model.pth')