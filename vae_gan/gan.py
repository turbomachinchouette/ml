import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

# set random seed
torch.manual_seed(42)
np.random.seed(42)

# Global state
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
print(f"Using device {DEVICE}")

# Load MNIST, normalize [0, 1]
transform_mnist = transforms.Compose([transforms.ToTensor()])
train_mnist = torchvision.datasets.MNIST(
    root="~/datasets", train=True, download=True, transform=transform_mnist
)
test_mnist = torchvision.datasets.MNIST(
    root="~/datasets", train=False, download=True, transform=transform_mnist
)

# REUSE cVAE
# Ensure tanh activation

# Implement the discriminator

# Inputs image tensor, arch Convolutional, Leaky ReLu, output Scalar
# and intermediate feature layer for feature matching 𝓛.
