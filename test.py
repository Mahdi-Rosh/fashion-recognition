import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import random

from utils import get_device
from utils import show_sample
from nn import NeuralNetwork

test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=ToTensor())

model = NeuralNetwork().to(get_device())
model.eval()
model.load_state_dict(torch.load("model.pth"))

while True:
  show_sample(random.choice(test_data), model)