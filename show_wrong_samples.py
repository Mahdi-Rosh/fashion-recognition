import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

from utils import show_sample
from utils import get_wrong_guesses
from utils import get_device
from nn import NeuralNetwork

test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=ToTensor())

model = NeuralNetwork().to(get_device())
model.eval()
model.load_state_dict(torch.load("model.pth"))

wrong_samples = get_wrong_guesses(test_data, model)
print(f'{len(wrong_samples)} Wrong guesses')

for s in wrong_samples:
  show_sample(s, model)