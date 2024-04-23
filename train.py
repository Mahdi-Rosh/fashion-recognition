import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import utils
from nn import NeuralNetwork

train_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=ToTensor())
test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=ToTensor())

print('Loaded data...')

batch_size = 128
train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

model = NeuralNetwork().to(utils.get_device())

def train(dataloader: DataLoader, model: NeuralNetwork, loss_fn: nn.CrossEntropyLoss, optimizer):
  model.train()
  size = len(dataloader.dataset)
  
  for batch,(x,y) in enumerate(dataloader):
    x,y = x.to(utils.get_device()), y.to(utils.get_device())
    pred = model(x)
    loss = loss_fn(pred, y)

    # Backpropogation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if batch % 100 == 0:
      loss, current = loss.item(), batch * batch_size + len(x)
      print(f"loss: {loss} [{current}/{size}]")

def test(dataloader: DataLoader, model: NeuralNetwork, loss_fn: nn.CrossEntropyLoss):
  model.eval()
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  test_loss, correct = 0,0

  with torch.no_grad():
    for x,y in dataloader:
      x,y = x.to(utils.get_device()), y.to(utils.get_device())
      pred = model(x)
      test_loss += loss_fn(pred, y).item()
      correct += (pred.argmax(1) == y).type(torch.float).sum().item()
  test_loss /= num_batches
  correct /= size
  print(f"Test error:\n accuracy {correct*100}%\n avg loss: {test_loss}\n")
  return correct

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
t = 0
correct = 0

print(f"Running on {utils.get_device()}")

try:
  while correct < 0.95:
    t += 1
    print(f"Epoch {t}\n")
    train(train_dataloader, model, loss_fn, optimizer)
    correct = test(test_dataloader, model, loss_fn)

  print('Saving model...')
  utils.save_model(model)
except KeyboardInterrupt:
  print('Keyboard interrupt, saving model...')
  utils.save_model(model)
  