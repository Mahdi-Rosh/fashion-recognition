import torch
from matplotlib import pyplot as plt
from nn import NeuralNetwork

def get_device():
  return get_device.device

get_device.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def show_sample(sample: tuple[any, any], model):
  x,y = sample[0], sample[1]
  with torch.no_grad():
    x = x.to(get_device())
    pred = model(x)
    pred,actual = classes[pred[0].argmax(0)], classes[y]
    
    plt.imshow(x.squeeze(), cmap='gray')
    plt.title(f'Predicted {pred}\nActual {actual}')
    plt.show()

def get_wrong_guesses(samples, model: NeuralNetwork):
  results = []
  for sample in samples:
    x,y = sample
    with torch.no_grad():
      x = x.to(get_device())
      pred = model(x)
      pred,actual = classes[pred[0].argmax(0)], classes[y]
      if pred != actual:
        results.append(sample)
  return results

def save_model(model: NeuralNetwork):
  torch.save(model.state_dict(), 'model.pth')