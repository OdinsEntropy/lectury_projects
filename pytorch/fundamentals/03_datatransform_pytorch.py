import torch
import torchvision

# Torch expects classes to be used as tranformations, when using the torchvision.transforms.Compose() function
# and calling Multfransforma() this will create an object which calls init and the composer will call __call__ method

class Multransforma():
    def __init__(self, mult_factor):
        self.mult_factor = mult_factor

    # This will only be the input data as the mnist dataset processes y labels with own transform called target_transform
    def __call__(self, sample):
        inputs = sample
        return self.mult_factor*inputs


transforma = torchvision.transforms.Compose([Multransforma(2), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,), (0.5,))])

dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforma)

print(dataset[0])

