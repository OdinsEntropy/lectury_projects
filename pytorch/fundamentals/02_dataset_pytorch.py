import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np

# Default structure for a dataset 
class WineDataset(Dataset):

    def __init__(self):
        # Initialize data, download, etc.
        # read with numpy or pandas
        data_url = 'https://raw.githubusercontent.com/patrickloeber/pytorchTutorial/refs/heads/master/data/wine/wine.csv'

        xy = np.genfromtxt(data_url, delimiter=',', dtype=np.float32, skip_header=1)   
        self.n_samples = xy.shape[0]

        # here the first column is the class label, the rest are the features
        self.x_data = torch.from_numpy(xy[:, 1:]) # size [n_samples, n_features]
        self.y_data = torch.from_numpy(xy[:, [0]]) # size [n_samples, 1]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples



# create dataset
# dataset = WineDataset()
dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=torchvision.transforms.ToTensor(), download=True) 
dataloader = DataLoader(dataset=dataset, batch_size=7, shuffle=True)

num_epochs = 5

for epoch in range(num_epochs):
    for inputs, output in dataloader:

        print(inputs.shape, output.shape)
        inputs.squeeze()
