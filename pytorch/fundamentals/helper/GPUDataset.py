import torch
from torch.utils.data import Dataset

# Define a Helper Function which converts a dataset to a GPU dataset
class GPUDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def ConvertToGPUDataset(DatasetToConvert, device):
    
    # convert data and labels to tensors
    transformed_data = [img.to(device) for img, label in DatasetToConvert]
    labels = torch.tensor(DatasetToConvert.targets).to(device)

    return GPUDataset(transformed_data, labels)






