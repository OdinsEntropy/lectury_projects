import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import wandb
from torch.utils.data import Dataset
from helper.GPUDataset import ConvertToGPUDataset

wandb.init(project="CIFAR10-classification_tranfered", mode="offline")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyperparameters
num_epochs = 50
batch_size = 256
learning_rate = 0.01

# Methods to transfer learning
# A Full retrain, doesnt take long and is the most accurate
tf_methods = 'retrain_full_model' #retrain_new_layer, retrain_new_fc_layer, retrain_last_layer, retrain_full_model



# Dataset

transform_pipe = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

pre_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_pipe)
pre_val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_pipe)

train_dataset = ConvertToGPUDataset(pre_train_dataset, device)
val_dataset = ConvertToGPUDataset(pre_val_dataset, device)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)




# create string classes to verify output
classes = tuple(pre_train_dataset.classes)

# Model

model = torchvision.models.resnet18(weights='DEFAULT')

###### Different modalities of transfer learning
# 1. Add a new layer behind the original model
# Freeze all layers
for param in model.parameters():
    param.requires_grad = False



match tf_methods:

    # 1. Train a new layer from 1000 -> 10 
    case 'retrain_new_layer':
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            model.fc,  # Keep the original fully connected layer
            nn.ReLU(),
            nn.Linear(1000, 10)  # Downsample to 10 classes
        )
    # 2. Train a new layer and the last layer of the original model
    case 'retrain_new_fc_layer':
        # Freeze all layers except the final fully connected layer
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 1000),  # Replace the original fully connected layer
            nn.ReLU(),
            nn.Linear(1000, 10)  # Downsample to 10 classes
        )
    # 3. Retrain the last layer of the original model
    case 'retrain_last_layer':
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 10)  # Downsample to 10 classes

    # 4. Retrain whole model
    case 'retrain_full_model':
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(num_features, 1000),  # Replace the original fully connected layer
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1000, 10)  # Downsample to 10 classes
        )
        for param in model.parameters():
            param.requires_grad = True


model = model.to(device) # Do this after modifying the model

# Count trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Trainable parameters: {trainable_params}')

# Evaluation
@torch.no_grad()
def evaluate():
    model.eval()

    correct = 0
    total = 0
    val_loss = 0

    for images, labels in val_loader:
        #images = images.to(device, non_blocking=True)
        #labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1) # Get the label with highest probability
        total += labels.size(0) # Increment total by batch size
        correct += (predicted == labels).sum().item() # Increment correct by number of correct predictions
        val_loss += loss_function(outputs, labels).item()

    val_loss /= len(val_loader)
    accuracy = correct / len(val_loader.dataset)
    wandb.log({"val_loss": val_loss, "accuracy": accuracy})
    print(f'Validation Loss: {val_loss:.2f} Accuracy: {accuracy:.2f}')
    model.train() # return to training mode

# Object instanciation

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    
    if epoch % 5 == 0 :
        evaluate()

    # Wand logger
    running_loss = 0.0
    running_accuracy = 0

    for images, labels in train_loader:
        #images = images.to(device)
        #labels = labels.to(device)

        outputs = model(images)
        loss = loss_function(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_accuracy += (torch.argmax(outputs, 1) == labels).sum().item()



    # Log metrics to wandb after each epoch
    wandb.log({"epoch": epoch + 1, "train_loss": running_loss / len(train_loader)})
    print(f'Epoch {epoch+1}, Loss: {(running_loss / len(train_loader)):.2f} Accuracy: {(running_accuracy / len(train_loader.dataset)):.2f}')
evaluate()  # Evaluate the model after training


torch.save(model.state_dict(), f"{tf_methods}_cifa_model_pretrained.pth")