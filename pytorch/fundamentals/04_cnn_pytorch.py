import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import wandb


wandb.init(project="CIFAR10-classification", mode="offline")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyperparameters
num_epochs = 40
batch_size = 128
learning_rate = 0.001


# Dataset

transform_pipe = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_pipe)
val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_pipe)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# create string classes to verify output
classes = tuple(train_dataset.classes)

# Model

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # Input is 3x32x32
        # 6 "channels"(more or less matrizes/featurespaces) with size 28x28 
        # Non overlapping 2x2 pooling -> 6x14x14
        # 16 "channels" with size 10x10
        # Linear layer with 120 neurons
        # Linear layer with 84 neurons
        # Linear layer with 10 neurons (output)

        self.conv1 = nn.Conv2d(3, 6, 5) # (6x28x28) 
        self.pool = nn.MaxPool2d(2, 2) # (6x14x14) 
        self.conv2 = nn.Conv2d(6, 16, 5) # (16x10x10)
        self.fc1 = nn.Linear(16 * 5 * 5, 300) # flattening
        self.fc2 = nn.Linear(300, 120) # flattening
        self.fc3 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, 10)

        self.dropout = nn.Dropout(0.2)
        self.flatten = nn.Flatten()

    def forward(self, x):

        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = F.relu(self.fc1(self.flatten(x)))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x) # CrossEntropyLoss already applies softmax, in evaluation it would be required
        return x

# Evaluation
@torch.no_grad()
def evaluate():
    model.eval()

    correct = 0
    total = 0
    val_loss = 0

    for images, labels in val_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

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

model = ConvNet().to(device)

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
        images = images.to(device)
        labels = labels.to(device)

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



torch.save(model.state_dict(), "cifa_model.pth")



