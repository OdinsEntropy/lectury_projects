import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import wandb

def main():
    # Initialize wandb
    wandb.init(project="mnist-classification")

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define a simple neural network
    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(28*28, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 10)
            )
        def forward(self, x):
            x = x.view(-1, 28*28)
            x = self.network(x)
            return x

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    # Load MNIST Train dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    train_x, train_y = batchpreload(train_dataset)
    train_targets_gpu = TensorDataset(train_x.to(device), train_y.to(device))

    train_loader = torch.utils.data.DataLoader(train_targets_gpu, batch_size=64, shuffle=True)


    # Load MNIST Validation dataset
    val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    val_x, val_y = batchpreload(val_dataset)
    val_targets_gpu = TensorDataset(val_x.to(device), val_y.to(device))

    val_loader = torch.utils.data.DataLoader(val_targets_gpu, batch_size=64, shuffle=True)

    # Create DataLoader with properly formatted tensors

    # print Dataset Infos
    print(f"Training dataset: {len(train_dataset)} Validation dataset: {len(val_dataset)}")

    # Initialize model, loss function, and optimizer
    model = SimpleNN().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Training loop
    for epoch in range(100):  # 10 epochs
        model.train()
        running_loss = 0.0
        for data, target in train_loader:

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Log metrics to wandb after each epoch
        wandb.log({"epoch": epoch + 1, "loss": running_loss / len(train_loader)})

        print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}')

        if epoch%10 == 0:
            evaluate(model, val_loader, criterion, device)

    # Save the model
    torch.save(model.state_dict(), "mnist_model.pth")
    # wandb.save("mnist_model.pth")

def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    wrong_images = []
    print(f"val size is {len(val_loader.dataset)}")
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Collect wrongly classified images
            for i in range(len(data)):
                if pred[i].item() != target[i].item() and len(wrong_images) < 8:
                    wrong_images.append(wandb.Image(data[i], caption=f"Pred: {pred[i].item()}, Label: {target[i].item()}"))

    val_loss /= len(val_loader.dataset)
    accuracy = 100. * correct / len(val_loader.dataset)
    wandb.log({"val_loss": val_loss, "accuracy": accuracy, "wrong_examples": wrong_images})

    print(f'Validation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(val_loader.dataset)} ({accuracy:.2f}%)')

def batchpreload(dataset):
    data_list = []
    target_list = []
    for data, target in dataset:
        data_list.append(data)
        target_list.append(target)
    return torch.stack(data_list), torch.tensor(target_list)

if __name__ == "__main__":
    main()