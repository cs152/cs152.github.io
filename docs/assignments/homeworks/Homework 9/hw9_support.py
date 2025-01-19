import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchsummary import summary

from torchvision.datasets import FashionMNIST, MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

from fastprogress.fastprogress import master_bar, progress_bar

import matplotlib.pyplot as plt

# Use the GPUs if they are available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using '{device}' device.")

# Mini-Batch SGD hyperparameters
batch_size = 256
num_epochs = 10
learning_rate = 0.001

criterion = nn.CrossEntropyLoss()

def get_fmnist_data_loaders(path, batch_size, valid_batch_size=0):
    # Computing normalization constants for Fashion-MNIST (commented out since we only need to do this once)
    # train_loader, valid_loader = get_fmnist_data_loaders(data_path, 0)
    # X, _ = next(iter(train_loader))
    # s, m = torch.std_mean(X)


    # Data specific transforms
    data_mean = (0.2860,)
    data_std = (0.3530,)
    xforms = Compose([ToTensor(), Normalize(data_mean, data_std)])

    # Training data loader
    train_dataset = FashionMNIST(root=path, train=True, download=True, transform=xforms)

    # Set the batch size to N if batch_size is 0
    tbs = len(train_dataset) if batch_size == 0 else batch_size
    train_loader = DataLoader(train_dataset, batch_size=tbs, shuffle=True)

    # Validation data loader
    valid_dataset = FashionMNIST(root=path, train=False, download=True, transform=xforms)

    # Set the batch size to N if batch_size is 0
    vbs = len(valid_dataset) if valid_batch_size == 0 else valid_batch_size
    valid_loader = DataLoader(valid_dataset, batch_size=vbs, shuffle=True)

    return train_loader, valid_loader

from torch.optim import Adam

# Here we'll define a function to train and evaluate a neural network with a specified architecture
# using a specified optimizer.
def run_model(data_path, model, optimizer=Adam, 
              learning_rate=0.001):
    
    # Get the dataset
    train_loader, valid_loader = get_fmnist_data_loaders(data_path, batch_size)
    return gradient_descent(model, train_loader, valid_loader, optimizer, learning_rate)

def gradient_descent(model, train_loader, valid_loader, optimizer=Adam, learning_rate=0.001):

    # Do model creation here so that the model is recreated each time the cell is run
    model = model.to(device)

    t = 0
    # Create the optimizer, just like we have with the built-in optimizer
    opt = optimizer(model.parameters(), learning_rate)

    # A master bar for fancy output progress
    mb = master_bar(range(num_epochs))

    # Information for plots
    mb.names = ["Train Loss", "Valid Loss"]
    train_losses = []
    valid_losses = []

    for epoch in mb:

        #
        # Training
        #
        model.train()

        train_N = len(train_loader.dataset)
        num_train_batches = len(train_loader)
        train_dataiterator = iter(train_loader)

        train_loss_mean = 0

        for batch in progress_bar(range(num_train_batches), parent=mb):

            # Grab the batch of data and send it to the correct device
            train_X, train_Y = next(train_dataiterator)
            train_X, train_Y = train_X.to(device), train_Y.to(device)

            # Compute the output
            train_output = model(train_X)

            # Compute loss
            train_loss = criterion(train_output, train_Y)

            num_in_batch = len(train_X)
            tloss = train_loss.item() * num_in_batch / train_N
            train_loss_mean += tloss
            train_losses.append(train_loss.item())

            # Compute gradient
            model.zero_grad()
            train_loss.backward()
            
            # Take a step of gradient descent
            t += 1
            with torch.no_grad():
                opt.step()

        #
        # Validation
        #
        model.eval()

        valid_N = len(valid_loader.dataset)
        num_valid_batches = len(valid_loader)

        valid_loss_mean = 0
        valid_correct = 0

        with torch.no_grad():

            # valid_loader is probably just one large batch, so not using progress bar
            for valid_X, valid_Y in valid_loader:

                valid_X, valid_Y = valid_X.to(device), valid_Y.to(device)

                valid_output = model(valid_X)

                valid_loss = criterion(valid_output, valid_Y)

                num_in_batch = len(valid_X)
                vloss = valid_loss.item() * num_in_batch / valid_N
                valid_loss_mean += vloss
                valid_losses.append(valid_loss.item())

                # Convert network output into predictions (one-hot -> number)
                predictions = valid_output.argmax(1)

                # Sum up total number that were correct
                valid_correct += (predictions == valid_Y).type(torch.float).sum().item()

        valid_accuracy = 100 * (valid_correct / valid_N)

        # Report information
        tloss = f"Train Loss = {train_loss_mean:.4f}"
        vloss = f"Valid Loss = {valid_loss_mean:.4f}"
        vaccu = f"Valid Accuracy = {(valid_accuracy):>0.1f}%"
        mb.write(f"[{epoch+1:>2}/{num_epochs}] {tloss}; {vloss}; {vaccu}")

        # Update plot data
        max_loss = max(max(train_losses), max(valid_losses))
        min_loss = min(min(train_losses), min(valid_losses))

        x_margin = 0.2
        x_bounds = [0 - x_margin, num_epochs + x_margin]

        y_margin = 0.1
        y_bounds = [min_loss - y_margin, max_loss + y_margin]

        valid_Xaxis = torch.linspace(0, epoch + 1, len(train_losses))
        valid_xaxis = torch.linspace(1, epoch + 1, len(valid_losses))
        graph_data = [[valid_Xaxis, train_losses], [valid_xaxis, valid_losses]]

        mb.update_graph(graph_data, x_bounds, y_bounds)

    print(f"[{epoch+1:>2}/{num_epochs}] {tloss}; {vloss}; {vaccu}")