import torch
from model import AutoEncoder
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.nn import Module
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List

# CUDA for Pytorch
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
torch.backends.cudnn.benchmark = True

BATCH_SIZE = 30
NUMBER_OF_EPOCHS = 20
DEBUG = True


def train(dataloader: DataLoader,
          model: Module,
          loss_fn: Module,
          optimizer: torch.optim.Optimizer,
          ) -> None:
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def loss_test(dataloader: DataLoader,
              model: Module,
              loss_fn: Module
              ) -> Tuple[float, float]:
    model.eval()
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            predicted = (pred > 0.5).float()
            correct += (predicted == y).sum().item()
            total += y.size(0)
    avg_test_loss = test_loss / len(dataloader)
    accuracy = correct / total

    return avg_test_loss, accuracy


def epoch(train_dataloader: DataLoader,
          test_dataloader: DataLoader,
          model: Module,
          loss_fn: Module,
          optimizer: torch.optim.Optimizer
          ) -> Tuple[float, float, float, float]:
    train(train_dataloader, model, loss_fn, optimizer)
    train_loss, train_accuracy = loss_test(train_dataloader, model, loss_fn)
    test_loss, test_accuracy = loss_test(test_dataloader, model, loss_fn)
    return train_loss, train_accuracy, test_loss, test_accuracy


def plot_train_test_loss(train_losses: List,
                         test_losses: List,
                         path: str = None):
    plt.plot(test_losses, 'r', label='test loss')
    plt.plot(train_losses, 'b', label='train loss')
    plt.xlabel('Epochs (#)')
    plt.ylabel('Binary Cross Entropy Loss')
    plt.legend(loc="upper right")
    plt.title('Train/Test Loss Over Epochs')

    plt.show()
    if path:
        plt.savefig(path)


def plot_train_test_accuracy(train_accuracies: List,
                             test_accuracies: List,
                             path: str = None):
    plt.plot(test_accuracies, 'r', label='test')
    plt.plot(train_accuracies, 'b', label='train')
    plt.xlabel('Epochs (#)')
    plt.ylabel('ACCURACY')
    plt.legend(loc="lower right")
    plt.title('Test/Train Accuracy Over Epochs')

    plt.show()
    if path:
        plt.savefig(path)


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize the images to [-1, 1] range
    ])
    train_dataloader = datasets.MNIST("data", train=True, download=True, transform=transform)
    test_dataloader = datasets.MNIST("data", train=False, download=True, transform=transform)

    model = AutoEncoder()
    loss_fn = torch.nn.L1Loss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer = torch.optim.Adam(model.parameters())
    train_losses = []
    test_losses = []
    # train_accuracies = []
    # test_accuracies = []

    epochs = range(NUMBER_OF_EPOCHS)
    for epoch_index in epochs:
        train_loss, train_accuracy, test_loss, test_accuracy = epoch(
            train_dataloader, test_dataloader, model, loss_fn, optimizer
        )
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        # train_accuracies.append(train_accuracy)
        # test_accuracies.append(test_accuracy)

        if DEBUG:
            print('-------------------')
            print(f'Epoch {epoch_index} is complete.')
            print('\tTrain\tTest')
            print(f'Loss\t{train_loss}\t{test_loss}')
            print(f'Accuracy\t{train_accuracy}\t{test_accuracy}')


    # plot_train_test_loss(train_losses, test_losses)
    # plot_train_test_accuracy(train_accuracies, test_accuracies)
    # torch.save(model.state_dict(), '../models/2d_model.pth')


if __name__ == '__main__':
    main()
