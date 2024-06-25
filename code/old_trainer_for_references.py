import torch
from models.model import AutoEncoder
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.nn import Module
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from tqdm import tqdm

# CUDA for Pytorch
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
    device = torch.device("cpu")

else:
    device = torch.device("mps")
    device = torch.device("cpu")

print(device)

BATCH_SIZE = 30
NUMBER_OF_EPOCHS = 50
DEBUG = True


def train(dataloader: DataLoader,
          model: Module,
          loss_fn: Module,
          optimizer: torch.optim.Optimizer,
          epoch_number: int
          ) -> None:
    model.train()
    for batch, (X, y) in tqdm(enumerate(dataloader), desc=str(epoch_number)):
        X = X.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, X)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


# def loss_test(dataloader: DataLoader,
#               model: Module,
#               loss_fn: Module
#               ) -> Tuple[float, float]:
#     model.eval()
#     test_loss, correct, total = 0, 0, 0
#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
#             pred = model(X)
#             test_loss += loss_fn(pred, y).item()
#             predicted = (pred > 0.5).float()
#             correct += (predicted == y).sum().item()
#             total += y.size(0)
#     avg_test_loss = test_loss / len(dataloader)
#     accuracy = correct / total

#     return avg_test_loss, accuracy


def epoch(train_dataloader: DataLoader,
          test_dataloader: DataLoader,
          model: Module,
          loss_fn: Module,
          optimizer: torch.optim.Optimizer,
          epoch_num: int
          ) -> Tuple[float, float, float, float]:
    train(train_dataloader, model, loss_fn, optimizer, epoch_num)
    if epoch_num % 5 == 0:
        visualize_reconstructions(model, test_dataloader)

    # train_loss, train_accuracy = loss_test(train_dataloader, model, loss_fn)
    # test_loss, test_accuracy = loss_test(test_dataloader, model, loss_fn)
    # return train_loss, train_accuracy, test_loss, test_accuracy

# Visualize some reconstructions
def visualize_reconstructions(model, test_loader, num_images=10):
    model.eval()
    images, _ = next(iter(test_loader))

# Select the first 10 images from the batch
    examples = images[:num_images]
    examples = examples.to(device)
    print(examples.size())
    with torch.no_grad():
        reconstructions = model(examples)
    fig, axes = plt.subplots(nrows=2, ncols=num_images, figsize=(num_images * 2, 4))
    for i in range(num_images):
        axes[0, i].imshow(examples[i].cpu().numpy().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstructions[i].cpu().numpy().squeeze(), cmap='gray')
        axes[1, i].axis('off')
    axes[0, 0].set_title('Original')
    axes[1, 0].set_title('Reconstructed')
    plt.tight_layout()
    plt.show()


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
    train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = datasets.MNIST("data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True)

    model = AutoEncoder()
    model = model.to(device)
    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters())

    epochs = range(NUMBER_OF_EPOCHS)
    for epoch_index in epochs:
        print(f'epoch #{epoch_index}')
        epoch(
            train_loader, test_loader, model, loss_fn, optimizer, epoch_index
        )
        # train_losses.append(train_loss)
        # test_losses.append(test_loss)
        # train_accuracies.append(train_accuracy)
        # test_accuracies.append(test_accuracy)

        # if DEBUG:
        #     print('-------------------')
        #     print(f'Epoch {epoch_index} is complete.')
        #     print('\tTrain\tTest')
        #     print(f'Loss\t{train_loss}\t{test_loss}')
        #     print(f'Accuracy\t{train_accuracy}\t{test_accuracy}')


    # plot_train_test_loss(train_losses, test_losses)
    # plot_train_test_accuracy(train_accuracies, test_accuracies)
    # torch.save(model.state_dict(), '../models/2d_model.pth')


if __name__ == '__main__':
    main()
