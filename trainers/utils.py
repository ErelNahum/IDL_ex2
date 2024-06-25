from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from functools import cache
import torch

TRAIN_DATA_SIZE = 100


def create_data_loaders(train_batch_size, test_batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize the images to [-1, 1] range
    ])
    train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_dataset = datasets.MNIST("data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

    return train_loader, test_loader


def plot_train_test_loss(train_losses,
                         test_losses,
                         ylabel):
    plt.plot(test_losses, 'r', label='test loss')
    plt.plot(train_losses, 'b', label='train loss')
    plt.xlabel('Epochs (#)')
    plt.ylabel(ylabel)
    plt.legend(loc="upper right")
    plt.title('Train/Test Loss Over Epochs')
    plt.show()


def plot_train_test_accuracy(train_accuracy,
                             test_accuracy):
    plt.plot(train_accuracy, 'r', label='train accuracy')
    plt.plot(test_accuracy, 'b', label='test accuracy')
    plt.xlabel('Epochs (#)')
    plt.ylabel('Accuracy')
    plt.legend(loc="lower right")
    plt.title('Train/Test Accuracy Over Epochs')
    plt.show()


def visualize_reconstructions(encoder_model,
                              fc_model,
                              ifc_model,
                              decoder_model,
                              device,
                              test_loader,
                              num_images=10):
    examples = pick_images(test_loader, device)
    with torch.no_grad():
        reconstructions = decoder_model(ifc_model(fc_model(encoder_model(examples))))
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


@cache
def pick_images(test_loader, device, num_labels=10):
    images_dict = dict()
    for images, labels in test_loader:
        for image, label in zip(images, labels):
            label = label.item()  # Convert tensor to integer
            if label not in images_dict:
                images_dict[label] = image
            if len(images_dict) == num_labels:
                break

    # Sort images by label and stack them into a tensor
    sorted_images = [images_dict[i] for i in range(num_labels)]
    return torch.stack(sorted_images).to(device)


def get_small_train_dataloader(batch_size: int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize the images to [-1, 1] range
    ])

    train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    indices = torch.arange(TRAIN_DATA_SIZE)
    train_loader_CLS = Subset(train_dataset, indices)
    return torch.utils.data.DataLoader(train_loader_CLS,
                                       batch_size=batch_size, shuffle=True, num_workers=0)
