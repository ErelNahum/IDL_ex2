import torch
from model import Encoder, Decoder, FCLayer, InverseFCLayer
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from functools import cache

TRAIN_BATCH_SIZE = 30
TEST_BATCH_SIZE = 30
NUMBER_OF_EPOCHS = 40

device = torch.device("cpu")


def create_data_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize the images to [-1, 1] range
    ])
    train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    test_dataset = datasets.MNIST("data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True)

    return train_loader, test_loader


def train(dataloader, encoder_model, fc_model, ifc_model, decoder_model, loss_fn, optimizer):
    encoder_model.train()
    fc_model.train()
    ifc_model.train()
    decoder_model.train()

    for batch_idx, (images, labels) in tqdm(enumerate(dataloader),
                                            total=len(dataloader),
                                            desc='Training'):

        # Compute prediction error
        recunstructed_images = decoder_model(ifc_model(fc_model(encoder_model(images))))
        loss = loss_fn(recunstructed_images, images)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test(dataloader, encoder_model, fc_model, ifc_model, decoder_model, loss_fn):
    encoder_model.eval()
    fc_model.eval()
    ifc_model.eval()
    decoder_model.eval()

    total_loss = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader,
                                   total=len(dataloader),
                                   desc='Calculating Loss'):
            recunstructed_images = decoder_model(ifc_model(fc_model(encoder_model(images))))
            total_loss += loss_fn(recunstructed_images, images).item()
    
    total_loss /= len(dataloader)
    print(total_loss)
    return total_loss



def plot_train_test_loss(train_losses, test_losses):
    plt.plot(test_losses, 'r', label='test loss')
    plt.plot(train_losses, 'b', label='train loss')
    plt.xlabel('Epochs (#)')
    plt.ylabel('L1 Loss')
    plt.legend(loc="upper right")
    plt.title('Train/Test Loss Over Epochs')
    plt.show()

def visualize_reconstructions(encoder_model,
                              fc_model,
                              ifc_model,
                              decoder_model,test_loader, num_images=10):
    
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

def main():
    trainloader, testloader = create_data_loaders()

    encoder_model = Encoder().to(device)
    fc_model = FCLayer().to(device)
    ifc_model = InverseFCLayer().to(device)
    decoder_model = Decoder().to(device)

    optimizer = torch.optim.Adam(
        list(encoder_model.parameters())
        + list(fc_model.parameters())
        + list(ifc_model.parameters())
        + list(decoder_model.parameters())
    )

    loss_fn = torch.nn.L1Loss()
    train_loss_over_epochs = []
    test_loss_over_epochs = []
    for epoch_number in range(NUMBER_OF_EPOCHS):
        print(f'Epoch#: {epoch_number + 1}\n------------')
        train(trainloader, encoder_model, fc_model, ifc_model, decoder_model, loss_fn, optimizer)
        train_loss_over_epochs.append(
            test(trainloader, encoder_model, fc_model, ifc_model, decoder_model, loss_fn)
        )
        test_loss_over_epochs.append(
            test(testloader, encoder_model, fc_model, ifc_model, decoder_model, loss_fn)
        )
    
    plot_train_test_loss(train_loss_over_epochs, test_loss_over_epochs)
    visualize_reconstructions(encoder_model, fc_model, ifc_model, decoder_model, testloader)

    torch.save(encoder_model.state_dict(), 'models/encoder.pth')
    torch.save(fc_model.state_dict(), 'models/fc.pth')
    torch.save(ifc_model.state_dict(), 'models/ifc.pth')
    torch.save(decoder_model.state_dict(), 'models/decoder.pth')


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


if __name__ == '__main__':
    main()

