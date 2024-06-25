import torch
import torch.utils.data as data_utils
from tqdm import tqdm
from torch import Dataloader
from models.encoder import Encoder
from models.decoder import Decoder
from models.fc_layer import FCLayer
from models.ifc_layer import InverseFCLayer
from torchvision import transforms, datasets
from utils import create_data_loaders, plot_train_test_loss, visualize_reconstructions

TRAIN_DATA_SIZE = 100

TRAIN_BATCH_SIZE = 30
TEST_BATCH_SIZE = 30
NUMBER_OF_EPOCHS = 40

device = torch.device("cpu")


def train(dataloader, encoder_model, fc_model, ifc_model, decoder_model, loss_fn, optimizer):
    encoder_model.train()
    fc_model.train()
    ifc_model.train()
    decoder_model.train()

    for images, labels in tqdm(dataloader,
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


def get_q4_train_dataloader(batch_size: int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize the images to [-1, 1] range
    ])

    train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    indices = torch.arange(TRAIN_DATA_SIZE)
    train_loader_CLS = data_utils.Subset(train_dataset, indices)
    return torch.utils.data.DataLoader(train_loader_CLS,
                                                   batch_size=batch_size, shuffle=True, num_workers=0)


def main():
    _, testloader = create_data_loaders(TRAIN_BATCH_SIZE, TEST_BATCH_SIZE)
    trainloader = get_q4_train_dataloader(TRAIN_BATCH_SIZE)

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

    plot_train_test_loss(train_loss_over_epochs, test_loss_over_epochs, 'L1 Loss')
    visualize_reconstructions(encoder_model, fc_model, ifc_model, decoder_model, device, testloader)

    torch.save(encoder_model.state_dict(), 'weights/q4/encoder.pth')
    torch.save(fc_model.state_dict(), 'weights/q4/fc.pth')
    torch.save(ifc_model.state_dict(), 'weights/q4/ifc.pth')
    torch.save(decoder_model.state_dict(), 'weights/q4/decoder.pth')
