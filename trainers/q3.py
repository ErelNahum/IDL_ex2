import numpy as np
import torch
from matplotlib import pyplot as plt

from tqdm import tqdm
from models.decoder import Decoder
from models.encoder import Encoder
from models.fc_layer import FCLayer
from models.ifc_layer import InverseFCLayer
from models.latent_classifier import Classifier
from trainers.utils import plot_train_test_loss, create_data_loaders

LATENT_DIMENSION = 10

TRAIN_BATCH_SIZE = 30
TEST_BATCH_SIZE = 30
NUMBER_OF_EPOCHS = 40

device = torch.device("cpu")


def train(dataloader, encoder_model, fc_model, latent_classifier, ifc_model, decoder_model, loss_fn, optimizer):
    ifc_model.train()
    decoder_model.train()

    for batch_idx, (images, labels) in tqdm(enumerate(dataloader),
                                            total=len(dataloader),
                                            desc='Training'):
        # Compute prediction error
        reconstructed_images = decoder_model(ifc_model(latent_classifier(fc_model(encoder_model(images)))))
        loss = loss_fn(reconstructed_images, images)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test(dataloader, encoder_model, fc_model, latent_classifier, ifc_model, decoder_model, loss_fn):
    encoder_model.eval()
    fc_model.eval()
    ifc_model.eval()
    decoder_model.eval()
    latent_classifier.eval()

    total_loss = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader,
                                   total=len(dataloader),
                                   desc='Calculating Loss'):
            recunstructed_images = decoder_model(ifc_model(latent_classifier(fc_model(encoder_model(images)))))
            total_loss += loss_fn(recunstructed_images, images).item()

    total_loss /= len(dataloader)
    print(total_loss)
    return total_loss


def main():
    trainloader, testloader = create_data_loaders()

    encoder_model = Encoder().to(device)
    fc_model = FCLayer().to(device)
    latent_classifier = Classifier().to(device)
    ifc_model = InverseFCLayer(LATENT_DIMENSION).to(device)
    decoder_model = Decoder().to(device)
    old_ifc = InverseFCLayer().to(device)
    old_decoder = Decoder().to(device)

    load_trained_models(encoder_model, fc_model, latent_classifier, old_ifc, old_decoder)

    optimizer = torch.optim.Adam(
        list(ifc_model.parameters())
        + list(decoder_model.parameters())
    )

    loss_fn = torch.nn.L1Loss()
    train_loss_over_epochs = []
    test_loss_over_epochs = []
    for epoch_number in range(NUMBER_OF_EPOCHS):
        print(f'Epoch#: {epoch_number + 1}\n------------')
        train(trainloader, encoder_model, fc_model, ifc_model, decoder_model, loss_fn, optimizer)
        train_loss_over_epochs.append(
            test(trainloader, encoder_model, fc_model, latent_classifier, ifc_model, decoder_model, loss_fn)
        )
        test_loss_over_epochs.append(
            test(testloader, encoder_model, fc_model, latent_classifier, ifc_model, decoder_model, loss_fn)
        )

    plot_train_test_loss(train_loss_over_epochs, test_loss_over_epochs, 'L1 Loss')

    plot_images_arrays(decoder_model, encoder_model, fc_model, ifc_model,
                       latent_classifier, old_decoder, old_ifc, testloader)

    torch.save(ifc_model.state_dict(), 'weights/classifier_ifc.pth')
    torch.save(decoder_model.state_dict(), 'weights/classifier_decoder.pth')


def plot_images_arrays(decoder_model, encoder_model, fc_model, ifc_model, latent_classifier, old_decoder, old_ifc,
                       testloader):
    examples = get_random_images(testloader)
    reconstructions_array_plot(encoder_model, fc_model, ifc_model, decoder_model, examples,
                               latent_classifier, title='Classifier Decoder')
    reconstructions_array_plot(encoder_model, fc_model, old_ifc, old_decoder, examples)


def load_trained_models(encoder_model, fc_model, latent_classifier, old_ifc, old_decoder):
    encoder_model.load_state_dict(torch.load('weights/encoder.pth'))
    fc_model.load_state_dict(torch.load('weights/fc.pth'))
    latent_classifier.load_state_dict(torch.load('weights/latent_classifier.pth'))
    old_ifc.load_state_dict(torch.load('weights/ifc.pth'))
    old_decoder.load_state_dict(torch.load('weights/decoder.pth'))


def reconstructions_array_plot(encoder_model,
                               fc_model,
                               ifc_model,
                               decoder_model,
                               examples,
                               classifier_model=None,
                               num_images=50,
                               images_per_row=10,
                               title='Regular Decoder'):
    with torch.no_grad():
        if classifier_model is None:
            reconstructions = decoder_model(ifc_model(fc_model(encoder_model(examples))))
        else:
            reconstructions = decoder_model(ifc_model(classifier_model(fc_model(encoder_model(examples)))))
    fig, axes = plt.subplots(nrows=num_images // images_per_row, ncols=images_per_row, figsize=(num_images, 4))
    for i in range(num_images):
        axes[i // images_per_row, i % images_per_row].imshow(reconstructions[i].cpu().numpy().squeeze(), cmap='gray')
        axes[i // images_per_row, i % images_per_row].axis('off')
    axes[1, 0].set_title(title)
    plt.tight_layout()
    plt.show()
    plt.savefig('q3_' + title + '.png')


def get_random_images(test_loader, num_images: int = 50):
    total_samples = len(test_loader.dataset)
    indices = np.random.choice(total_samples, num_images, replace=False)
    return torch.stack([image for image, _ in test_loader.dataset[indices]])
