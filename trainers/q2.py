import torch
from models.encoder import Encoder
from models.fc_layer import FCLayer
from models.latent_classifier import Classifier
from tqdm import tqdm
from utils import create_data_loaders, plot_train_test_loss, visualize_reconstructions

TRAIN_BATCH_SIZE = 30
TEST_BATCH_SIZE = 30
NUMBER_OF_EPOCHS = 4

device = torch.device("cpu")

def train(dataloader, encoder_model, fc_model, classifier_model, loss_fn, optimizer):
    encoder_model.eval()
    fc_model.eval()
    classifier_model.train()

    for images, labels in tqdm(dataloader,
                                total=len(dataloader),
                                desc='Training'):

        # Compute prediction error
        preds = classifier_model(fc_model(encoder_model(images)))
        labels = torch.nn.functional.one_hot(labels, 10)
        loss = loss_fn(preds, labels)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test(dataloader, encoder_model, fc_model, classifier_model, loss_fn):
    encoder_model.eval()
    fc_model.eval()
    classifier_model.eval()

    total_loss = 0
    total_successes = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader,
                                   total=len(dataloader),
                                   desc='Calculating Loss'):
            preds = classifier_model(fc_model(encoder_model(images)))
            labels = torch.nn.functional.one_hot(labels, 10)
            total_loss += loss_fn(preds, labels).item()
            
            labels = torch.argmax(labels, dim=1)
            preds = torch.argmax(labels, dim=1)
            total_successes += (labels == preds).sum().item()
    
    total_loss /= len(dataloader)
    total_successes /= len(dataloader.dataset)
    print(total_loss, total_successes)
    return total_loss, total_successes





def main():
    trainloader, testloader = create_data_loaders(TRAIN_BATCH_SIZE, TEST_BATCH_SIZE)

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

    torch.save(encoder_model.state_dict(), 'weights/q1/encoder.pth')
    torch.save(fc_model.state_dict(), 'weights/q1/fc.pth')
    torch.save(ifc_model.state_dict(), 'weights/q1/ifc.pth')
    torch.save(decoder_model.state_dict(), 'weights/q1/decoder.pth')




if __name__ == '__main__':
    main()

