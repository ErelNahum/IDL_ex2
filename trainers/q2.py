import torch
from models.encoder import Encoder
from models.fc_layer import FCLayer
from models.latent_classifier import Classifier
from tqdm import tqdm
from utils import create_data_loaders, plot_train_test_loss, plot_train_test_accuracy

TRAIN_BATCH_SIZE = 30
TEST_BATCH_SIZE = 30
NUMBER_OF_EPOCHS = 20

device = torch.device("cpu")

def train(dataloader, encoder_model, fc_model, classifier_model, loss_fn, optimizer):
    encoder_model.train()
    fc_model.train()
    classifier_model.train()

    for images, labels in tqdm(dataloader,
                                total=len(dataloader),
                                desc='Training'):

        # Compute prediction error
        preds = classifier_model(fc_model(encoder_model(images)))
        labels = torch.nn.functional.one_hot(labels, 10).type(torch.float32)
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
            labels = torch.nn.functional.one_hot(labels, 10).type(torch.float32)
            total_loss += loss_fn(preds, labels).item()
            
            labels = torch.argmax(labels, dim=1)
            preds = torch.argmax(preds, dim=1)
            total_successes += (labels == preds).sum().item()
    
    total_loss /= len(dataloader)
    total_successes /= len(dataloader.dataset)
    print(total_loss, total_successes)
    return total_loss, total_successes


def main():
    trainloader, testloader = create_data_loaders(TRAIN_BATCH_SIZE, TEST_BATCH_SIZE)

    encoder_model = Encoder().to(device)
    fc_model = FCLayer().to(device)
    classifier_model = Classifier().to(device)

    optimizer = torch.optim.Adam(
        list(encoder_model.parameters())
        + list(fc_model.parameters())
        + list(classifier_model.parameters())
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    train_loss_over_epochs = []
    test_loss_over_epochs = []
    train_accuracy_over_epochs = []
    test_accuracy_over_epochs = []
    for epoch_number in range(NUMBER_OF_EPOCHS):
        print(f'Epoch#: {epoch_number + 1}\n------------')
        train(trainloader, encoder_model, fc_model,classifier_model, loss_fn, optimizer)
        train_loss, train_accuracy = test(trainloader, encoder_model, fc_model,classifier_model, loss_fn)
        test_loss, test_accuracy = test(testloader, encoder_model, fc_model,classifier_model, loss_fn)
        train_loss_over_epochs.append(train_loss)
        train_accuracy_over_epochs.append(train_accuracy)
        test_loss_over_epochs.append(test_loss)
        test_accuracy_over_epochs.append(test_accuracy)
    
    plot_train_test_loss(train_loss_over_epochs, test_loss_over_epochs, 'Cross Entropy Loss')
    plot_train_test_accuracy(train_accuracy_over_epochs, test_accuracy_over_epochs)

    torch.save(encoder_model.state_dict(), 'weights/q2/encoder.pth')
    torch.save(fc_model.state_dict(), 'weights/q2/fc.pth')
    torch.save(classifier_model.state_dict(), 'weights/q2/classifier.pth')




if __name__ == '__main__':
    main()

