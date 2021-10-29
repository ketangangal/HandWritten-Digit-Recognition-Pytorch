import os
import torch.nn as nn
import torch
from src.utils.data_mgmt import load_dataset, create_loader, save_plot, save_model
from src.utils.model import NeuralNetwork
from src.utils.common import read_config
from logger.logger import Logger


def training(config):
    # Load Dataset and create Loader
    logs.info(log_type='INFO', log_message='Training Started')
    train_data, test_data = load_dataset(config=config)
    logs.info(log_type='INFO', log_message='Checking and Fetching Data')
    train_data_loader, test_data_loader = create_loader(train_data, test_data, config=config)
    logs.info(log_type='INFO', log_message='Data Loaders Created')

    # Model Creation
    model = NeuralNetwork(config['param']['input_size'], config['param']['output_size'])
    logs.info(log_type='INFO', log_message=f'Model Created {model}')

    # Error and Optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    logs.info(log_type='INFO', log_message='Loss Function and Optimizer Created')

    # Device Selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logs.info(log_type='INFO', log_message='Checking and Fetching Data')

    model.to(device)
    n_epoch = config['param']['n_epoch']
    loss_ = []
    acc_ = []

    # Forward and Backward pass
    for epoch in range(n_epoch):
        print(f'Epoch {epoch + 1}/{n_epoch} ')
        logs.info(log_type='INFO', log_message=f'Epoch {epoch + 1}/{n_epoch} ')
        for batch, data in enumerate(train_data_loader):
            x = data[0].to(device)
            y = data[1].to(device)

            optimizer.zero_grad()

            y_pred = model(x.reshape(-1, 784))
            loss = criterion(y_pred, y)
            loss_.append(loss.item())
            loss.backward()
            optimizer.step()

            accuracy = torch.mean((torch.argmax(y_pred, 1) == y).float()).item()
            acc_.append(accuracy)

            if batch % 100 == 0:
                print(f'   Batch {batch} Loss {loss.item():.4f} Accuracy {accuracy:.4f}')
                logs.info(log_type='INFO',
                          log_message=f'   Batch {batch} Loss {loss.item():.4f} Accuracy {accuracy:.4f}')

    # Save plot into artifacts
    path = os.path.join(config['directory']['main'], config['directory']['plot'])
    save_plot(loss_, acc_, 'plot', path)
    logs.info(log_type='INFO', log_message='Plot Saved')

    # Test on test data
    test_loss = 0
    test_accuracy = 0
    total = 0
    with torch.no_grad():
        for batch, data in enumerate(test_data_loader):
            x = data[0].to(device)
            y = data[1].to(device)

            y_pred = model(x.reshape(-1, 784))
            loss = criterion(y_pred, y)
            test_loss += loss.item()
            test_accuracy += torch.mean((torch.argmax(y_pred, 1) == y).float()).item()
            total += 1

    # log Results
    print(f'Test Loss : {test_loss / total:.4f} Test_accuracy : {test_accuracy / total:.4f}')
    logs.info(log_type='INFO',
              log_message=f'Test Loss : {test_loss / total:.4f} Test_accuracy : {test_accuracy / total:.4f}')

    # Save Model
    path = os.path.join(config['directory']['main'], config['directory']['model_dir'])
    save_model(model, 'model', path)
    logs.info(log_type='INFO', log_message='Model Saved')


if __name__ == '__main__':
    config = read_config(r'C:\Users\ketan\Desktop\DeepLearning\HandWritten-Digit-Recognition-Pytorch\config.yaml')
    logs = Logger(os.path.join(config['directory']['main'], config['directory']['logs']))
    training(config=config)
