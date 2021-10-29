from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import pandas as pd
import time
import os


def load_dataset(config):
    path = os.path.join(config['directory']['main'],config['directory']['root_dir'])
    train_data = datasets.MNIST(
        root=path,
        train=True,
        transform=transforms.ToTensor(),
        target_transform=None,
        download=True
    )

    test_data = datasets.MNIST(
        root=path,
        train=False,
        transform=transforms.ToTensor(),
        target_transform=None,
        download=True
    )
    return train_data, test_data


def create_loader(train_data, test_data, config):

    train_data_loader = DataLoader(dataset=train_data,
                                   batch_size=config['param']['batch_size'],
                                   shuffle=config['param']['shuffle'])

    test_data_loader = DataLoader(dataset=test_data,
                                  batch_size=config['param']['batch_size'])
    return train_data_loader, test_data_loader


def get_unique_filename(filename, typ):
    if typ == 'Plot':
        unique_filename = time.strftime(f"{filename}._%Y_%m_%d_%H_%M.png")
        return unique_filename
    elif typ == 'Model':
        unique_filename = time.strftime(f"{filename}._%Y_%m_%d_%H_%M.pt")
        return unique_filename
    else:
        return None


def save_plot(loss, acc, name, path):
    unique_name1 = get_unique_filename(name, typ='Plot')
    path_to_plot1 = os.path.join(path, unique_name1)
    fig = pd.DataFrame(data={'Loss': loss,'Accuracy': acc}).plot()
    plt.xlabel('Accuracy')
    plt.ylabel('Loss')
    plt.savefig(path_to_plot1)


def save_model(model, name, path):
    unique_name = get_unique_filename(name, typ='Model')
    path_to_model = os.path.join(path, unique_name)
    torch.save(model, path_to_model)
