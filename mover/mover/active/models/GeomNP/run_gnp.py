import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from omegaconf import OmegaConf
import utils
from layers import GraphConvolution
from tqdm import tqdm

# Load configuration
config = OmegaConf.load("config.yaml")
device = torch.device("cuda" if config.device_settings.use_cuda else "cpu")

class GraphEncoder(nn.Module):
    def __init__(self, in_features=17, out_features=17):
        super(GraphEncoder, self).__init__()
        self.gc1 = GraphConvolution(in_features, out_features)
        self.gc2 = GraphConvolution(out_features, out_features)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        return x

class NonGraphEncoder(nn.Module):
    def __init__(self, eigen_m_size):
        super(NonGraphEncoder, self).__init__()
        layers = []
        input_size = 2 * eigen_m_size + 4
        for size in config.model_params.layer_sizes:
            layers.append(nn.Linear(input_size, size))
            layers.append(nn.ReLU())
            input_size = size
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return torch.mean(self.model(x), dim=0)

class Decoder(nn.Module):
    def __init__(self, eigen_m_size):
        super(Decoder, self).__init__()
        layers = [
            nn.Linear(256 + 2 * eigen_m_size + 4, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, r, x):
        n = x.shape[0]
        x = torch.cat((x, r.expand(n, -1)), 1)
        return self.model(x)


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """Save checkpoint if a new best is achieved"""
    print("=> Saving a new best")
    torch.save(state, filename)  # save checkpoint
    print("=> checkpoint saved to {}".format(filename))


if __name__ == "__main__":
    current_directory = os.getcwd()
    dataset_directory = os.path.join(current_directory, config.paths.dataset_directory)
    results_directory = os.path.join(current_directory, config.paths.results_directory)
    os.makedirs(results_directory, exist_ok=True)

    # Example data path, you might need to adjust it based on actual usage
    data_path = os.path.join(dataset_directory, "tox21_ahr.pkl")
    train, test = utils.get_data(path=data_path, filter_graphs_min=50)

    # Code to load a checkpoint if it exists



    epochs = config.training_params.epochs
    eigen_m_size = config.model_params.eigen_m_size

    # checkpoint_path = 'path_to_checkpoint/checkpoint_epoch_5.pth.tar'
    # if os.path.isfile(checkpoint_path):
    #     checkpoint = torch.load(checkpoint_path)
    #     encoder.load_state_dict(checkpoint['encoder_state_dict'])
    #     decoder.load_state_dict(checkpoint['decoder_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     start_epoch = checkpoint['epoch']
    #     print(f"Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")
    # else:
    #     start_epoch = 1
    #     print("No checkpoint found at '{}', starting training from scratch".format(checkpoint_path))


    for epoch in range(1, epochs + 1):

        encoder = NonGraphEncoder(eigen_m_size).to(device)
        decoder = Decoder(eigen_m_size).to(device)
        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))
        loss = nn.CrossEntropyLoss()

        for data in train:
            optimizer.zero_grad()
            data = data.to(device)
            output = encoder(data)
            loss_val = loss(output, data)
            loss_val.backward()
            optimizer.step()

        # Checkpointing
        if epoch % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_val,
            }
            save_checkpoint(checkpoint,
                            filename=os.path.join(results_directory, f'checkpoint_epoch_{epoch}.pth.tar'))

        # Evaluate on test data
        with torch.no_grad():
            for data in test:
                data = data.to(device)
                output = encoder(data)
                # Calculate metrics, log results, etc.

        # Save models and results
        torch.save(encoder.state_dict(), os.path.join(results_directory, 'encoder.pth'))
        torch.save(decoder.state_dict(), os.path.join(results_directory, 'decoder.pth'))
