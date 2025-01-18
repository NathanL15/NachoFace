import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from capsnet import CapsNet
from data_load import VGGFace2Dataset
from tqdm import tqdm

USE_MPS = True if torch.backends.mps.is_available() else False
BATCH_SIZE = 100
N_EPOCHS = 30
LEARNING_RATE = 0.01
MOMENTUM = 0.9

'''
Config class to determine the parameters for capsule net
'''
class Config:
    def __init__(self, dataset='vgg'):
        if dataset == 'vgg':
            # Capsule NN (cnn)
            self.cnn_in_channels = 3
            self.cnn_out_channels = 256
            self.cnn_kernel_size = 3

            # Primary Capsule (pc)
            self.pc_num_capsules = 8
            self.pc_in_channels = 256
            self.pc_out_channels = 32
            self.pc_kernel_size = 3
            self.pc_num_routes = 32 * 6 * 6

            # Digit Capsule (dc)
            self.dc_num_capsules = 10
            self.dc_num_routes = 32 * 6 * 6
            self.dc_in_channels = 8
            self.dc_out_channels = 16

            # Decoder
            self.input_width = 28
            self.input_height = 2

        elif dataset == 'inputimg':
            pass


def save_model(model, filename="model_weights.pth"):
    torch.save(model.state_dict(), filename)


def train_triplet(model, optimizer, train_loader, epoch, triplet_loss):
    model.train()
    total_loss = 0

    for batch_id, (anchor_data, positive_data, negative_data) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()

        anchor_embedding, _, _ = model(anchor_data)
        positive_embedding, _, _ = model(positive_data)
        negative_embedding, _, _ = model(negative_data)

        # Calculate the triplet loss
        loss = triplet_loss(anchor_embedding, positive_embedding, negative_embedding)
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

        if batch_id % 100 == 0:
            print(f"Epoch [{epoch}/{N_EPOCHS}], Batch [{batch_id}/{len(train_loader)}], Loss: {loss.item():.6f}")

    # Save the model weights every 5 epochs
    if epoch % 5 == 0:
        save_model(model, f"capsule_net_epoch_{epoch}.pth")
        
    print(f"Epoch [{epoch}/{N_EPOCHS}], Total Loss: {total_loss / len(train_loader):.6f}")

# For testing, use embedding comparison
def test(capsule_net, test_loader, epoch, known_embeddings, known_labels):
    capsule_net.eval()
    test_loss = 0
    correct = 0
    for batch_id, (data, target) in enumerate(test_loader):
        ...
        # Compare embeddings using Euclidean distance
        distances = [F.pairwise_distance(test_embedding, known_embedding) for known_embedding in known_embeddings]
        min_distance_idx = torch.argmin(torch.tensor(distances))
        predicted_label = known_labels[min_distance_idx]
        ...


def recognize_face(model, test_image, known_embeddings, known_labels):
    model.eval()

    # Preprocess and extract embedding for test image
    test_embedding, _, _ = model(test_image)

    # Compare test embedding with known embeddings
    distances = [F.pairwise_distance(test_embedding, known_embedding) for known_embedding in known_embeddings]
    
    # Get the index of the most similar known embedding
    min_distance_idx = torch.argmin(torch.tensor(distances))
    predicted_label = known_labels[min_distance_idx]

    return predicted_label



if __name__ == '__main__':
    torch.manual_seed(1)
    dataset = 'vgg'
    config = Config(dataset)
    vgg = VGGFace2Dataset(dataset, BATCH_SIZE)

    capsule_net = CapsNet(config)
    capsule_net = torch.nn.DataParallel(capsule_net)
    if USE_MPS:
        capsule_net = capsule_net.mps()
    capsule_net = capsule_net.module

    optimizer = torch.optim.Adam(capsule_net.parameters())

    for e in range(1, N_EPOCHS + 1):
        train_triplet(capsule_net, optimizer, vgg.train_loader, e, triplet_loss)  # Call train_triplet instead of train
        test(capsule_net, vgg.test_loader, e)

    save_model(capsule_net, "final_capsule_net_weights.pth")
