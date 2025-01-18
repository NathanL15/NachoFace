import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from capsnet import CapsNet
from data_load import VGGFace2Dataset
from tqdm import tqdm
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--learning_rate', type=float, default=0.001)
args = parser.parse_args()

BATCH_SIZE = args.batch_size
N_EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate
MOMENTUM = 0.9
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

'''
Config class to determine the parameters for capsule net
'''
class Config:
    def __init__(self, dataset='vgg'):
        if dataset == 'vgg':
            self.cnn_in_channels = 3
            self.cnn_out_channels = 256
            self.cnn_kernel_size = 3

            # Primary Capsule
            self.pc_num_capsules = 8
            self.pc_in_channels = 256
            self.pc_out_channels = 32
            self.pc_kernel_size = 3
            # Adjusting num_routes for larger input size
            self.pc_num_routes = 8 * 112 * 112  # Adjusted for 224x224 input with stride 2

            # Digit Capsule
            self.dc_num_capsules = 10
            self.dc_num_routes = 8 * 112 * 112  # Same adjustment
            self.dc_in_channels = 8
            self.dc_out_channels = 16

            # Decoder
            self.input_width = 16  # Updated as needed for new capsule sizes
            self.input_height = 16  # Same here




def save_model(model, filename="model_weights.pth"):
    torch.save(model.state_dict(), os.path.join("modelcheckpoints", filename))

def train_triplet(model, optimizer, train_loader, epoch, triplet_loss):
    model.train()
    total_loss = 0

    for batch_id, (anchor_data, positive_data, negative_data) in enumerate(tqdm(train_loader)):
        anchor_data = anchor_data.to(device)
        positive_data = positive_data.to(device)
        negative_data = negative_data.to(device)

        optimizer.zero_grad()
        anchor_embedding, _, _ = model(anchor_data)
        positive_embedding, _, _ = model(positive_data)
        negative_embedding, _, _ = model(negative_data)

        loss = model.compute_triplet_loss(anchor_embedding, positive_embedding, negative_embedding)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_id % 32 == 0:
            print(f"Epoch [{epoch}/{N_EPOCHS}], Batch [{batch_id}/{len(train_loader)}], Loss: {loss.item():.6f}")

    # Save the model weights every epoch
    save_model(model, f"capsule_net_epoch_{epoch}.pth")
        
    print(f"Epoch [{epoch}/{N_EPOCHS}], Total Loss: {total_loss / len(train_loader):.6f}")

# For testing, use embedding comparison
def test(capsule_net, test_loader, epoch, known_embeddings, known_labels):
    capsule_net.eval()
    test_loss = 0
    correct = 0
    for batch_id, (data, target) in enumerate(test_loader):
        data = data.to(device)
        embeddings, _, _ = capsule_net(data)

        for test_embedding, true_label in zip(embeddings, target):
            distances = [F.pairwise_distance(test_embedding.unsqueeze(0), known_embedding.unsqueeze(0)) for known_embedding in known_embeddings]
            predicted_label = known_labels[torch.argmin(torch.tensor(distances))]
            if predicted_label == true_label:
                correct += 1
    accuracy = correct / len(test_loader.dataset)
    print(f"Test Accuracy: {accuracy:.4f}")

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
    
    #Load dataset
    datasetroot = './vggface2_preprocessed'
    vgg = VGGFace2Dataset(root_dir=datasetroot, batch_size=BATCH_SIZE)
    
    train_loader = vgg.train_loader
    test_loader = vgg.test_loader
    
    # CapsNet initialization
    config = Config(dataset='vgg')
    capsule_net = CapsNet(config)
    capsule_net = torch.nn.DataParallel(capsule_net)
    
    capsule_net = capsule_net.to(device)

    optimizer = torch.optim.Adam(
        capsule_net.parameters(),
        lr=LEARNING_RATE
    )
    
    triplet_loss_fn = capsule_net.module.compute_triplet_loss
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    for e in range(1, N_EPOCHS + 1):
        train_triplet(capsule_net, optimizer, vgg.train_loader, e,  triplet_loss_fn)
        test(capsule_net, vgg.test_loader, e)
        scheduler.step()

    save_model(capsule_net, "final_capsule_net_weights.pth")
