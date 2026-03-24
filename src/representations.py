import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


class ResNet18Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        return self.resnet(x)


def load_encoder(weights_path, device='cuda'):
    encoder = ResNet18Encoder()
    encoder.load_state_dict(torch.load(weights_path, map_location=device))
    encoder = encoder.to(device)
    encoder.eval()
    return encoder


def extract_embeddings(encoder, dataloader, device='cuda'):
    encoder.eval()
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            h = encoder(images)
            h = torch.nn.functional.normalize(h, dim=1)  # L2 normalize (paper: F.1)
            all_embeddings.append(h.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.vstack(all_embeddings), np.hstack(all_labels)
