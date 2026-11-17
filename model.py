
import torch
import torch.nn as nn
from torchvision import models


class BoneAgeNet(nn.Module):

    def __init__(self):
        super(BoneAgeNet, self).__init__()

        # ResNet50 image feature extractor
        self.cnn = models.resnet50(weights=None)
        self.cnn.fc = nn.Identity()

        # Patient sex feature branch
        self.sex_branch = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU()
        )

        # Combined regression head
        self.regressor = nn.Sequential(
            nn.Linear(2080, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, images, sexes):

        image_features = self.cnn(images)
        sex_features = self.sex_branch(sexes)

        combined = torch.cat(
            [image_features, sex_features],
            dim=1
        )

        return self.regressor(combined)
