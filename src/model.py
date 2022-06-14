import torch
import torch.nn as nn
from catalyst.contrib import registry
import timm


@registry.Model
class ContainersModel(nn.Module):
    def __init__(
            self,
            encoder_name: str,
            pretrained: bool,
            num_classes: int,
            dropout_rate: float = 0.5
    ):
        super().__init__()
        if 'efficientnet' in encoder_name:
            self.model = timm.create_model(model_name=encoder_name, pretrained=pretrained)
            n_features = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Linear(n_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(512, num_classes),
            )
        else:
            raise RuntimeError("unsupported encoder")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.model(features)
        return x
