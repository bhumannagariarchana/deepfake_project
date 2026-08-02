import os
import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNetB0Classifier(nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientNetB0Classifier, self).__init__()
        # Load EfficientNet-B0 using the modern Weights API or fallback for older versions
        try:
            if pretrained:
                self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            else:
                self.backbone = models.efficientnet_b0(weights=None)
        except AttributeError:
            # Fallback for legacy torchvision versions
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            
        # Replace the final classification linear layer with a binary classifier logit
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, 1) # Outputs a raw logit for BCE Loss

    def forward(self, x):
        return self.backbone(x)

def get_model(pretrained=True):
    """
    Helper function to instantiate the EfficientNet-B0 deepfake detector model.
    """
    return EfficientNetB0Classifier(pretrained=pretrained)
