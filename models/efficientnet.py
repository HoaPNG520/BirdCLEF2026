# models/efficientnet.py
import torch.nn as nn
import timm

class EfficientNetClassifier(nn.Module):
    def __init__(self, n_classes, model_name='efficientnet_b3', pretrained=True):
        super().__init__()
        
        # in_chans=1 adapts the first convolution layer for our 1-channel Mel spectrograms
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=1,
            num_classes=0,  # Strip ImageNet head
            global_pool=''  # We handle pooling manually
        )
        
        num_features = self.backbone.num_features
        
        # Competition-specific head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(num_features, n_classes)
        )

    def forward(self, x):
        # x expected shape: (batch_size, 1, 128, time_steps)
        features = self.backbone.forward_features(x)
        return self.head(features)