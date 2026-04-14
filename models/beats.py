# models/beats.py
import torch
import torch.nn as nn

class BEATsClassifier(nn.Module):
    def __init__(self, n_classes, checkpoint_path=None):
        super().__init__()

        # load pretrained BEATs backbone
        # checkpoint downloaded from Microsoft's GitHub
        self.beats = load_beats_backbone(checkpoint_path)

        # replace the classification head with our own
        # for 234 BirdCLEF species
        self.head = nn.Linear(768, n_classes)

    def forward(self, waveform):
        # waveform shape: (batch, samples) — raw audio, not mel
        features = self.beats.extract_features(waveform)
        return self.head(features)