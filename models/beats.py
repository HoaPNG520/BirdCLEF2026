# models/beats.py
import torch
import torch.nn as nn

class BEATsClassifier(nn.Module):
    def __init__(self, n_classes, checkpoint_path=None):
        super().__init__()
        
        try:
            from BEATs import BEATs, BEATsConfig
        except ImportError:
            raise ImportError("BEATs module not found. Please clone Microsoft's unilm repo.")

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            cfg = BEATsConfig(checkpoint['cfg'])
            self.beats = BEATs(cfg)
            self.beats.load_state_dict(checkpoint['model'])
        else:
            cfg = BEATsConfig()
            self.beats = BEATs(cfg)

        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(768, n_classes) # BEATs hidden dimension is 768
        )

    def forward(self, waveform):
        # waveform shape: (batch, samples) — raw audio, not mel
        features, _ = self.beats.extract_features(waveform)
        
        # Pool across the sequence length dimension
        pooled_features = features.mean(dim=1)
        return self.head(pooled_features)