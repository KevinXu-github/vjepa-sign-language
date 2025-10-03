import torch
import torch.nn as nn

class VJEPAClassifier(nn.Module):
    def __init__(self, vjepa_encoder, num_classes=100, feature_dim=1024, freeze_encoder=True):
        """
        Args:
            vjepa_encoder: Pretrained V-JEPA encoder
            num_classes: Number of sign classes
            feature_dim: Dimension of V-JEPA features
            freeze_encoder: Whether to freeze encoder during training
        """
        super().__init__()
        self.encoder = vjepa_encoder
        self.freeze_encoder = freeze_encoder
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Temporal pooling (average across all tokens)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input video [B, C, T, H, W]
        Returns:
            logits: [B, num_classes]
        """
        # Extract features from V-JEPA
        features = self.encoder(x)  # [B, num_tokens, feature_dim]
        
        # Pool across tokens
        pooled = self.pool(features.transpose(1, 2)).squeeze(-1)  # [B, feature_dim]
        
        # Classify
        logits = self.classifier(pooled)  # [B, num_classes]
        
        return logits
