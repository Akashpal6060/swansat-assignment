"""
=============================================================================
CropRiskEncoder - Deep Learning Model for Field-Level Risk Assessment
=============================================================================
Architecture:
  1. Spatial Encoder (2D-CNN): Extracts per-month spatial features from
     each 64x64 patch. Shared weights across time steps.
  2. Temporal Encoder (1D-CNN): Learns month-to-month growth patterns
     from the sequence of spatial embeddings.
  3. Fusion Layer: Concatenates temporal embedding with auxiliary features
     (weather, soil, elevation).
  4. Multi-Task Heads:
     - NDVI Prediction (self-supervised pretraining)
     - Risk Classification (pseudo-label training)

Explainability:
  - Grad-CAM on spatial encoder → which pixels matter
  - Temporal attention weights → which months matter
  - Feature attribution on fusion layer → which aux features matter
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialEncoder(nn.Module):
    """
    2D-CNN that processes a single time step (C bands × 64 × 64 pixels).
    Extracts a spatial feature vector. Shared across all 6 months.
    
    Uses a simple but effective architecture:
    - 3 conv blocks with BatchNorm and MaxPool
    - Global average pooling
    - Output: 128-dim embedding
    """
    def __init__(self, in_channels=9):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1: (9, 64, 64) → (32, 32, 32)
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2: (32, 32, 32) → (64, 16, 16)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3: (64, 16, 16) → (128, 8, 8)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Global average pooling → 128-dim
        self.gap = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        """
        Args:
            x: (batch, channels, 64, 64) - single time step
        Returns:
            embedding: (batch, 128)
            feature_maps: (batch, 128, 8, 8) - for Grad-CAM
        """
        feat_maps = self.features(x)
        embedding = self.gap(feat_maps).squeeze(-1).squeeze(-1)
        return embedding, feat_maps


class TemporalEncoder(nn.Module):
    """
    1D-CNN that processes a sequence of spatial embeddings (6 months × 128).
    Learns temporal growth patterns and month-to-month transitions.
    
    Why 1D-CNN over LSTM?
    - Only 6 time steps — LSTM is overkill
    - 1D-CNN captures local transitions (adjacent months) effectively
    - Easier to apply Grad-CAM for temporal explainability
    """
    def __init__(self, in_features=128, hidden_dim=64):
        super().__init__()
        
        self.temporal_conv = nn.Sequential(
            # (batch, 128, 6) → (batch, 64, 6)
            nn.Conv1d(in_features, 64, kernel_size=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            # (batch, 64, 7) → (batch, 64, 7)  (padding adds 1)
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        
        # Attention mechanism over time steps
        self.temporal_attention = nn.Sequential(
            nn.Conv1d(64, 1, kernel_size=1),  # (batch, 1, T)
        )
        
        self.output_dim = 64
        
    def forward(self, x):
        """
        Args:
            x: (batch, 6, 128) - sequence of monthly embeddings
        Returns:
            temporal_embedding: (batch, 64)
            attention_weights: (batch, 6) - which months matter most
        """
        # Reshape for 1D conv: (batch, channels, time_steps)
        x = x.permute(0, 2, 1)  # (batch, 128, 6)
        
        features = self.temporal_conv(x)  # (batch, 64, T)
        
        # Temporal attention
        attn_logits = self.temporal_attention(features)  # (batch, 1, T)
        # Trim to match original 6 time steps
        attn_logits = attn_logits[:, :, :6]
        features = features[:, :, :6]
        
        attn_weights = F.softmax(attn_logits, dim=-1)  # (batch, 1, 6)
        
        # Weighted sum across time
        temporal_embedding = (features * attn_weights).sum(dim=-1)  # (batch, 64)
        
        return temporal_embedding, attn_weights.squeeze(1)


class CropRiskEncoder(nn.Module):
    """
    Full model: Spatial + Temporal + Fusion + Multi-Task Heads.
    
    This is NOT a black box because:
    1. Spatial encoder → Grad-CAM shows WHICH pixels matter
    2. Temporal attention → shows WHICH months matter
    3. Fusion layer → SHAP shows which auxiliary features matter
    4. Each head's output is individually interpretable
    """
    def __init__(self, n_bands=9, n_months=6, n_aux_features=15, n_classes=3):
        super().__init__()
        
        self.n_months = n_months
        self.n_bands = n_bands
        
        # Spatial encoder (shared across months)
        self.spatial_encoder = SpatialEncoder(in_channels=n_bands)
        
        # Temporal encoder
        self.temporal_encoder = TemporalEncoder(in_features=128)
        
        # Fusion layer
        fusion_dim = 64 + n_aux_features  # temporal embedding + aux features
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )
        
        # === MULTI-TASK HEADS ===
        
        # Head 1: NDVI Prediction (self-supervised)
        # Predicts NDVI for a masked month from remaining months + weather
        self.ndvi_predictor = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),  # NDVI is [0, 1]
        )
        
        # Head 2: Risk Classification
        self.risk_classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, n_classes),
        )
        
        # Head 3: Risk Score (regression)
        self.risk_regressor = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),  # Score in [0, 1]
        )
        
    def forward(self, patches, aux_features, mask_month=None):
        """
        Args:
            patches: (batch, 6, 9, 64, 64) - multi-temporal satellite patches
            aux_features: (batch, n_aux) - weather/soil/elevation features
            mask_month: int or None - if set, mask this month for self-supervised training
            
        Returns:
            dict with keys:
                'risk_class': (batch, 3) - class logits
                'risk_score': (batch, 1) - continuous risk score
                'ndvi_pred': (batch, 1) - predicted NDVI for masked month
                'attention': (batch, 6) - temporal attention weights
                'embedding': (batch, 32) - final embedding (for analysis)
        """
        batch_size = patches.shape[0]
        
        # === SPATIAL ENCODING ===
        # Process each month independently through shared spatial encoder
        monthly_embeddings = []
        spatial_feature_maps = []
        
        for t in range(self.n_months):
            month_data = patches[:, t, :, :, :]  # (batch, 9, 64, 64)
            
            # If masking this month (self-supervised), zero it out
            if mask_month is not None and t == mask_month:
                month_data = torch.zeros_like(month_data)
            
            emb, feat_map = self.spatial_encoder(month_data)
            monthly_embeddings.append(emb)
            spatial_feature_maps.append(feat_map)
        
        # Stack: (batch, 6, 128)
        temporal_sequence = torch.stack(monthly_embeddings, dim=1)
        
        # === TEMPORAL ENCODING ===
        temporal_emb, attention_weights = self.temporal_encoder(temporal_sequence)
        
        # === FUSION ===
        fused = torch.cat([temporal_emb, aux_features], dim=1)
        embedding = self.fusion(fused)
        
        # === TASK HEADS ===
        risk_class = self.risk_classifier(embedding)
        risk_score = self.risk_regressor(embedding)
        ndvi_pred = self.ndvi_predictor(embedding)
        
        return {
            'risk_class': risk_class,
            'risk_score': risk_score,
            'ndvi_pred': ndvi_pred,
            'attention': attention_weights,
            'embedding': embedding,
            'spatial_features': spatial_feature_maps,  # For Grad-CAM
        }


class CropRiskLoss(nn.Module):
    """
    Multi-task loss combining:
    1. NDVI prediction loss (MSE) - self-supervised signal
    2. Risk classification loss (CrossEntropy) - pseudo-label signal
    3. Risk regression loss (MSE) - continuous score signal
    
    Weights are adjusted during training phases.
    """
    def __init__(self, ndvi_weight=1.0, class_weight=1.0, reg_weight=0.5,
                 ce_class_weights=None):
        super().__init__()
        self.ndvi_weight = ndvi_weight
        self.class_weight = class_weight
        self.reg_weight = reg_weight
        
        self.ce_loss = nn.CrossEntropyLoss(weight=ce_class_weights)
        self.mse_loss = nn.MSELoss()
        
    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict from CropRiskEncoder.forward()
            targets: dict with keys 'ndvi_true', 'risk_label', 'risk_score'
        """
        total_loss = 0.0
        loss_dict = {}
        
        # NDVI prediction loss (self-supervised)
        if 'ndvi_true' in targets and targets['ndvi_true'] is not None:
            ndvi_loss = self.mse_loss(
                outputs['ndvi_pred'].squeeze(), targets['ndvi_true']
            )
            total_loss += self.ndvi_weight * ndvi_loss
            loss_dict['ndvi_loss'] = ndvi_loss.item()
        
        # Classification loss
        if 'risk_label' in targets:
            class_loss = self.ce_loss(
                outputs['risk_class'], targets['risk_label']
            )
            total_loss += self.class_weight * class_loss
            loss_dict['class_loss'] = class_loss.item()
        
        # Regression loss
        if 'risk_score' in targets:
            reg_loss = self.mse_loss(
                outputs['risk_score'].squeeze(), targets['risk_score']
            )
            total_loss += self.reg_weight * reg_loss
            loss_dict['reg_loss'] = reg_loss.item()
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# TEST
# ============================================================================

if __name__ == '__main__':
    # Quick test with dummy data
    print("Testing CropRiskEncoder...")
    
    model = CropRiskEncoder(n_bands=9, n_months=6, n_aux_features=15, n_classes=3)
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Dummy input
    batch = 4
    patches = torch.randn(batch, 6, 9, 64, 64)
    aux = torch.randn(batch, 15)
    
    # Forward pass
    outputs = model(patches, aux)
    
    print(f"\nOutput shapes:")
    print(f"  risk_class: {outputs['risk_class'].shape}")
    print(f"  risk_score: {outputs['risk_score'].shape}")
    print(f"  ndvi_pred:  {outputs['ndvi_pred'].shape}")
    print(f"  attention:  {outputs['attention'].shape}")
    print(f"  embedding:  {outputs['embedding'].shape}")
    print(f"  attention weights: {outputs['attention'][0].detach().numpy()}")
    
    # Test with masked month (self-supervised)
    outputs_masked = model(patches, aux, mask_month=3)
    print(f"\nWith masked month 3:")
    print(f"  ndvi_pred: {outputs_masked['ndvi_pred'][0].item():.4f}")
    
    # Test loss
    loss_fn = CropRiskLoss()
    targets = {
        'ndvi_true': torch.rand(batch),
        'risk_label': torch.randint(0, 3, (batch,)),
        'risk_score': torch.rand(batch),
    }
    loss, loss_dict = loss_fn(outputs, targets)
    print(f"\nLoss: {loss_dict}")
    
    print("\n✓ Model test passed!")
