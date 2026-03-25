"""
PyTorch Dataset for satellite patch data.
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class PatchDataset(Dataset):
    """
    Dataset that loads pre-processed satellite patches with auxiliary features.
    
    Supports two modes:
    1. Self-supervised: Masks a random month, target = NDVI of masked month
    2. Supervised: Uses pseudo-labels for classification + regression
    """
    
    def __init__(self, patches, aux_features, labels=None, risk_scores=None,
                 mode='supervised', augment=False):
        """
        Args:
            patches: (N, 6, 9, 64, 64) numpy array
            aux_features: (N, D) numpy array
            labels: (N,) numpy array of class labels (0,1,2)
            risk_scores: (N,) numpy array of continuous scores
            mode: 'supervised' or 'self_supervised'
            augment: whether to apply data augmentation
        """
        self.patches = torch.FloatTensor(patches)
        self.aux_features = torch.FloatTensor(aux_features)
        self.labels = torch.LongTensor(labels) if labels is not None else None
        self.risk_scores = torch.FloatTensor(risk_scores) if risk_scores is not None else None
        self.mode = mode
        self.augment = augment
        
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = self.patches[idx]  # (6, 9, 64, 64)
        aux = self.aux_features[idx]
        
        # Data augmentation (spatial only - don't augment temporal order)
        if self.augment and self.mode == 'supervised':
            patch = self._augment(patch)
        
        if self.mode == 'self_supervised':
            # Randomly mask one month
            mask_month = np.random.randint(0, 6)
            
            # Target: mean NDVI (band 6) of the masked month
            ndvi_band = patch[mask_month, 6, :, :]  # (64, 64)
            valid_pixels = ndvi_band[ndvi_band > 0]
            ndvi_true = valid_pixels.mean() if len(valid_pixels) > 0 else torch.tensor(0.0)
            
            # Create masked version
            masked_patch = patch.clone()
            masked_patch[mask_month] = 0  # Zero out masked month
            
            return {
                'patches': masked_patch,
                'aux_features': aux,
                'mask_month': mask_month,
                'ndvi_true': ndvi_true,
            }
        
        else:  # supervised
            result = {
                'patches': patch,
                'aux_features': aux,
            }
            if self.labels is not None:
                result['risk_label'] = self.labels[idx]
            if self.risk_scores is not None:
                result['risk_score'] = self.risk_scores[idx]
            return result
    
    def _augment(self, patch):
        """Simple spatial augmentation: random flip and rotation."""
        # Random horizontal flip
        if np.random.random() > 0.5:
            patch = torch.flip(patch, [-1])
        
        # Random vertical flip
        if np.random.random() > 0.5:
            patch = torch.flip(patch, [-2])
        
        # Random 90-degree rotation
        k = np.random.randint(0, 4)
        if k > 0:
            patch = torch.rot90(patch, k, [-2, -1])
        
        return patch
