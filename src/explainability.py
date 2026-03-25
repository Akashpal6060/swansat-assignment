"""
Explainability utilities for CropRiskEncoder.
- Grad-CAM: Which pixels in the satellite image drove the prediction
- Temporal Attention: Which months mattered most
- Feature Attribution: Which auxiliary features (weather/soil) contributed
"""

import torch
import torch.nn.functional as F
import numpy as np


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for the spatial encoder.
    Shows which spatial regions in a 64x64 patch the model focuses on.
    """
    
    def __init__(self, model, target_layer=None):
        self.model = model
        self.gradients = None
        self.activations = None
        
        # Hook into the last conv layer of spatial encoder
        if target_layer is None:
            # Get the last conv layer in spatial encoder
            target_layer = model.spatial_encoder.features[-3]  # Last conv before final pool
        
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, patches, aux_features, target_month=None, target_class=None):
        """
        Generate Grad-CAM heatmap for a specific month and class.
        
        Args:
            patches: (1, 6, 9, 64, 64)
            aux_features: (1, D)
            target_month: which month to visualize (0-5)
            target_class: which risk class to explain (0=low, 1=med, 2=high)
            
        Returns:
            heatmap: (64, 64) numpy array, normalized [0, 1]
        """
        self.model.eval()
        patches.requires_grad_(True)
        
        # Forward pass
        outputs = self.model(patches, aux_features)
        
        # Get target score
        if target_class is None:
            target_class = outputs['risk_class'].argmax(dim=1).item()
        
        score = outputs['risk_class'][0, target_class]
        
        # Backward pass
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            return np.zeros((64, 64))
        
        # Compute weights (global average pooling of gradients)
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)  # Only positive contributions
        
        # Upsample to 64x64
        cam = F.interpolate(cam, size=(64, 64), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam


def extract_temporal_attention(model, patches, aux_features):
    """
    Extract temporal attention weights from the model.
    Shows which months the model considers most important.
    
    Returns:
        attention: (6,) numpy array - weight per month
        month_names: list of month name strings
    """
    model.eval()
    with torch.no_grad():
        outputs = model(patches, aux_features)
    
    attention = outputs['attention'][0].cpu().numpy()  # (6,)
    month_names = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
    
    return attention, month_names


def compute_feature_attribution(model, patches, aux_features, aux_feature_names,
                                 n_samples=50):
    """
    Simple perturbation-based feature attribution for auxiliary features.
    For each feature, measures how much the risk score changes when we
    replace it with a random value.
    
    This is a simplified version of SHAP's permutation importance.
    
    Returns:
        importance: dict of {feature_name: importance_score}
    """
    model.eval()
    
    with torch.no_grad():
        base_output = model(patches, aux_features)
        base_score = base_output['risk_score'][0].item()
    
    importance = {}
    
    for i, feat_name in enumerate(aux_feature_names):
        deltas = []
        
        for _ in range(n_samples):
            # Perturb this feature
            perturbed_aux = aux_features.clone()
            perturbed_aux[0, i] = torch.rand(1).item()  # Random value in [0,1]
            
            with torch.no_grad():
                perturbed_output = model(patches, perturbed_aux)
                perturbed_score = perturbed_output['risk_score'][0].item()
            
            deltas.append(abs(perturbed_score - base_score))
        
        importance[feat_name] = np.mean(deltas)
    
    # Normalize
    total = sum(importance.values())
    if total > 0:
        importance = {k: v/total for k, v in importance.items()}
    
    return importance


def generate_explanation(model, patches, aux_features, aux_feature_names,
                         patch_id, risk_class_names=['Low Risk', 'Medium Risk', 'High Risk']):
    """
    Generate a human-readable explanation for a single patch's risk prediction.
    
    Returns:
        explanation: dict with all explainability components
    """
    model.eval()
    
    with torch.no_grad():
        outputs = model(patches, aux_features)
    
    pred_class = outputs['risk_class'][0].argmax().item()
    pred_score = outputs['risk_score'][0].item()
    attention, month_names = extract_temporal_attention(model, patches, aux_features)
    
    # Feature attribution
    feat_imp = compute_feature_attribution(
        model, patches, aux_features, aux_feature_names, n_samples=30
    )
    
    # Top contributing months
    top_months = sorted(zip(month_names, attention), key=lambda x: -x[1])[:3]
    
    # Top contributing features
    top_features = sorted(feat_imp.items(), key=lambda x: -x[1])[:5]
    
    explanation = {
        'patch_id': patch_id,
        'predicted_class': risk_class_names[pred_class],
        'risk_score': round(pred_score, 3),
        'confidence': round(F.softmax(outputs['risk_class'][0], dim=0).max().item(), 3),
        'critical_months': [(m, round(float(w), 3)) for m, w in top_months],
        'top_features': [(f, round(float(v), 3)) for f, v in top_features],
        'temporal_attention': {m: round(float(w), 3) for m, w in zip(month_names, attention)},
    }
    
    return explanation
