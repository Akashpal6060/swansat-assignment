"""
=============================================================================
Script 04: Inference - Generate Risk Scores + Explainability
=============================================================================
Loads trained model and runs inference on all patches.
Generates: risk scores, Grad-CAM heatmaps, temporal attention, feature importance.

Runs on: A100 GPU via SLURM
Time: ~10-15 minutes
=============================================================================
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm

from src.model import CropRiskEncoder
from src.explainability import GradCAM, extract_temporal_attention, generate_explanation

# ============================================================================
# CONFIGURATION
# ============================================================================

PATCH_DIR = Path('data/patches')
MODEL_DIR = Path('models')
OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64


def main():
    print("="*60)
    print("  INFERENCE: GENERATING RISK SCORES")
    print(f"  Device: {DEVICE}")
    print("="*60)

    # Load data
    patches = np.load(PATCH_DIR / 'patches.npy')
    aux_features = np.load(PATCH_DIR / 'aux_features.npy')
    coords_df = pd.read_csv(PATCH_DIR / 'patch_coords.csv')
    
    with open(PATCH_DIR / 'metadata.json') as f:
        metadata = json.load(f)

    n_bands = metadata['n_bands']
    n_aux = aux_features.shape[1]
    aux_names = metadata.get('aux_feature_names', [f'feat_{i}' for i in range(n_aux)])

    print(f"  Patches: {patches.shape}")
    print(f"  Generating predictions for {len(patches)} patches...")

    # Load model
    model = CropRiskEncoder(
        n_bands=n_bands, n_months=6,
        n_aux_features=n_aux, n_classes=3
    ).to(DEVICE)

    model_path = MODEL_DIR / 'risk_model.pth'
    if not model_path.exists():
        model_path = MODEL_DIR / 'pretrained_encoder.pth'
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"  Loaded model: {model_path.name}")

    # ========================================================================
    # BATCH INFERENCE
    # ========================================================================
    
    all_risk_classes = []
    all_risk_scores = []
    all_confidences = []
    all_attention = []
    all_embeddings = []

    patches_tensor = torch.FloatTensor(patches)
    aux_tensor = torch.FloatTensor(aux_features)

    with torch.no_grad():
        for i in range(0, len(patches), BATCH_SIZE):
            batch_patches = patches_tensor[i:i+BATCH_SIZE].to(DEVICE)
            batch_aux = aux_tensor[i:i+BATCH_SIZE].to(DEVICE)

            outputs = model(batch_patches, batch_aux)

            probs = F.softmax(outputs['risk_class'], dim=1)
            pred_class = probs.argmax(dim=1)
            confidence = probs.max(dim=1).values

            all_risk_classes.append(pred_class.cpu().numpy())
            all_risk_scores.append(outputs['risk_score'].squeeze().cpu().numpy())
            all_confidences.append(confidence.cpu().numpy())
            all_attention.append(outputs['attention'].cpu().numpy())
            all_embeddings.append(outputs['embedding'].cpu().numpy())

    risk_classes = np.concatenate(all_risk_classes)
    risk_scores = np.concatenate(all_risk_scores)
    confidences = np.concatenate(all_confidences)
    attention = np.concatenate(all_attention)
    embeddings = np.concatenate(all_embeddings)

    # ========================================================================
    # BUILD RESULTS DATAFRAME
    # ========================================================================

    class_names = ['Low Risk', 'Medium Risk', 'High Risk']

    results_df = coords_df.copy()
    results_df['risk_class'] = risk_classes
    results_df['risk_label'] = [class_names[c] for c in risk_classes]
    results_df['risk_score'] = np.round(risk_scores, 4)
    results_df['confidence'] = np.round(confidences, 4)

    # Add temporal attention per month
    month_names = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
    for j, mname in enumerate(month_names):
        results_df[f'attn_{mname}'] = np.round(attention[:, j], 4)

    results_df.to_csv(OUTPUT_DIR / 'risk_scores.csv', index=False)

    # Save embeddings for visualization
    np.save(OUTPUT_DIR / 'embeddings.npy', embeddings)
    np.save(OUTPUT_DIR / 'attention_weights.npy', attention)

    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================

    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")

    for cls, name in enumerate(class_names):
        mask = risk_classes == cls
        count = mask.sum()
        pct = 100 * count / len(risk_classes)
        mean_score = risk_scores[mask].mean() if count > 0 else 0
        mean_conf = confidences[mask].mean() if count > 0 else 0
        print(f"  {name:15s}: {count:5d} patches ({pct:5.1f}%) | "
              f"avg score: {mean_score:.3f} | avg conf: {mean_conf:.3f}")

    print(f"\n  Mean temporal attention:")
    for j, mname in enumerate(month_names):
        mean_attn = attention[:, j].mean()
        bar = '█' * int(mean_attn * 50)
        print(f"    {mname}: {mean_attn:.3f} {bar}")

    # ========================================================================
    # EXPLAINABILITY: SAMPLE EXPLANATIONS
    # ========================================================================

    print(f"\n{'='*60}")
    print(f"  SAMPLE EXPLANATIONS")
    print(f"{'='*60}")

    # Generate explanations for 3 example patches (one per risk class)
    explanations = []
    for cls in range(3):
        cls_indices = np.where(risk_classes == cls)[0]
        if len(cls_indices) == 0:
            continue

        # Pick the most confident prediction for this class
        cls_confs = confidences[cls_indices]
        best_idx = cls_indices[cls_confs.argmax()]

        patch_tensor = torch.FloatTensor(patches[best_idx:best_idx+1]).to(DEVICE)
        aux_tensor_single = torch.FloatTensor(aux_features[best_idx:best_idx+1]).to(DEVICE)

        explanation = generate_explanation(
            model, patch_tensor, aux_tensor_single, aux_names,
            patch_id=int(best_idx)
        )
        explanations.append(explanation)

        print(f"\n  --- {class_names[cls]} Example (Patch {best_idx}) ---")
        print(f"  Risk Score: {explanation['risk_score']}")
        print(f"  Confidence: {explanation['confidence']}")
        print(f"  Critical Months: {explanation['critical_months']}")
        print(f"  Top Features: {explanation['top_features'][:3]}")

    with open(OUTPUT_DIR / 'sample_explanations.json', 'w') as f:
        json.dump(explanations, f, indent=2)

    # ========================================================================
    # GRAD-CAM FOR SAMPLE PATCHES
    # ========================================================================

    print(f"\n  Generating Grad-CAM heatmaps for sample patches...")

    gradcam = GradCAM(model)
    gradcam_maps = {}

    for cls in range(3):
        cls_indices = np.where(risk_classes == cls)[0]
        if len(cls_indices) == 0:
            continue

        idx = cls_indices[confidences[cls_indices].argmax()]
        patch_tensor = torch.FloatTensor(patches[idx:idx+1]).to(DEVICE)
        aux_single = torch.FloatTensor(aux_features[idx:idx+1]).to(DEVICE)

        heatmaps = []
        for month in range(6):
            # Need to forward through spatial encoder for each month separately
            try:
                cam = gradcam.generate(patch_tensor, aux_single, 
                                       target_month=month, target_class=cls)
                heatmaps.append(cam)
            except Exception:
                heatmaps.append(np.zeros((64, 64)))

        gradcam_maps[class_names[cls]] = {
            'patch_id': int(idx),
            'heatmaps': np.array(heatmaps),  # (6, 64, 64)
        }

    np.savez(OUTPUT_DIR / 'gradcam_samples.npz',
             **{f'{k}_heatmaps': v['heatmaps'] for k, v in gradcam_maps.items()})

    print(f"\n{'='*60}")
    print(f"  INFERENCE COMPLETE")
    print(f"  Results: {OUTPUT_DIR / 'risk_scores.csv'}")
    print(f"  Next: python scripts/05_visualize.py")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
