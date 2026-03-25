"""
=============================================================================
Script 03: Train CropRiskEncoder Model
=============================================================================
Two-phase training:
  Phase 1: Self-supervised pretraining (NDVI prediction with masked months)
  Phase 2: Risk classification with pseudo-labels + fine-tuning

Runs on: A100 GPU via SLURM
Time: ~30-60 minutes
=============================================================================
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import json
import time
from pathlib import Path

from src.model import CropRiskEncoder, CropRiskLoss, count_parameters
from src.dataset import PatchDataset

# ============================================================================
# CONFIGURATION
# ============================================================================

PATCH_DIR = Path('data/patches')
MODEL_DIR = Path('models')
MODEL_DIR.mkdir(exist_ok=True)

# Training hyperparameters
BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE_PRETRAIN = 1e-3
LEARNING_RATE_FINETUNE = 5e-4
PRETRAIN_EPOCHS = 30
FINETUNE_EPOCHS = 50
WEIGHT_DECAY = 1e-4

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load preprocessed patches and metadata."""
    print("Loading data...")
    
    patches = np.load(PATCH_DIR / 'patches.npy')
    aux_features = np.load(PATCH_DIR / 'aux_features.npy')
    labels = np.load(PATCH_DIR / 'pseudo_labels.npy')
    risk_scores = np.load(PATCH_DIR / 'risk_scores_pseudo.npy')
    
    with open(PATCH_DIR / 'metadata.json') as f:
        metadata = json.load(f)
    
    print(f"  Patches: {patches.shape}")
    print(f"  Aux features: {aux_features.shape}")
    print(f"  Labels distribution: {np.bincount(labels)}")
    print(f"  Device: {DEVICE}")
    
    return patches, aux_features, labels, risk_scores, metadata


# ============================================================================
# PHASE 1: SELF-SUPERVISED PRETRAINING
# ============================================================================

def pretrain(model, train_loader, val_loader, epochs):
    """
    Self-supervised pretraining: mask a random month, predict its NDVI.
    This teaches the model to understand normal crop growth trajectories.
    """
    print("\n" + "="*60)
    print("  PHASE 1: SELF-SUPERVISED PRETRAINING")
    print("  Task: Predict masked month's NDVI from remaining months")
    print(f"  Epochs: {epochs}")
    print("="*60)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE_PRETRAIN,
                           weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    mse_loss = nn.MSELoss()
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_losses = []
        
        for batch in train_loader:
            patches = batch['patches'].to(DEVICE)
            aux = batch['aux_features'].to(DEVICE)
            mask_month = batch['mask_month'][0].item()  # Same mask for batch
            ndvi_true = batch['ndvi_true'].to(DEVICE)
            
            outputs = model(patches, aux, mask_month=mask_month)
            loss = mse_loss(outputs['ndvi_pred'].squeeze(), ndvi_true)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        scheduler.step()
        
        # --- Validate ---
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                patches = batch['patches'].to(DEVICE)
                aux = batch['aux_features'].to(DEVICE)
                mask_month = batch['mask_month'][0].item()
                ndvi_true = batch['ndvi_true'].to(DEVICE)
                
                outputs = model(patches, aux, mask_month=mask_month)
                loss = mse_loss(outputs['ndvi_pred'].squeeze(), ndvi_true)
                val_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_DIR / 'pretrained_encoder.pth')
            marker = ' ★ saved'
        else:
            marker = ''
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"LR: {lr:.2e}{marker}")
    
    print(f"\n  Best validation loss: {best_val_loss:.6f}")
    return history


# ============================================================================
# PHASE 2: PSEUDO-LABEL FINE-TUNING
# ============================================================================

def finetune(model, train_loader, val_loader, labels_array, epochs):
    """
    Fine-tune with pseudo-labels for risk classification.
    Uses multi-task loss: classification + regression + NDVI.
    
    HANDLES CLASS IMBALANCE via:
    1. Inverse-frequency class weights in CrossEntropyLoss
    2. Weighted random sampler (handled in dataloader creation)
    """
    print("\n" + "="*60)
    print("  PHASE 2: RISK CLASSIFICATION FINE-TUNING")
    print("  Task: Classify patches as Low/Medium/High risk")
    print(f"  Epochs: {epochs}")
    print("="*60)
    
    # Compute class weights (inverse frequency)
    class_counts = np.bincount(labels_array, minlength=3).astype(np.float32)
    class_weights = 1.0 / (class_counts + 1)  # +1 to avoid div by zero
    class_weights = class_weights / class_weights.sum() * 3  # Normalize so mean=1
    class_weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)
    print(f"  Class counts: {class_counts.astype(int)}")
    print(f"  Class weights: [{class_weights[0]:.2f}, {class_weights[1]:.2f}, {class_weights[2]:.2f}]")
    
    # Lower learning rate for fine-tuning
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE_FINETUNE,
                           weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = CropRiskLoss(ndvi_weight=0.3, class_weight=1.0, reg_weight=0.5,
                            ce_class_weights=class_weights_tensor)
    
    best_val_acc = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_losses = []
        
        for batch in train_loader:
            patches = batch['patches'].to(DEVICE)
            aux = batch['aux_features'].to(DEVICE)
            
            targets = {
                'risk_label': batch['risk_label'].to(DEVICE),
                'risk_score': batch['risk_score'].to(DEVICE),
            }
            
            outputs = model(patches, aux)
            loss, loss_dict = loss_fn(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss_dict['total_loss'])
        
        scheduler.step()
        
        # --- Validate ---
        model.eval()
        val_losses = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                patches = batch['patches'].to(DEVICE)
                aux = batch['aux_features'].to(DEVICE)
                
                targets = {
                    'risk_label': batch['risk_label'].to(DEVICE),
                    'risk_score': batch['risk_score'].to(DEVICE),
                }
                
                outputs = model(patches, aux)
                loss, loss_dict = loss_fn(outputs, targets)
                val_losses.append(loss_dict['total_loss'])
                
                preds = outputs['risk_class'].argmax(dim=1)
                correct += (preds == targets['risk_label']).sum().item()
                total += len(preds)
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_acc = correct / total if total > 0 else 0
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_DIR / 'risk_model.pth')
            marker = ' ★ saved'
        else:
            marker = ''
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.3f}{marker}")
    
    print(f"\n  Best validation accuracy: {best_val_acc:.3f}")
    return history


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*60)
    print("  TRAINING CROP RISK ENCODER")
    print(f"  Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("="*60)
    
    start_time = time.time()
    
    # Load data
    patches, aux_features, labels, risk_scores, metadata = load_data()
    
    n_aux = aux_features.shape[1]
    n_bands = metadata['n_bands']
    
    # --- Phase 1: Self-supervised datasets ---
    ss_dataset = PatchDataset(patches, aux_features, mode='self_supervised')
    
    n_val_ss = max(1, int(0.15 * len(ss_dataset)))
    n_train_ss = len(ss_dataset) - n_val_ss
    train_ss, val_ss = random_split(ss_dataset, [n_train_ss, n_val_ss])
    
    train_loader_ss = DataLoader(train_ss, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader_ss = DataLoader(val_ss, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True)
    
    # --- Initialize model ---
    model = CropRiskEncoder(
        n_bands=n_bands,
        n_months=6,
        n_aux_features=n_aux,
        n_classes=3
    ).to(DEVICE)
    
    print(f"\nModel parameters: {count_parameters(model):,}")
    
    # --- Phase 1: Pretrain ---
    pretrain_history = pretrain(model, train_loader_ss, val_loader_ss, PRETRAIN_EPOCHS)
    
    # --- Phase 2: Fine-tune with pseudo-labels ---
    # Load best pretrained weights
    model.load_state_dict(torch.load(MODEL_DIR / 'pretrained_encoder.pth',
                                      map_location=DEVICE))
    
    sup_dataset = PatchDataset(patches, aux_features, labels, risk_scores,
                                mode='supervised', augment=True)
    
    n_val_sup = max(1, int(0.15 * len(sup_dataset)))
    n_train_sup = len(sup_dataset) - n_val_sup
    train_sup, val_sup = random_split(sup_dataset, [n_train_sup, n_val_sup])
    
    # Weighted sampler: oversample minority classes during training
    train_indices = train_sup.indices
    train_labels = labels[train_indices]
    class_counts = np.bincount(train_labels, minlength=3).astype(np.float32)
    sample_weights = 1.0 / (class_counts[train_labels] + 1)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_indices), replacement=True
    )
    
    train_loader_sup = DataLoader(train_sup, batch_size=BATCH_SIZE, sampler=sampler,
                                   num_workers=NUM_WORKERS, pin_memory=True)
    val_loader_sup = DataLoader(val_sup, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=NUM_WORKERS, pin_memory=True)
    
    finetune_history = finetune(model, train_loader_sup, val_loader_sup, labels, FINETUNE_EPOCHS)
    
    # Save training log
    training_log = {
        'pretrain': pretrain_history,
        'finetune': finetune_history,
        'config': {
            'batch_size': BATCH_SIZE,
            'lr_pretrain': LEARNING_RATE_PRETRAIN,
            'lr_finetune': LEARNING_RATE_FINETUNE,
            'pretrain_epochs': PRETRAIN_EPOCHS,
            'finetune_epochs': FINETUNE_EPOCHS,
            'n_patches': len(patches),
            'n_bands': n_bands,
            'n_aux_features': n_aux,
            'model_params': count_parameters(model),
            'device': str(DEVICE),
        },
        'total_time_minutes': round((time.time() - start_time) / 60, 1),
    }
    
    with open(MODEL_DIR / 'training_log.json', 'w') as f:
        json.dump(training_log, f, indent=2)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Models saved to: {MODEL_DIR}")
    print(f"  Next: python scripts/04_inference.py")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
