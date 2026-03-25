"""
=============================================================================
Script 05: Visualization - Risk Maps, Explainability Plots
=============================================================================
Generates all output visualizations for the assignment submission.

Runs on: Login node (CPU)
Time: ~2-5 minutes
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import json
from pathlib import Path

OUTPUT_DIR = Path('outputs')
PATCH_DIR = Path('data/patches')
DATA_DIR = Path('data')

# Color scheme
RISK_COLORS = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c'}  # Green, Orange, Red
RISK_NAMES = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}
MONTH_NAMES = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']


def plot_risk_map(results_df):
    """Plot spatial risk map showing Low/Medium/High risk distribution."""
    print("  Generating risk map...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Risk classification map
    ax = axes[0]
    for cls in [0, 1, 2]:
        mask = results_df['risk_class'] == cls
        ax.scatter(results_df.loc[mask, 'lon'], results_df.loc[mask, 'lat'],
                   c=RISK_COLORS[cls], s=8, alpha=0.7, label=RISK_NAMES[cls])

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Field-Level Risk Classification\nGuntur District, Kharif 2024 (Cotton)')
    ax.legend(loc='upper right', fontsize=10, markerscale=3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Right: Continuous risk score heatmap
    ax2 = axes[1]
    scatter = ax2.scatter(results_df['lon'], results_df['lat'],
                          c=results_df['risk_score'], cmap='RdYlGn_r',
                          s=8, alpha=0.8, vmin=0, vmax=1)
    plt.colorbar(scatter, ax=ax2, label='Risk Score (0=Low, 1=High)')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_title('Continuous Risk Score Map\n(Higher = More Risk)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'risk_map_static.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    ✓ risk_map_static.png")


def plot_risk_distribution(results_df):
    """Plot risk score distribution and class breakdown."""
    print("  Generating risk distribution plots...")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Histogram of risk scores
    ax = axes[0]
    ax.hist(results_df['risk_score'], bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(0.30, color='orange', linestyle='--', label='Med threshold')
    ax.axvline(0.55, color='red', linestyle='--', label='High threshold')
    ax.set_xlabel('Risk Score')
    ax.set_ylabel('Number of Patches')
    ax.set_title('Risk Score Distribution')
    ax.legend()

    # Pie chart
    ax2 = axes[1]
    class_counts = results_df['risk_class'].value_counts().sort_index()
    labels = [RISK_NAMES[i] for i in class_counts.index]
    colors = [RISK_COLORS[i] for i in class_counts.index]
    ax2.pie(class_counts.values, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 11})
    ax2.set_title('Risk Class Distribution')

    # Confidence distribution per class
    ax3 = axes[2]
    for cls in [0, 1, 2]:
        mask = results_df['risk_class'] == cls
        if mask.sum() > 0:
            ax3.hist(results_df.loc[mask, 'confidence'], bins=20,
                     color=RISK_COLORS[cls], alpha=0.5, label=RISK_NAMES[cls])
    ax3.set_xlabel('Prediction Confidence')
    ax3.set_ylabel('Count')
    ax3.set_title('Model Confidence by Risk Class')
    ax3.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'risk_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ risk_distribution.png")


def plot_temporal_attention(results_df):
    """Plot which months the model considers most important."""
    print("  Generating temporal attention plot...")

    attn_cols = [f'attn_{m}' for m in MONTH_NAMES]
    available_cols = [c for c in attn_cols if c in results_df.columns]

    if not available_cols:
        print("    ⚠ No attention columns found, skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall mean attention
    ax = axes[0]
    mean_attn = results_df[available_cols].mean().values
    months = [c.replace('attn_', '') for c in available_cols]
    bars = ax.bar(months, mean_attn, color='teal', alpha=0.8, edgecolor='white')

    # Highlight peak
    peak_idx = np.argmax(mean_attn)
    bars[peak_idx].set_color('#e74c3c')
    bars[peak_idx].set_edgecolor('darkred')

    ax.set_xlabel('Month')
    ax.set_ylabel('Mean Attention Weight')
    ax.set_title('Temporal Attention: Which Months Matter Most?')
    ax.set_ylim(0, max(mean_attn) * 1.3)

    # Add cotton growth stage annotations
    stages = ['Sowing', 'Vegetative', 'Peak Veg', 'Flowering\n(Critical)', 'Boll Dev', 'Harvest']
    for i, (month, stage) in enumerate(zip(months, stages)):
        ax.text(i, mean_attn[i] + 0.005, stage, ha='center', va='bottom',
                fontsize=8, color='gray', style='italic')

    # Per-class attention
    ax2 = axes[1]
    width = 0.25
    x = np.arange(len(months))

    for cls in [0, 1, 2]:
        mask = results_df['risk_class'] == cls
        if mask.sum() > 0:
            cls_attn = results_df.loc[mask, available_cols].mean().values
            ax2.bar(x + cls * width, cls_attn, width,
                    color=RISK_COLORS[cls], label=RISK_NAMES[cls], alpha=0.8)

    ax2.set_xlabel('Month')
    ax2.set_ylabel('Mean Attention Weight')
    ax2.set_title('Attention by Risk Class')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(months)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'temporal_attention.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ temporal_attention.png")


def plot_training_curves():
    """Plot training loss and accuracy curves."""
    print("  Generating training curves...")

    log_path = Path('models/training_log.json')
    if not log_path.exists():
        print("    ⚠ training_log.json not found, skipping.")
        return

    with open(log_path) as f:
        log = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Pretraining loss
    if 'pretrain' in log and log['pretrain']['train_loss']:
        ax = axes[0]
        ax.plot(log['pretrain']['train_loss'], label='Train', color='steelblue')
        ax.plot(log['pretrain']['val_loss'], label='Val', color='coral')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.set_title('Phase 1: Self-Supervised Pretraining\n(NDVI Prediction)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Fine-tuning loss
    if 'finetune' in log and log['finetune']['train_loss']:
        ax2 = axes[1]
        ax2.plot(log['finetune']['train_loss'], label='Train', color='steelblue')
        ax2.plot(log['finetune']['val_loss'], label='Val', color='coral')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Multi-Task Loss')
        ax2.set_title('Phase 2: Risk Classification\n(Pseudo-Label Fine-Tuning)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Accuracy
        ax3 = axes[2]
        ax3.plot(log['finetune']['val_acc'], color='green', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Validation Accuracy')
        ax3.set_title('Risk Classification Accuracy')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ training_curves.png")


def plot_feature_importance():
    """Plot feature importance from sample explanations."""
    print("  Generating feature importance plot...")

    expl_path = OUTPUT_DIR / 'sample_explanations.json'
    if not expl_path.exists():
        print("    ⚠ sample_explanations.json not found, skipping.")
        return

    with open(expl_path) as f:
        explanations = json.load(f)

    fig, axes = plt.subplots(1, len(explanations), figsize=(6*len(explanations), 5))
    if len(explanations) == 1:
        axes = [axes]

    for i, expl in enumerate(explanations):
        ax = axes[i]
        features = expl.get('top_features', [])
        if features:
            names = [f[0][:25] for f in features[:8]]
            values = [f[1] for f in features[:8]]
            colors_list = ['#e74c3c' if v > 0.15 else '#3498db' for v in values]
            ax.barh(names, values, color=colors_list, edgecolor='white')
            ax.set_xlabel('Attribution Score')
            ax.set_title(f"{expl.get('predicted_class', 'Unknown')}\n"
                        f"(Score: {expl.get('risk_score', 'N/A')})")
            ax.invert_yaxis()

    plt.suptitle('Feature Attribution per Risk Class', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ feature_importance.png")


def plot_gradcam():
    """Plot Grad-CAM heatmaps for sample patches."""
    print("  Generating Grad-CAM visualizations...")

    gcam_path = OUTPUT_DIR / 'gradcam_samples.npz'
    if not gcam_path.exists():
        print("    ⚠ gradcam_samples.npz not found, skipping.")
        return

    data = np.load(gcam_path)

    n_classes = len(data.files)
    fig, axes = plt.subplots(n_classes, 6, figsize=(18, 3 * n_classes))
    if n_classes == 1:
        axes = axes.reshape(1, -1)

    for i, key in enumerate(sorted(data.files)):
        heatmaps = data[key]  # (6, 64, 64)
        class_name = key.replace('_heatmaps', '')

        for j in range(min(6, heatmaps.shape[0])):
            ax = axes[i, j]
            ax.imshow(heatmaps[j], cmap='jet', vmin=0, vmax=1)
            ax.set_title(MONTH_NAMES[j], fontsize=10)
            ax.axis('off')

        axes[i, 0].set_ylabel(class_name, fontsize=12, rotation=90, labelpad=15)

    plt.suptitle('Grad-CAM: Which Spatial Regions Drive Risk Prediction', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'gradcam_examples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ gradcam_examples.png")


def generate_interactive_map(results_df):
    """Generate an interactive HTML map using folium."""
    print("  Generating interactive risk map...")

    try:
        import folium
        from folium.plugins import HeatMap
    except ImportError:
        print("    ⚠ folium not installed, skipping interactive map.")
        return

    center_lat = results_df['lat'].mean()
    center_lon = results_df['lon'].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=11,
                   tiles='OpenStreetMap')

    # Add risk markers
    color_map = {0: 'green', 1: 'orange', 2: 'red'}

    # Sample for performance (max 2000 markers)
    sample = results_df
    if len(results_df) > 2000:
        sample = results_df.sample(2000, random_state=42)

    for _, row in sample.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=3,
            color=color_map[row['risk_class']],
            fill=True,
            fill_opacity=0.7,
            popup=f"Risk: {row['risk_label']}<br>"
                  f"Score: {row['risk_score']:.3f}<br>"
                  f"Confidence: {row['confidence']:.3f}",
        ).add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
         background:white;padding:10px;border-radius:5px;border:2px solid gray;">
    <b>Risk Level</b><br>
    <i style="background:green;width:12px;height:12px;display:inline-block;border-radius:50%;"></i> Low Risk<br>
    <i style="background:orange;width:12px;height:12px;display:inline-block;border-radius:50%;"></i> Medium Risk<br>
    <i style="background:red;width:12px;height:12px;display:inline-block;border-radius:50%;"></i> High Risk
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(str(OUTPUT_DIR / 'risk_map.html'))
    print(f"    ✓ risk_map.html")


def print_credit_workflow():
    """Print how this would be used in a credit decision."""
    print(f"\n{'='*60}")
    print("  CREDIT DECISION WORKFLOW")
    print(f"{'='*60}")
    print("""
  How this risk signal integrates into farmer credit scoring:

  ┌─────────────────────────────────────────────────────┐
  │  FARMER APPLIES FOR CROP LOAN                      │
  │  (Location: lat/lon of farm)                        │
  └───────────────────┬─────────────────────────────────┘
                      ▼
  ┌─────────────────────────────────────────────────────┐
  │  AUTOMATED RISK ASSESSMENT (This System)            │
  │  • Satellite imagery → Crop health score            │
  │  • Weather data → Environmental stress score        │
  │  • Soil/terrain → Structural vulnerability score    │
  │  • DL model → Final risk classification             │
  │  • Explainability → Human-readable justification    │
  └───────────────────┬─────────────────────────────────┘
                      ▼
  ┌─────────────────────────────────────────────────────┐
  │  DECISION ROUTING                                   │
  │                                                     │
  │  LOW RISK  → Auto-approve, standard interest rate   │
  │  MED RISK  → Approve with crop insurance required   │
  │  HIGH RISK → Flag for manual field inspection       │
  │              Adjust collateral / interest rate       │
  │              Offer agricultural advisory services    │
  └─────────────────────────────────────────────────────┘

  Key advantages over traditional assessment:
  • Scalable: Assess thousands of fields simultaneously
  • Objective: Satellite data is unbiased
  • Timely: Updated every 5 days (Sentinel-2 revisit)
  • Explainable: Every prediction has a justification
  • Seasonal: Mid-season updates for portfolio monitoring
    """)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*60)
    print("  GENERATING VISUALIZATIONS")
    print("="*60)

    results_path = OUTPUT_DIR / 'risk_scores.csv'
    if not results_path.exists():
        print(f"  ✗ {results_path} not found. Run 04_inference.py first.")
        return

    results_df = pd.read_csv(results_path)
    print(f"  Loaded {len(results_df)} patches\n")

    plot_risk_map(results_df)
    plot_risk_distribution(results_df)
    plot_temporal_attention(results_df)
    plot_training_curves()
    plot_feature_importance()
    plot_gradcam()
    generate_interactive_map(results_df)
    print_credit_workflow()

    print(f"\n{'='*60}")
    print(f"  ALL VISUALIZATIONS COMPLETE")
    print(f"  Output files in: {OUTPUT_DIR}")
    for f in sorted(OUTPUT_DIR.iterdir()):
        if f.suffix in ['.png', '.html', '.csv', '.json']:
            size = f.stat().st_size / 1024
            print(f"    {f.name:40s} {size:.0f} KB")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
