"""
=============================================================================
Script 02: Preprocess Satellite Imagery into Patch Tensors
=============================================================================
Takes the downloaded GeoTIFF rasters and:
1. Reads all 6 monthly S2 + S1 composites
2. Stacks them into a multi-temporal tensor
3. Cuts into 64x64 pixel patches (each ~640m x 640m)
4. Filters out patches that are mostly nodata/water
5. Attaches auxiliary features (weather, soil, elevation) per patch
6. Saves as numpy arrays ready for PyTorch

Output: data/patches/patches.npy, patch_coords.csv, aux_features.npy
Runs on: Login node (CPU)
Time: ~5-10 minutes
=============================================================================
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
from pathlib import Path
from scipy.interpolate import griddata
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

SATELLITE_DIR = Path('data/satellite')
PATCH_DIR = Path('data/patches')
DATA_DIR = Path('data')
PATCH_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = 64  # 64x64 pixels = 1.28km x 1.28km at 20m resolution
PATCH_STRIDE = 48  # Overlap of 16 pixels for better coverage
MIN_VALID_FRACTION = 0.7  # Patches must have >70% valid pixels

MONTHS = [6, 7, 8, 9, 10, 11]
S2_BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'NDVI']  # 7 bands
S1_BANDS = ['VV', 'VH']  # 2 bands
TOTAL_BANDS = len(S2_BANDS) + len(S1_BANDS)  # 9 bands per month


# ============================================================================
# STEP 1: READ SATELLITE RASTERS
# ============================================================================

def read_rasters():
    """Read all monthly S2 and S1 composites into numpy arrays."""
    print("Step 1: Reading satellite rasters...")

    s2_data = {}
    s1_data = {}
    meta = None

    for month in MONTHS:
        # Sentinel-2
        s2_path = SATELLITE_DIR / f'S2_2024_{month:02d}.tif'
        if s2_path.exists():
            with rasterio.open(s2_path) as src:
                s2_data[month] = src.read()  # (bands, H, W)
                if meta is None:
                    meta = {
                        'transform': src.transform,
                        'crs': src.crs,
                        'height': src.height,
                        'width': src.width,
                    }
                print(f"  ✓ S2 month {month}: shape={s2_data[month].shape}, "
                      f"dtype={s2_data[month].dtype}")
        else:
            print(f"  ✗ S2 month {month}: NOT FOUND")

        # Sentinel-1
        s1_path = SATELLITE_DIR / f'S1_2024_{month:02d}.tif'
        if s1_path.exists():
            with rasterio.open(s1_path) as src:
                s1_data[month] = src.read()
                print(f"  ✓ S1 month {month}: shape={s1_data[month].shape}")
        else:
            print(f"  ✗ S1 month {month}: NOT FOUND")

    if not meta:
        raise RuntimeError("No satellite files found in data/satellite/!")

    print(f"\n  Raster size: {meta['width']} x {meta['height']} pixels")
    print(f"  CRS: {meta['crs']}")
    print(f"  S2 months available: {sorted(s2_data.keys())}")
    print(f"  S1 months available: {sorted(s1_data.keys())}")

    return s2_data, s1_data, meta


# ============================================================================
# STEP 2: BUILD MULTI-TEMPORAL TENSOR
# ============================================================================

def build_temporal_stack(s2_data, s1_data, meta):
    """
    Stack all months into a single tensor.
    Output shape: (6 months, 9 bands, H, W)
    Bands: [B2, B3, B4, B8, B11, B12, NDVI, VV, VH]
    """
    print("\nStep 2: Building multi-temporal stack...")

    H, W = meta['height'], meta['width']

    # Initialize with zeros (nodata)
    stack = np.zeros((6, TOTAL_BANDS, H, W), dtype=np.float32)

    for i, month in enumerate(MONTHS):
        # Sentinel-2 bands (7 bands)
        if month in s2_data:
            s2 = s2_data[month].astype(np.float32)
            n_s2_bands = min(s2.shape[0], len(S2_BANDS))
            stack[i, :n_s2_bands, :, :] = s2[:n_s2_bands]

        # Sentinel-1 bands (2 bands, appended after S2)
        if month in s1_data:
            s1 = s1_data[month].astype(np.float32)
            n_s1_bands = min(s1.shape[0], len(S1_BANDS))
            stack[i, len(S2_BANDS):len(S2_BANDS)+n_s1_bands, :, :] = s1[:n_s1_bands]

    print(f"  Temporal stack shape: {stack.shape}")
    print(f"  = ({stack.shape[0]} months, {stack.shape[1]} bands, "
          f"{stack.shape[2]}H, {stack.shape[3]}W)")
    print(f"  Memory: {stack.nbytes / 1024 / 1024:.1f} MB")

    return stack


# ============================================================================
# STEP 3: CUT INTO PATCHES
# ============================================================================

def extract_patches(stack, meta):
    """
    Cut the large raster into 64x64 patches.
    Filter out patches that are mostly nodata (water, cloud gaps, edges).
    Record center coordinates of each valid patch.
    """
    print("\nStep 3: Extracting patches...")

    T, C, H, W = stack.shape
    patches = []
    coords = []

    n_rows = (H - PATCH_SIZE) // PATCH_STRIDE + 1
    n_cols = (W - PATCH_SIZE) // PATCH_STRIDE + 1
    total_possible = n_rows * n_cols
    print(f"  Grid: {n_rows} rows x {n_cols} cols = {total_possible} possible patches")

    for row_start in range(0, H - PATCH_SIZE + 1, PATCH_STRIDE):
        for col_start in range(0, W - PATCH_SIZE + 1, PATCH_STRIDE):
            patch = stack[:, :, row_start:row_start+PATCH_SIZE,
                         col_start:col_start+PATCH_SIZE]

            # Check valid pixel fraction
            # A pixel is "valid" if it has non-zero values in at least
            # the NDVI band (index 6) for at least 3 months
            ndvi_band = patch[:, 6, :, :]  # (6, 64, 64)
            valid_months_per_pixel = np.sum(ndvi_band != 0, axis=0)  # (64, 64)
            valid_fraction = np.mean(valid_months_per_pixel >= 3)

            if valid_fraction >= MIN_VALID_FRACTION:
                patches.append(patch)

                # Get center coordinates
                center_row = row_start + PATCH_SIZE // 2
                center_col = col_start + PATCH_SIZE // 2
                lon, lat = xy(meta['transform'], center_row, center_col)
                coords.append({
                    'patch_id': len(patches) - 1,
                    'lat': lat,
                    'lon': lon,
                    'row_start': row_start,
                    'col_start': col_start,
                    'valid_fraction': round(valid_fraction, 3),
                })

    patches = np.array(patches, dtype=np.float32)
    coords_df = pd.DataFrame(coords)

    print(f"  Valid patches: {len(patches)} / {total_possible} "
          f"({100*len(patches)/total_possible:.1f}%)")
    print(f"  Patch tensor shape: {patches.shape}")
    print(f"  = ({patches.shape[0]} patches, {patches.shape[1]} months, "
          f"{patches.shape[2]} bands, {patches.shape[3]}x{patches.shape[4]} pixels)")

    return patches, coords_df


# ============================================================================
# STEP 4: NORMALIZE PATCHES
# ============================================================================

def normalize_patches(patches):
    """
    Normalize each band to [0, 1] range across all patches.
    Handles nodata values: Sentinel-2 int16 uses 0 or -32768 as nodata.
    Stores normalization stats for inference.
    """
    print("\nStep 4: Normalizing patches...")

    N, T, C, H, W = patches.shape
    norm_stats = {}

    for c in range(C):
        band_data = patches[:, :, c, :, :]
        
        # Treat -32768 (int16 min) and 0 as nodata for S2 bands
        # For S1 bands (VV/VH, indices 7-8), 0 is nodata
        valid_mask = (band_data > -32000) & (band_data != 0)

        if valid_mask.any():
            vmin = np.percentile(band_data[valid_mask], 2)
            vmax = np.percentile(band_data[valid_mask], 98)
        else:
            vmin, vmax = 0, 1

        if vmax - vmin < 1e-6:
            vmax = vmin + 1

        # Normalize valid pixels
        normalized = np.clip((band_data - vmin) / (vmax - vmin), 0, 1)
        # Set nodata pixels to 0
        normalized[~valid_mask] = 0
        patches[:, :, c, :, :] = normalized

        band_name = (S2_BANDS + S1_BANDS)[c] if c < TOTAL_BANDS else f'band_{c}'
        valid_pct = 100 * valid_mask.sum() / valid_mask.size
        norm_stats[band_name] = {'min': float(vmin), 'max': float(vmax)}
        print(f"  Band {c} ({band_name}): [{vmin:.1f}, {vmax:.1f}] → [0, 1] ({valid_pct:.1f}% valid)")

    return patches, norm_stats


# ============================================================================
# STEP 5: ATTACH AUXILIARY FEATURES
# ============================================================================

def build_auxiliary_features(coords_df):
    """
    For each patch, compute auxiliary features from weather, soil, elevation.
    These are non-image features that get concatenated in the model's fusion layer.
    """
    print("\nStep 5: Building auxiliary features...")

    features_list = []

    # --- Load weather data ---
    weather_df = pd.read_csv(DATA_DIR / 'weather_data.csv')
    weather_monthly_avg = pd.read_csv(DATA_DIR / 'weather_monthly_avg_2015_2023.csv')

    # Compute seasonal weather features per location
    weather_features = {}
    for loc in weather_df['location'].unique():
        loc_data = weather_df[weather_df['location'] == loc]
        lat, lon = loc_data.iloc[0]['lat'], loc_data.iloc[0]['lon']

        # Total season rainfall
        total_precip = loc_data['precipitation_mm'].sum()

        # Rainfall anomaly (vs long-term average)
        lt_total = weather_monthly_avg['avg_monthly_precip_mm'].sum()
        rain_anomaly = (total_precip - lt_total) / lt_total if lt_total > 0 else 0

        # Max dry spell (consecutive days < 2mm rain)
        precip_series = loc_data['precipitation_mm'].fillna(0).values
        max_dry = 0
        current_dry = 0
        for p in precip_series:
            if p < 2.0:
                current_dry += 1
                max_dry = max(max_dry, current_dry)
            else:
                current_dry = 0

        # Heat stress days (> 38°C)
        heat_days = (loc_data['temp_max_c'] > 38).sum()

        # Mean temperature
        mean_temp = loc_data['temp_mean_c'].mean()

        # Excess rain events (daily > 50mm)
        excess_rain_events = (loc_data['precipitation_mm'] > 50).sum()

        # Sept rainfall (critical flowering month)
        sept_data = loc_data[loc_data['date'].str.startswith('2024-09')]
        sept_precip = sept_data['precipitation_mm'].sum() if len(sept_data) > 0 else 0

        weather_features[(round(lat, 1), round(lon, 1))] = {
            'total_precip_mm': total_precip,
            'rain_anomaly_frac': rain_anomaly,
            'max_dry_spell_days': max_dry,
            'heat_stress_days': heat_days,
            'mean_temp_c': mean_temp,
            'excess_rain_events': excess_rain_events,
            'sept_precip_mm': sept_precip,
        }

    # --- Load soil data ---
    soil_df = pd.read_csv(DATA_DIR / 'soil_data.csv')
    soil_points = soil_df[['lat', 'lon']].values
    soil_features_names = [c for c in soil_df.columns
                           if c not in ['lat', 'lon', 'soil_type', 'soil_type_estimated',
                                        'drainage_label', 'source']]

    # --- Load elevation data ---
    elev_df = pd.read_csv(DATA_DIR / 'elevation_data.csv')
    elev_points = elev_df[['lat', 'lon']].values

    # --- For each patch, interpolate nearest auxiliary features ---
    for _, patch_row in coords_df.iterrows():
        plat, plon = patch_row['lat'], patch_row['lon']
        feat = {}

        # Weather: find nearest weather station
        best_dist = float('inf')
        best_wf = None
        for (wlat, wlon), wf in weather_features.items():
            d = (plat - wlat)**2 + (plon - wlon)**2
            if d < best_dist:
                best_dist = d
                best_wf = wf
        if best_wf:
            feat.update(best_wf)

        # Soil: nearest neighbor interpolation
        if len(soil_points) > 0:
            dists = np.sqrt((soil_points[:, 0] - plat)**2 +
                           (soil_points[:, 1] - plon)**2)
            nearest_idx = np.argmin(dists)
            for col in soil_features_names:
                val = soil_df.iloc[nearest_idx][col]
                if isinstance(val, (int, float, np.integer, np.floating)):
                    feat[f'soil_{col}'] = float(val)

        # Elevation: nearest neighbor
        if len(elev_points) > 0:
            dists = np.sqrt((elev_points[:, 0] - plat)**2 +
                           (elev_points[:, 1] - plon)**2)
            nearest_idx = np.argmin(dists)
            feat['elevation_m'] = float(elev_df.iloc[nearest_idx]['elevation_m'])
            feat['slope_deg'] = float(elev_df.iloc[nearest_idx]['slope_degrees'])

        features_list.append(feat)

    # Convert to DataFrame then numpy
    aux_df = pd.DataFrame(features_list)

    # Fill NaN with column median
    for col in aux_df.columns:
        if aux_df[col].isna().any():
            aux_df[col] = aux_df[col].fillna(aux_df[col].median())

    # Normalize auxiliary features to [0, 1]
    aux_stats = {}
    for col in aux_df.columns:
        vmin = aux_df[col].min()
        vmax = aux_df[col].max()
        if vmax - vmin < 1e-6:
            vmax = vmin + 1
        aux_df[col] = (aux_df[col] - vmin) / (vmax - vmin)
        aux_stats[col] = {'min': float(vmin), 'max': float(vmax)}

    aux_array = aux_df.values.astype(np.float32)

    print(f"  Auxiliary features: {aux_array.shape[1]} dimensions per patch")
    print(f"  Features: {list(aux_df.columns)}")

    return aux_array, aux_stats, list(aux_df.columns)


# ============================================================================
# STEP 6: GENERATE PSEUDO-LABELS
# ============================================================================

def generate_pseudo_labels(patches, aux_features, aux_feature_names, coords_df):
    """
    Generate risk pseudo-labels using UNSUPERVISED CLUSTERING.
    
    Instead of a weighted formula with arbitrary weights, we:
    1. Extract meaningful features from satellite patches + auxiliary data
    2. Run KMeans clustering (k=3) to find natural groupings
    3. Assign risk labels by ranking clusters on crop health (NDVI)
    
    This is data-driven: the clusters emerge from actual patterns in the
    satellite imagery, not from hand-tuned rules.
    """
    print("\nStep 6: Generating pseudo-labels via unsupervised clustering...")
    
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    N = len(patches)
    
    # ================================================================
    # EXTRACT FEATURES FROM SATELLITE PATCHES
    # ================================================================
    print("  Extracting features from satellite patches...")
    
    patch_features = []
    
    # NDVI is band 6, VV is band 7, VH is band 8
    ndvi_all = patches[:, :, 6, :, :]   # (N, 6, 64, 64)
    vv_all = patches[:, :, 7, :, :]     # (N, 6, 64, 64)
    vh_all = patches[:, :, 8, :, :]     # (N, 6, 64, 64)
    
    for i in range(N):
        ndvi = ndvi_all[i]  # (6, 64, 64)
        vv = vv_all[i]
        vh = vh_all[i]
        
        # --- NDVI-derived features ---
        # Per-month spatial means (valid pixels only)
        ndvi_monthly = []
        for m in range(6):
            valid = ndvi[m][ndvi[m] > 0.01]
            ndvi_monthly.append(valid.mean() if len(valid) > 0 else 0)
        ndvi_monthly = np.array(ndvi_monthly)
        
        ndvi_valid = ndvi[ndvi > 0.01]
        ndvi_mean = ndvi_valid.mean() if len(ndvi_valid) > 0 else 0
        ndvi_peak = ndvi_monthly.max()
        ndvi_peak_month = ndvi_monthly.argmax()
        
        # Decline rate: peak to last month
        if ndvi_peak > 0.01 and ndvi_peak_month < 5:
            decline = (ndvi_monthly[ndvi_peak_month] - ndvi_monthly[5]) / (ndvi_peak + 1e-6)
        else:
            decline = 0
        
        # Spatial variability (coefficient of variation within patch)
        spatial_cv = ndvi_valid.std() / (ndvi_mean + 1e-6) if len(ndvi_valid) > 10 else 0
        
        # Temporal variability
        valid_months = ndvi_monthly[ndvi_monthly > 0.01]
        temporal_std = valid_months.std() if len(valid_months) > 1 else 0
        
        # Growth trajectory: early season vs late season
        early = ndvi_monthly[:3].mean()  # Jun-Aug
        late = ndvi_monthly[3:].mean()   # Sep-Nov
        season_ratio = late / (early + 1e-6)
        
        # --- SAR-derived features ---
        vv_valid = vv[vv != 0]
        vh_valid = vh[vh != 0]
        vv_mean = vv_valid.mean() if len(vv_valid) > 0 else 0
        vh_mean = vh_valid.mean() if len(vh_valid) > 0 else 0
        
        # VV/VH ratio trend (proxy for crop structure change)
        vv_monthly = []
        for m in range(6):
            v = vv[m][vv[m] != 0]
            vv_monthly.append(v.mean() if len(v) > 0 else 0)
        vv_monthly = np.array(vv_monthly)
        vv_trend = vv_monthly[5] - vv_monthly[0] if vv_monthly[0] != 0 else 0
        
        # Fraction of valid optical pixels (cloud-free coverage)
        optical_valid_frac = (ndvi[ndvi > 0.01].size) / max(ndvi.size, 1)
        
        patch_features.append({
            'ndvi_mean': ndvi_mean,
            'ndvi_peak': ndvi_peak,
            'ndvi_peak_month': ndvi_peak_month,
            'ndvi_decline': decline,
            'ndvi_spatial_cv': spatial_cv,
            'ndvi_temporal_std': temporal_std,
            'ndvi_season_ratio': season_ratio,
            'ndvi_early': early,
            'ndvi_late': late,
            'vv_mean': vv_mean,
            'vh_mean': vh_mean,
            'vv_trend': vv_trend,
            'optical_coverage': optical_valid_frac,
        })
    
    # Combine patch-derived features with auxiliary features
    patch_df = pd.DataFrame(patch_features)
    aux_df = pd.DataFrame(aux_features, columns=aux_feature_names)
    
    # Select the most informative auxiliary features
    aux_cols_to_use = [c for c in aux_feature_names if any(k in c for k in 
        ['precip', 'rain', 'dry_spell', 'heat', 'drainage', 'elevation', 'soc', 'fertility'])]
    
    combined = pd.concat([patch_df, aux_df[aux_cols_to_use]], axis=1)
    
    # Fill NaN and standardize
    combined = combined.fillna(0)
    feature_names = list(combined.columns)
    X = StandardScaler().fit_transform(combined.values)
    
    print(f"  Clustering on {X.shape[1]} features ({len(patch_df.columns)} from satellite + {len(aux_cols_to_use)} auxiliary)")
    
    # ================================================================
    # KMEANS CLUSTERING
    # ================================================================
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10, max_iter=300)
    clusters = kmeans.fit_predict(X)
    
    # ================================================================
    # MAP CLUSTERS TO RISK LEVELS
    # ================================================================
    # Rank clusters by mean NDVI: highest NDVI cluster = Low Risk
    cluster_ndvi = {}
    for c_id in range(3):
        mask = clusters == c_id
        cluster_ndvi[c_id] = patch_df.loc[mask, 'ndvi_mean'].mean()
    
    # Sort: highest NDVI → Low Risk (0), lowest NDVI → High Risk (2)
    sorted_clusters = sorted(cluster_ndvi.keys(), key=lambda k: cluster_ndvi[k], reverse=True)
    cluster_to_risk = {sorted_clusters[0]: 0, sorted_clusters[1]: 1, sorted_clusters[2]: 2}
    
    labels = np.array([cluster_to_risk[c] for c in clusters], dtype=np.int64)
    
    # ================================================================
    # COMPUTE CONTINUOUS RISK SCORES
    # ================================================================
    # Distance from the "Low Risk" cluster center = risk score
    low_risk_center = kmeans.cluster_centers_[sorted_clusters[0]]
    high_risk_center = kmeans.cluster_centers_[sorted_clusters[2]]
    
    # Project each point onto the Low→High risk axis
    risk_axis = high_risk_center - low_risk_center
    risk_axis_norm = risk_axis / (np.linalg.norm(risk_axis) + 1e-6)
    
    risk_scores = np.array([
        np.dot(X[i] - low_risk_center, risk_axis_norm) for i in range(N)
    ], dtype=np.float32)
    
    # Normalize to [0, 1]
    rs_min, rs_max = risk_scores.min(), risk_scores.max()
    if rs_max - rs_min > 1e-6:
        risk_scores = (risk_scores - rs_min) / (rs_max - rs_min)
    
    # ================================================================
    # DIAGNOSTICS
    # ================================================================
    print(f"\n  Cluster characteristics:")
    risk_names = ['Low Risk', 'Medium Risk', 'High Risk']
    for risk_level in range(3):
        orig_cluster = sorted_clusters[risk_level]
        mask = labels == risk_level
        count = mask.sum()
        ndvi_m = patch_df.loc[mask, 'ndvi_mean'].mean()
        ndvi_p = patch_df.loc[mask, 'ndvi_peak'].mean()
        decline_m = patch_df.loc[mask, 'ndvi_decline'].mean()
        cv_m = patch_df.loc[mask, 'ndvi_spatial_cv'].mean()
        print(f"    {risk_names[risk_level]:12s}: {count:5d} patches ({100*count/N:.1f}%) | "
              f"NDVI mean={ndvi_m:.3f}, peak={ndvi_p:.3f}, decline={decline_m:.3f}, spatial_cv={cv_m:.3f}")
    
    print(f"\n  Risk score range: [{risk_scores.min():.3f}, {risk_scores.max():.3f}]")
    print(f"  Features used for clustering: {feature_names}")
    
    # Verify clusters make sense (Low Risk should have highest NDVI)
    ndvi_by_risk = [patch_df.loc[labels == r, 'ndvi_mean'].mean() for r in range(3)]
    if ndvi_by_risk[0] > ndvi_by_risk[1] > ndvi_by_risk[2]:
        print(f"\n  ✓ Cluster ordering validated: Low Risk NDVI ({ndvi_by_risk[0]:.3f}) > "
              f"Medium ({ndvi_by_risk[1]:.3f}) > High ({ndvi_by_risk[2]:.3f})")
    else:
        print(f"\n  ⚠ Cluster NDVI ordering: {ndvi_by_risk} — check if mapping is correct")
    
    return labels, risk_scores


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*60)
    print("  PREPROCESSING SATELLITE DATA INTO PATCHES")
    print("="*60)

    # Step 1: Read rasters
    s2_data, s1_data, meta = read_rasters()

    # Step 2: Build temporal stack
    stack = build_temporal_stack(s2_data, s1_data, meta)

    # Step 3: Extract patches
    patches, coords_df = extract_patches(stack, meta)

    # Free memory
    del stack, s2_data, s1_data

    # Step 4: Normalize
    patches, norm_stats = normalize_patches(patches)

    # Step 5: Auxiliary features
    aux_features, aux_stats, aux_names = build_auxiliary_features(coords_df)

    # Step 6: Pseudo-labels
    labels, risk_scores = generate_pseudo_labels(
        patches, aux_features, aux_names, coords_df
    )

    # Save everything
    print("\nSaving outputs...")

    np.save(PATCH_DIR / 'patches.npy', patches)
    print(f"  ✓ patches.npy: {patches.shape} ({patches.nbytes/1024/1024:.1f}MB)")

    np.save(PATCH_DIR / 'aux_features.npy', aux_features)
    print(f"  ✓ aux_features.npy: {aux_features.shape}")

    np.save(PATCH_DIR / 'pseudo_labels.npy', labels)
    np.save(PATCH_DIR / 'risk_scores_pseudo.npy', risk_scores)
    print(f"  ✓ pseudo_labels.npy: {labels.shape}")

    coords_df.to_csv(PATCH_DIR / 'patch_coords.csv', index=False)
    print(f"  ✓ patch_coords.csv: {len(coords_df)} patches")

    # Save metadata
    metadata = {
        'patch_size': PATCH_SIZE,
        'patch_stride': PATCH_STRIDE,
        'n_patches': len(patches),
        'n_months': 6,
        'n_bands': TOTAL_BANDS,
        's2_bands': S2_BANDS,
        's1_bands': S1_BANDS,
        'aux_feature_names': aux_names,
        'norm_stats': norm_stats,
        'aux_stats': aux_stats,
        'raster_meta': {
            'height': meta['height'],
            'width': meta['width'],
            'crs': str(meta['crs']),
        }
    }
    with open(PATCH_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ metadata.json")

    print(f"\n{'='*60}")
    print(f"  PREPROCESSING COMPLETE")
    print(f"  {len(patches)} patches ready for training")
    print(f"  Run: sbatch run_gpu.sh (or python scripts/03_train_model.py)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()