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
    Generate risk pseudo-labels using domain knowledge rules.
    These become training targets for the DL model.

    APPROACH: Compute a continuous risk score from multiple signals,
    then use PERCENTILE-BASED thresholds to ensure meaningful class 
    distribution (~50% Low, ~30% Medium, ~20% High).

    This reflects reality: in any given season, most fields do okay,
    some face moderate stress, and a smaller fraction fails badly.
    """
    print("\nStep 6: Generating pseudo-labels...")

    N = len(patches)

    # ================================================================
    # COMPONENT 1: CROP HEALTH SCORE (from NDVI, weight=0.45)
    # ================================================================
    # Extract NDVI time series per patch (band 6, spatial mean)
    ndvi_ts = patches[:, :, 6, :, :].mean(axis=(2, 3))  # (N, 6)

    crop_health = np.zeros(N, dtype=np.float32)

    for i in range(N):
        ndvi = ndvi_ts[i]
        valid = ndvi[ndvi > 0.01]  # ignore nodata

        if len(valid) < 2:
            crop_health[i] = 1.0  # No valid data = max risk
            continue

        ndvi_mean = valid.mean()
        ndvi_peak = valid.max()
        ndvi_std = valid.std()

        # Peak timing: cotton should peak around month 3-4 (Aug-Sep)
        peak_month = ndvi.argmax()
        peak_timing_penalty = abs(peak_month - 3) * 0.05  # Penalty for off-schedule peak

        # Decline rate from peak to harvest
        if ndvi_peak > 0.01 and peak_month < 5:
            decline = (ndvi[peak_month] - ndvi[5]) / ndvi[peak_month]
            decline = max(0, decline)
        else:
            decline = 0.5

        # Spatial variability within patch (band 6)
        ndvi_spatial = patches[i, :, 6, :, :]  # (6, 64, 64)
        valid_spatial = ndvi_spatial[ndvi_spatial > 0.01]
        spatial_cv = valid_spatial.std() / (valid_spatial.mean() + 1e-6) if len(valid_spatial) > 0 else 0.5

        # Combine: higher score = MORE risk
        # Invert NDVI (low NDVI = high risk), add decline and variability
        crop_health[i] = (
            0.35 * (1.0 - ndvi_mean) +          # Low overall greenness
            0.25 * min(decline, 1.0) +            # Sharp decline
            0.15 * min(spatial_cv, 1.0) +          # Within-field variability
            0.15 * (1.0 - ndvi_peak) +             # Low peak vigor
            0.10 * peak_timing_penalty              # Off-schedule growth
        )

    # ================================================================
    # COMPONENT 2: ENVIRONMENTAL STRESS (from weather, weight=0.35)
    # ================================================================
    env_stress = np.zeros(N, dtype=np.float32)

    for i in range(N):
        aux = dict(zip(aux_feature_names, aux_features[i]))

        # All features are already normalized to [0, 1]
        rain_anom = aux.get('rain_anomaly_frac', 0.5)
        dry_spell = aux.get('max_dry_spell_days', 0.5)
        heat = aux.get('heat_stress_days', 0.5)
        excess_rain = aux.get('excess_rain_events', 0.0)
        sept_precip = aux.get('sept_precip_mm', 0.5)

        env_stress[i] = (
            0.25 * rain_anom +                     # Rainfall anomaly
            0.25 * dry_spell +                     # Drought duration
            0.20 * heat +                          # Heat stress
            0.15 * excess_rain +                   # Flood events
            0.15 * (1.0 - sept_precip)              # Low Sept rain (critical for flowering)
        )

    # ================================================================
    # COMPONENT 3: STRUCTURAL VULNERABILITY (soil/terrain, weight=0.20)
    # ================================================================
    struct_vuln = np.zeros(N, dtype=np.float32)

    for i in range(N):
        aux = dict(zip(aux_feature_names, aux_features[i]))

        drainage = aux.get('soil_drainage_class', 0.5)
        soc = aux.get('soil_soc_pct_0_15cm', 0.5)
        fertility = aux.get('soil_fertility_index', 0.5)
        elevation = aux.get('elevation_m', 0.5)
        slope = aux.get('slope_deg', 0.5)

        struct_vuln[i] = (
            0.30 * (1.0 - drainage) +              # Poor drainage = higher risk
            0.25 * (1.0 - soc) +                   # Low organic carbon
            0.20 * (1.0 - fertility) +              # Low fertility
            0.15 * (1.0 - elevation) +              # Low-lying = flood prone
            0.10 * (1.0 - slope)                    # Flat = waterlogging
        )

    # ================================================================
    # COMBINE INTO FINAL SCORE
    # ================================================================
    risk_scores = (
        0.45 * crop_health +
        0.35 * env_stress +
        0.20 * struct_vuln
    )

    # Rescale to use full [0, 1] range
    rs_min, rs_max = risk_scores.min(), risk_scores.max()
    if rs_max - rs_min > 1e-6:
        risk_scores = (risk_scores - rs_min) / (rs_max - rs_min)

    # ================================================================
    # PERCENTILE-BASED CLASSIFICATION
    # ================================================================
    # Target distribution: ~50% Low, ~30% Medium, ~20% High
    # This is realistic: most fields survive, some struggle, few fail
    p50 = np.percentile(risk_scores, 50)
    p80 = np.percentile(risk_scores, 80)

    labels = np.zeros(N, dtype=np.int64)
    labels[risk_scores < p50] = 0   # Low risk (bottom 50%)
    labels[(risk_scores >= p50) & (risk_scores < p80)] = 1  # Medium risk (50-80%)
    labels[risk_scores >= p80] = 2  # High risk (top 20%)

    # Print distribution and diagnostics
    print(f"\n  Component score ranges:")
    print(f"    Crop health:  [{crop_health.min():.3f}, {crop_health.max():.3f}] (mean: {crop_health.mean():.3f})")
    print(f"    Env stress:   [{env_stress.min():.3f}, {env_stress.max():.3f}] (mean: {env_stress.mean():.3f})")
    print(f"    Struct vuln:  [{struct_vuln.min():.3f}, {struct_vuln.max():.3f}] (mean: {struct_vuln.mean():.3f})")
    print(f"  Final risk scores: [{risk_scores.min():.3f}, {risk_scores.max():.3f}]")
    print(f"  Thresholds: Low < {p50:.3f} < Medium < {p80:.3f} < High")
    print(f"\n  Label distribution:")
    for label, name in enumerate(['Low Risk', 'Medium Risk', 'High Risk']):
        count = (labels == label).sum()
        print(f"    {name}: {count} patches ({100*count/N:.1f}%)")

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
