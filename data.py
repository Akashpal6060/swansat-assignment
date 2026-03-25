"""
=============================================================================
Script 01: Download Sentinel-2 + Sentinel-1 Satellite Imagery via GEE
=============================================================================
Downloads REAL satellite imagery for Guntur district, Kharif 2024 season.
- Sentinel-2 L2A: 6 monthly cloud-free median composites (B2,B3,B4,B8,B11,B12)
- Sentinel-1 GRD: 6 monthly median composites (VV, VH)
- Also computes and exports NDVI per month

Output: 12 GeoTIFF files in data/satellite/
Runs on: Login node (CPU, needs internet + GEE auth)
Time: ~15-30 minutes
=============================================================================
"""

import ee
import os
import time
import requests
import numpy as np
from pathlib import Path
import geemap  

# ============================================================================
# CONFIGURATION
# ============================================================================

GEE_PROJECT = 'swansat'  # Your GEE project ID
OUTPUT_DIR = Path('data/satellite')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Guntur District bounding box
# Using a slightly smaller focused area for manageable download size
# This covers the core agricultural zone of Guntur (~50km x 50km)
GUNTUR_BBOX = [79.8, 16.0, 80.6, 16.5]  # [west, south, east, north]

# Season
MONTHS = [
    (2024, 6, "Jun"), (2024, 7, "Jul"), (2024, 8, "Aug"),
    (2024, 9, "Sep"), (2024, 10, "Oct"), (2024, 11, "Nov")
]

# Sentinel-2 bands to download
S2_BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']  # Blue, Green, Red, NIR, SWIR1, SWIR2
S2_SCALE = 10  # meters per pixel

# Sentinel-1 bands
S1_BANDS = ['VV', 'VH']
S1_SCALE = 10

# ============================================================================
# INITIALIZE GEE
# ============================================================================

print("Initializing Google Earth Engine...")
ee.Initialize(project=GEE_PROJECT)
print(f"  ✓ Connected to project: {GEE_PROJECT}")

# Define region of interest
roi = ee.Geometry.Rectangle(GUNTUR_BBOX)
print(f"  ROI: {GUNTUR_BBOX}")

# Calculate expected image dimensions
width_deg = GUNTUR_BBOX[2] - GUNTUR_BBOX[0]
height_deg = GUNTUR_BBOX[3] - GUNTUR_BBOX[1]
width_px = int(width_deg * 111320 * np.cos(np.radians(16.25)) / S2_SCALE)
height_px = int(height_deg * 110540 / S2_SCALE)
print(f"  Expected image size: ~{width_px} x {height_px} pixels at {S2_SCALE}m")
print(f"  Coverage area: ~{width_deg*111:.0f}km x {height_deg*111:.0f}km")


# ============================================================================
# SENTINEL-2 FUNCTIONS
# ============================================================================

def mask_s2_clouds(image):
    """Cloud masking using the SCL (Scene Classification Layer) band."""
    scl = image.select('SCL')
    # Keep only: vegetation(4), bare soil(5), water(6)
    # Remove: cloud shadow(3), cloud medium prob(8), cloud high prob(9), cirrus(10)
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    return image.updateMask(mask)


def add_ndvi(image):
    """Add NDVI band to Sentinel-2 image."""
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)


def get_s2_monthly_composite(year, month):
    """
    Create a cloud-free monthly median composite from Sentinel-2 L2A.
    Uses SCL band for cloud masking, then takes pixel-wise median.
    """
    start_date = f'{year}-{month:02d}-01'
    if month == 12:
        end_date = f'{year+1}-01-01'
    else:
        end_date = f'{year}-{month+1:02d}-01'

    # Sentinel-2 Surface Reflectance (L2A)
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate(start_date, end_date)
        .filterBounds(roi)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
        .map(mask_s2_clouds)
        .map(add_ndvi))

    # Get image count for logging
    count = collection.size().getInfo()

    # Median composite (robust to remaining cloud artifacts)
    composite = collection.select(S2_BANDS + ['NDVI']).median()

    # Scale reflectance values to 0-10000 range and convert to int16
    # (reduces file size, standard for Sentinel-2)
    composite_scaled = composite.select(S2_BANDS).toInt16()
    ndvi_scaled = composite.select('NDVI').multiply(10000).toInt16()

    final = composite_scaled.addBands(ndvi_scaled)

    return final, count


# ============================================================================
# SENTINEL-1 FUNCTIONS
# ============================================================================

def get_s1_monthly_composite(year, month):
    """
    Create monthly median composite from Sentinel-1 GRD (SAR).
    SAR is cloud-free by nature - this gives us data even during monsoon.
    """
    start_date = f'{year}-{month:02d}-01'
    if month == 12:
        end_date = f'{year+1}-01-01'
    else:
        end_date = f'{year}-{month+1:02d}-01'

    collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterDate(start_date, end_date)
        .filterBounds(roi)
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .select(S1_BANDS))

    count = collection.size().getInfo()

    # Median composite with speckle reduction effect
    composite = collection.median()

    # Scale to int16 (multiply by 100 to preserve decimal precision)
    composite_scaled = composite.multiply(100).toInt16()

    return composite_scaled, count


# ============================================================================
# DOWNLOAD FUNCTION
# ============================================================================

def download_image(image, filename, scale, bands_info=""):
    """Download a GEE image as GeoTIFF using geemap to bypass size limits."""
    filepath = OUTPUT_DIR / filename

    if filepath.exists():
        size = filepath.stat().st_size
        if size > 10000:  # > 10KB means likely valid
            print(f"    ⏭  Already exists: {filename} ({size/1024/1024:.1f}MB)")
            return True

    try:
        print(f"    ⬇  Downloading {filename} {bands_info} (chunking automatically)...")
        
        # geemap automatically calculates a grid, downloads tiles < 48MB, 
        # and merges them locally into a single GeoTIFF file.
        geemap.download_ee_image(
            image=image, 
            filename=str(filepath), 
            scale=scale, 
            region=roi, 
            crs='EPSG:4326'
        )

        # Verify the file was saved
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"    ✓  Saved: {filename} ({size_mb:.1f}MB)")
            return True
        else:
            print(f"    ✗  Failed: {filename} - File not created.")
            return False

    except Exception as e:
        print(f"    ✗  Failed: {filename} - {e}")
        if filepath.exists():
            filepath.unlink() # Clean up corrupted/partial files
        return False


# ============================================================================
# MAIN DOWNLOAD LOOP
# ============================================================================

def main():
    print("\n" + "="*60)
    print("  DOWNLOADING SATELLITE IMAGERY")
    print("  District: Guntur (core agricultural zone)")
    print(f"  Area: {GUNTUR_BBOX}")
    print(f"  Season: Jun-Nov 2024")
    print("="*60)

    results = {"s2": [], "s1": []}

    # Download Sentinel-2 composites
    print("\n--- SENTINEL-2 (Optical: B2,B3,B4,B8,B11,B12 + NDVI) ---")
    for year, month, month_name in MONTHS:
        print(f"\n  {month_name} {year}:")
        composite, count = get_s2_monthly_composite(year, month)
        print(f"    Images in collection: {count}")

        if count == 0:
            print(f"    ⚠  No images found for {month_name}! (heavy cloud cover)")
            results["s2"].append(False)
            continue

        filename = f'S2_{year}_{month:02d}.tif'
        success = download_image(
            composite, filename, S2_SCALE,
            f"[{count} images → median composite, 7 bands]"
        )
        results["s2"].append(success)

    # Download Sentinel-1 composites
    print("\n--- SENTINEL-1 (SAR: VV, VH - cloud-free) ---")
    for year, month, month_name in MONTHS:
        print(f"\n  {month_name} {year}:")
        composite, count = get_s1_monthly_composite(year, month)
        print(f"    Images in collection: {count}")

        if count == 0:
            print(f"    ⚠  No SAR images for {month_name}!")
            results["s1"].append(False)
            continue

        filename = f'S1_{year}_{month:02d}.tif'
        success = download_image(
            composite, filename, S1_SCALE,
            f"[{count} images → median composite, 2 bands]"
        )
        results["s1"].append(success)

    # Summary
    print("\n" + "="*60)
    print("  DOWNLOAD SUMMARY")
    print("="*60)
    s2_ok = sum(results["s2"])
    s1_ok = sum(results["s1"])
    print(f"  Sentinel-2: {s2_ok}/6 months downloaded")
    print(f"  Sentinel-1: {s1_ok}/6 months downloaded")

    # List files
    print(f"\n  Files in {OUTPUT_DIR}:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"    {f.name:30s} {size_mb:.1f} MB")

    if s2_ok < 4:
        print("\n  ⚠  Less than 4 S2 months downloaded.")
        print("     Sentinel-1 SAR data will fill the gaps (cloud-free).")

    if s2_ok + s1_ok >= 8:
        print("\n  ✓  Sufficient data for model training. Proceed to Script 02.")
    else:
        print("\n  ✗  Insufficient data. Check GEE auth and try again.")


if __name__ == '__main__':
    main()