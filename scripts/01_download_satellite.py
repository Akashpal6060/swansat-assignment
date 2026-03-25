"""
=============================================================================
Script 01: Download Sentinel-2 + Sentinel-1 Satellite Imagery via GEE
=============================================================================
Downloads real satellite imagery in tiles to stay under GEE's 50MB limit.

Key fixes:
- Uses 20m resolution (still excellent for 64x64 patches = 1.28km fields)
- Splits the area into longitude strips for download
- Merges tiles back into single GeoTIFF per month
- Skips already-downloaded files on re-run

Output: 12 GeoTIFF files in data/satellite/
Runs on: Login node (CPU, needs internet + GEE auth)
Time: ~20-40 minutes
=============================================================================
"""

import ee
import os
import time
import requests
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

GEE_PROJECT = 'swansat'
OUTPUT_DIR = Path('data/satellite')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Guntur core agricultural zone
BBOX = [79.8, 16.0, 80.6, 16.5]  # [west, south, east, north]

# 20m resolution keeps files manageable while being sub-field resolution
SCALE = 20

MONTHS = [
    (2024, 6, "Jun"), (2024, 7, "Jul"), (2024, 8, "Aug"),
    (2024, 9, "Sep"), (2024, 10, "Oct"), (2024, 11, "Nov")
]

S2_BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
S1_BANDS = ['VV', 'VH']

# Number of longitude tiles (4 tiles keeps each under ~40MB)
N_TILES = 4

# ============================================================================
# INITIALIZE
# ============================================================================

print("Initializing Google Earth Engine...")
ee.Initialize(project=GEE_PROJECT)
roi = ee.Geometry.Rectangle(BBOX)
print(f"  ✓ Connected to project: {GEE_PROJECT}")
print(f"  ROI: {BBOX}")
print(f"  Scale: {SCALE}m")
print(f"  Tiles per image: {N_TILES}")

# ============================================================================
# COMPOSITE FUNCTIONS
# ============================================================================

def mask_s2_clouds(image):
    scl = image.select('SCL')
    mask = (scl.neq(3).And(scl.neq(8)).And(scl.neq(9))
            .And(scl.neq(10)).And(scl.neq(11)))
    return image.updateMask(mask)

def add_ndvi(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

def get_s2_composite(year, month):
    start = f'{year}-{month:02d}-01'
    end = f'{year}-{month+1:02d}-01' if month < 12 else f'{year+1}-01-01'
    col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate(start, end).filterBounds(roi)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
        .map(mask_s2_clouds).map(add_ndvi))
    count = col.size().getInfo()
    composite = col.select(S2_BANDS + ['NDVI']).median()
    return composite, count

def get_s1_composite(year, month):
    start = f'{year}-{month:02d}-01'
    end = f'{year}-{month+1:02d}-01' if month < 12 else f'{year+1}-01-01'
    col = (ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterDate(start, end).filterBounds(roi)
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .select(S1_BANDS))
    count = col.size().getInfo()
    composite = col.median()
    return composite, count

# ============================================================================
# TILED DOWNLOAD
# ============================================================================

def download_tiled(image, filename, n_bands):
    """Download image in N_TILES longitude strips, then merge."""
    final_path = OUTPUT_DIR / filename

    if final_path.exists() and final_path.stat().st_size > 50000:
        sz = final_path.stat().st_size / 1024 / 1024
        print(f"    ⏭  Already exists: {filename} ({sz:.1f}MB)")
        return True

    lon_step = (BBOX[2] - BBOX[0]) / N_TILES
    tile_paths = []

    for t in range(N_TILES):
        tile_west = BBOX[0] + t * lon_step
        tile_east = BBOX[0] + (t + 1) * lon_step
        tile_roi = ee.Geometry.Rectangle([tile_west, BBOX[1], tile_east, BBOX[3]])

        tile_name = f"{filename.replace('.tif', '')}_t{t}.tif"
        tile_path = OUTPUT_DIR / tile_name

        if tile_path.exists() and tile_path.stat().st_size > 5000:
            print(f"    ⏭  Tile {t} exists ({tile_path.stat().st_size/1024/1024:.1f}MB)")
            tile_paths.append(tile_path)
            continue

        try:
            url = image.getDownloadURL({
                'region': tile_roi,
                'scale': SCALE,
                'format': 'GEO_TIFF',
                'maxPixels': 1e9,
            })

            print(f"    ⬇  Tile {t+1}/{N_TILES} "
                  f"[{tile_west:.2f}° – {tile_east:.2f}°]...", end='', flush=True)

            resp = requests.get(url, timeout=600, stream=True)
            resp.raise_for_status()

            total = 0
            with open(tile_path, 'wb') as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
                    total += len(chunk)

            print(f" ✓ {total/1024/1024:.1f}MB")
            tile_paths.append(tile_path)
            time.sleep(1)

        except Exception as e:
            print(f" ✗ {e}")
            if tile_path.exists():
                tile_path.unlink()

    if not tile_paths:
        print(f"    ✗ No tiles downloaded for {filename}")
        return False

    # Merge tiles
    if len(tile_paths) == 1:
        # Only one tile, just rename
        tile_paths[0].rename(final_path)
        print(f"    ✓ {filename} ({final_path.stat().st_size/1024/1024:.1f}MB)")
        return True

    try:
        import rasterio
        from rasterio.merge import merge as rasterio_merge

        datasets = [rasterio.open(str(p)) for p in tile_paths]
        merged_array, merged_transform = rasterio_merge(datasets)

        meta = datasets[0].meta.copy()
        meta.update({
            'height': merged_array.shape[1],
            'width': merged_array.shape[2],
            'transform': merged_transform,
            'driver': 'GTiff',
            'compress': 'lzw',
        })

        with rasterio.open(str(final_path), 'w', **meta) as dst:
            dst.write(merged_array)

        for ds in datasets:
            ds.close()

        # Remove tiles
        for tp in tile_paths:
            tp.unlink()

        sz = final_path.stat().st_size / 1024 / 1024
        print(f"    ✓ Merged → {filename} "
              f"({merged_array.shape[2]}x{merged_array.shape[1]} px, "
              f"{merged_array.shape[0]} bands, {sz:.1f}MB)")
        return True

    except ImportError:
        print("    ⚠ rasterio not installed. Keeping tiles as separate files.")
        print("      Install with: pip install rasterio")
        # Keep tiles, Script 02 can handle multiple tiles
        return True

    except Exception as e:
        print(f"    ⚠ Merge error: {e}. Keeping tiles.")
        return True


# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"\n{'='*60}")
    print(f"  DOWNLOADING SATELLITE IMAGERY")
    print(f"  Guntur District | Kharif 2024 | {SCALE}m resolution")
    print(f"{'='*60}")

    results = {"s2": [], "s1": []}

    print("\n─── SENTINEL-2 (Optical: 6 bands + NDVI) ───")
    for year, month, mname in MONTHS:
        print(f"\n  {mname} {year}:")
        composite, count = get_s2_composite(year, month)
        print(f"    Source images: {count}")
        if count == 0:
            print(f"    ⚠ No images found. Skipping.")
            results["s2"].append(False)
            continue
        ok = download_tiled(composite, f'S2_{year}_{month:02d}.tif', 7)
        results["s2"].append(ok)

    print("\n─── SENTINEL-1 (SAR: VV + VH, cloud-free) ───")
    for year, month, mname in MONTHS:
        print(f"\n  {mname} {year}:")
        composite, count = get_s1_composite(year, month)
        print(f"    Source images: {count}")
        if count == 0:
            print(f"    ⚠ No images found. Skipping.")
            results["s1"].append(False)
            continue
        ok = download_tiled(composite, f'S1_{year}_{month:02d}.tif', 2)
        results["s1"].append(ok)

    # Summary
    s2_ok = sum(results['s2'])
    s1_ok = sum(results['s1'])

    print(f"\n{'='*60}")
    print(f"  DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"  Sentinel-2: {s2_ok}/6 months ✓")
    print(f"  Sentinel-1: {s1_ok}/6 months ✓")

    total_mb = 0
    print(f"\n  Files:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        sz = f.stat().st_size / 1024 / 1024
        total_mb += sz
        print(f"    {f.name:40s} {sz:.1f} MB")
    print(f"  Total: {total_mb:.0f} MB")

    if s2_ok + s1_ok >= 8:
        print(f"\n  ✓ Ready. Next: python scripts/02_preprocess_patches.py")
    elif s2_ok + s1_ok > 0:
        print(f"\n  ⚠ Partial download. Re-run to retry failed months.")
    else:
        print(f"\n  ✗ No data downloaded. Check GEE authentication.")


if __name__ == '__main__':
    main()
