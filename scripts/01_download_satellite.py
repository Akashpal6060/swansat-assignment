import ee
import os
import time
import requests
import numpy as np
from pathlib import Path
import geemap  

GEE_PROJECT = 'swansat'
OUTPUT_DIR = Path('data/satellite')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GUNTUR_BBOX = [79.8, 16.0, 80.6, 16.5]
MONTHS = [(2024, 6, "Jun"), (2024, 7, "Jul"), (2024, 8, "Aug"), (2024, 9, "Sep"), (2024, 10, "Oct"), (2024, 11, "Nov")]
S2_BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
S2_SCALE = 10
S1_BANDS = ['VV', 'VH']
S1_SCALE = 10

print("Initializing Google Earth Engine...")
ee.Initialize(project=GEE_PROJECT)
roi = ee.Geometry.Rectangle(GUNTUR_BBOX)

def mask_s2_clouds(image):
    scl = image.select('SCL')
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    return image.updateMask(mask)

def add_ndvi(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

def get_s2_monthly_composite(year, month):
    start_date = f'{year}-{month:02d}-01'
    end_date = f'{year+1}-01-01' if month == 12 else f'{year}-{month+1:02d}-01'
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate(start_date, end_date)
        .filterBounds(roi)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
        .map(mask_s2_clouds).map(add_ndvi))
    count = collection.size().getInfo()
    composite = collection.select(S2_BANDS + ['NDVI']).median()
    composite_scaled = composite.select(S2_BANDS).toInt16()
    ndvi_scaled = composite.select('NDVI').multiply(10000).toInt16()
    return composite_scaled.addBands(ndvi_scaled), count

def get_s1_monthly_composite(year, month):
    start_date = f'{year}-{month:02d}-01'
    end_date = f'{year+1}-01-01' if month == 12 else f'{year}-{month+1:02d}-01'
    collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterDate(start_date, end_date)
        .filterBounds(roi)
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .select(S1_BANDS))
    count = collection.size().getInfo()
    composite = collection.median()
    return composite.multiply(100).toInt16(), count

def download_image(image, filename, scale, bands_info=""):
    filepath = OUTPUT_DIR / filename
    if filepath.exists() and filepath.stat().st_size > 10000:
        print(f"    ⏭  Already exists: {filename}")
        return True
    try:
        print(f"    ⬇  Downloading {filename} {bands_info}...")
        # Notice scale_mul is COMPLETELY GONE here
        geemap.download_ee_image(
            image=image, 
            filename=str(filepath), 
            scale=scale, 
            region=roi, 
            crs='EPSG:4326'
        )
        if filepath.exists():
            print(f"    ✓  Saved: {filename}")
            return True
        return False
    except Exception as e:
        print(f"    ✗  Failed: {filename} - {e}")
        if filepath.exists(): filepath.unlink()
        return False

def main():
    print("\nDOWNLOADING SATELLITE IMAGERY...")
    results = {"s2": [], "s1": []}
    for year, month, mname in MONTHS:
        comp, count = get_s2_monthly_composite(year, month)
        if count > 0: results["s2"].append(download_image(comp, f'S2_{year}_{month:02d}.tif', S2_SCALE))
    for year, month, mname in MONTHS:
        comp, count = get_s1_monthly_composite(year, month)
        if count > 0: results["s1"].append(download_image(comp, f'S1_{year}_{month:02d}.tif', S1_SCALE))
    print(f"\nDone! S2: {sum(results['s2'])}/6 | S1: {sum(results['s1'])}/6")

if __name__ == '__main__':
    main()
