
# =============================================================================
# Google Earth Engine Script - Sentinel-2 NDVI for Guntur District
# =============================================================================
# To use this:
# 1. Sign up at https://earthengine.google.com/signup/
# 2. pip install earthengine-api geemap
# 3. Run: earthengine authenticate
# 4. Then run this script
# =============================================================================

import ee
import geemap
import pandas as pd

# Initialize Earth Engine
ee.Initialize()

# Guntur District boundary (approximate)
guntur_bbox = ee.Geometry.Rectangle([79.15, 15.75, 80.85, 16.85])

# Cloud masking function for Sentinel-2
def mask_s2_clouds(image):
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
        qa.bitwiseAnd(cirrus_bit_mask).eq(0)
    )
    return image.updateMask(mask).divide(10000)

# Sentinel-2 Surface Reflectance
s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterDate('2024-06-01', '2024-11-30')
    .filterBounds(guntur_bbox)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
    .map(mask_s2_clouds))

# Add NDVI band
def add_ndvi(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

s2_ndvi = s2.map(add_ndvi)

# Monthly composites
months = ee.List.sequence(6, 11)

def make_monthly_composite(month):
    month = ee.Number(month)
    start = ee.Date.fromYMD(2024, month, 1)
    end = start.advance(1, 'month')
    monthly = s2_ndvi.filterDate(start, end).select('NDVI').median()
    return monthly.set('month', month)

monthly_ndvi = ee.ImageCollection.fromImages(
    months.map(make_monthly_composite)
)

# Export to Drive
for month in range(6, 12):
    img = monthly_ndvi.filter(ee.Filter.eq('month', month)).first()
    task = ee.batch.Export.image.toDrive(
        image=img.clip(guntur_bbox),
        description=f'NDVI_Guntur_2024_{month:02d}',
        folder='SwanSAT_Data',
        scale=10,
        region=guntur_bbox,
        maxPixels=1e13,
        fileFormat='GeoTIFF'
    )
    task.start()
    print(f"Export started for month {month}")

# Also export Sentinel-1 SAR data (cloud-free alternative)
s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterDate('2024-06-01', '2024-11-30')
    .filterBounds(guntur_bbox)
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .select(['VV', 'VH']))

# Monthly SAR composites
for month in range(6, 12):
    start = f'2024-{month:02d}-01'
    end = f'2024-{month+1:02d}-01' if month < 12 else '2025-01-01'
    monthly_s1 = s1.filterDate(start, end).mean()
    
    task = ee.batch.Export.image.toDrive(
        image=monthly_s1.clip(guntur_bbox),
        description=f'SAR_Guntur_2024_{month:02d}',
        folder='SwanSAT_Data',
        scale=10,
        region=guntur_bbox,
        maxPixels=1e13,
        fileFormat='GeoTIFF'
    )
    task.start()
    print(f"SAR export started for month {month}")

print("All exports started. Check Google Drive > SwanSAT_Data folder.")
