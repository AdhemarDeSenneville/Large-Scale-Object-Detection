#!/usr/bin/env python3
"""
Fetch the Copernicus VHR 2021 mosaic for the Paris area via WMS
"""

import os
from owslib.wms import WebMapService
from pyproj import Transformer

# 1. WMS endpoint (True Colour 2 m 2021 mosaic, EPSG:3035)
WMS_URL = "https://copernicus.discomap.eea.europa.eu/arcgis/services/" \
          "GioLand/VHR_2021_LAEA/ImageServer/WMSServer"

# 2. Connect
wms = WebMapService(WMS_URL, version="1.3.0")

# 3. Inspect available layers (for demo, we’ll grab the first)
layer_name = list(wms.contents)[0]
print("Using layer:", layer_name)

# 4. Define Paris bbox in lon/lat (EPSG:4326)
#    SW corner: approx (2.2241° E, 48.8156° N)
#    NE corner: approx (2.4699° E, 48.9022° N)
min_lon, min_lat = 2.2241, 48.8156
max_lon, max_lat = 2.4699, 48.9022

# 5. Reproject to EPSG:3035
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
min_x, min_y = transformer.transform(min_lon, min_lat)
max_x, max_y = transformer.transform(max_lon, max_lat)

# 6. Fetch the map as GeoTIFF (size 1024×1024)
img = wms.getmap(
    layers=[layer_name],
    styles=[''],
    srs='EPSG:3035',
    bbox=(min_x, min_y, max_x, max_y),
    size=(1024, 1024),
    format='image/geotiff'
)

# 7. Save to disk
out_path = "paris_vhr_2021.tif"
with open(out_path, "wb") as f:
    f.write(img.read())

print(f"Saved Paris tile to {out_path}")
