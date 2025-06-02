"""

# 2016–2020
wget -O ortho_2016_2020.json \
  "https://data.geopf.fr/wfs/ows?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature\
&TYPENAMES=ORTHOIMAGERY.ORTHOPHOTOS.GRAPHE.2016-2020:graphe_bdortho\
&SRSNAME=EPSG:2154&OUTPUTFORMAT=application/json"

# 2021–now
wget -O ortho_2021_now.json \
  "https://data.geopf.fr/wfs/ows?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature\
&TYPENAMES=ORTHOIMAGERY.ORTHOPHOTOS.GRAPHE-MOSAIQUAGE:graphe_bdortho\
&SRSNAME=EPSG:2154&OUTPUTFORMAT=application/json"

ogr2ogr -f GPKG ortho_2016_2020.gpkg ortho_2016_2020.json
ogr2ogr -f GPKG ortho_2021_now.gpkg ortho_2021_now.json


# warning no temporal changes on that api layer, only few are not white images:
    'ORTHOIMAGERY.ORTHOPHOTOS.ORTHO-EXPRESS.2017', 
    'ORTHOIMAGERY.ORTHOPHOTOS.ORTHO-EXPRESS.2018', 
    'ORTHOIMAGERY.ORTHOPHOTOS.ORTHO-EXPRESS.2019', 
    'ORTHOIMAGERY.ORTHOPHOTOS.ORTHO-EXPRESS.2021', 
    'ORTHOIMAGERY.ORTHOPHOTOS.ORTHO-EXPRESS.2023', 
    'ORTHOIMAGERY.ORTHOPHOTOS.ORTHO-EXPRESS.2024', 

For spot most of the images are available:
    'ORTHOIMAGERY.ORTHO-SAT.SPOT.2013', 'ORTHOIMAGERY.ORTHO-SAT.SPOT.2014', 'ORTHOIMAGERY.ORTHO-SAT.SPOT.2015', 
    'ORTHOIMAGERY.ORTHO-SAT.SPOT.2016', 'ORTHOIMAGERY.ORTHO-SAT.SPOT.2017', 'ORTHOIMAGERY.ORTHO-SAT.SPOT.2018', 
    'ORTHOIMAGERY.ORTHO-SAT.SPOT.2019', 'ORTHOIMAGERY.ORTHO-SAT.SPOT.2020', 'ORTHOIMAGERY.ORTHO-SAT.SPOT.2021', 
    'ORTHOIMAGERY.ORTHO-SAT.SPOT.2022', 'ORTHOIMAGERY.ORTHO-SAT.SPOT.2023', 'ORTHOIMAGERY.ORTHO-SAT.SPOT.2024', 
"""
import os
from io import BytesIO
import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
import fiona

# Directory where cached features will be stored


# Mapping of human-friendly labels to WFS type names


# Base WFS endpoint
WFS_URL = (
    "https://data.geopf.fr/wfs/ows?"
    "SERVICE=WFS"
    "&VERSION=2.0.0"
    "&REQUEST=GetFeature"
    "&SRSNAME=EPSG:2154"
    "&OUTPUTFORMAT=application/json"
)

def download_full_france(use_cache=True):
    layers = {
        "2000-2005": "ORTHOIMAGERY.ORTHOPHOTOS.GRAPHE.2000-2005:graphe_bdortho",
        "2006-2010": "ORTHOIMAGERY.ORTHOPHOTOS.GRAPHE.2006-2010:graphe_bdortho",
        "2011-2015": "ORTHOIMAGERY.ORTHOPHOTOS.GRAPHE.2011-2015:graphe_bdortho",
        "2016-2020": "ORTHOIMAGERY.ORTHOPHOTOS.GRAPHE.2016-2020:graphe_bdortho",
        "2021-now": "ORTHOIMAGERY.ORTHOPHOTOS.GRAPHE-MOSAIQUAGE:graphe_bdortho"
    }
    path_to_save = '/home/adhemar/Bureau/datasets/France/france_ign_meta/ortho_features'
    """
    Download the full-feature datasets for all year ranges for France
    and cache them locally.

    Parameters:
    - use_cache: bool, if True skips download when cache exists
    """
    for label, typename in layers.items():
        # Define cache file path
        filename = f"{label.replace(' ', '_').replace('-', '_')}.gpkg"
        cache_file = os.path.join(path_to_save, filename)

        if use_cache and os.path.exists(cache_file):
            print(f"Cache exists for {label}, skipping download.")
            continue

        # No bbox filter for full download
        url = f"{WFS_URL}&TYPENAMES={typename}"
        print(f"Downloading full dataset for {label}...")
        print(url)
        resp = requests.get(url)
        resp.raise_for_status()
        gdf = gpd.read_file(BytesIO(resp.content))

        print(f"Saving full dataset for {label} to {cache_file}...")
        gdf.to_file(cache_file, driver="GPKG", layer=label)

    print("All layers downloaded and cached.")


def get_available_years(bbox, return_all = False):
    layers = {
        "2000-2005": "ORTHOIMAGERY.ORTHOPHOTOS.GRAPHE.2000-2005:graphe_bdortho",
        "2006-2010": "ORTHOIMAGERY.ORTHOPHOTOS.GRAPHE.2006-2010:graphe_bdortho",
        "2011-2015": "ORTHOIMAGERY.ORTHOPHOTOS.GRAPHE.2011-2015:graphe_bdortho",
        "2016-2020": "ORTHOIMAGERY.ORTHOPHOTOS.GRAPHE.2016-2020:graphe_bdortho",
        "2021-now": "ORTHOIMAGERY.ORTHOPHOTOS.GRAPHE-MOSAIQUAGE:graphe_bdortho"
    }
    path_to_save = '/home/adhemar/Bureau/datasets/France/france_ign_meta/ortho_features'
    """
    Load cached full-France datasets and filter to the given bounding box.

    Parameters:
    - bbox: tuple (xmin, ymin, xmax, ymax) in EPSG:2154 coordinates

    Returns:
    - GeoDataFrame with features across all year ranges and a 'year_range' column
    """
    #bbox = *bbox

    gdfs = []
    for label, layer_identifier in layers.items():
        # Define cache file path
        filename = f"{label.replace(' ', '_').replace('-', '_')}.gpkg"
        cache_file = os.path.join(path_to_save, filename)

        #print(cache_file)
        layers_in_file = fiona.listlayers(cache_file)
        gdf = gpd.read_file(cache_file, layer=layers_in_file[0], bbox=tuple(bbox))
        
        #gdf = gpd.read_file(cache_file, layer=label, bbox=tuple(bbox) )
        if gdf.empty:
            continue
        gdf["year_range"] = label
        gdf["layer_identifier"] = layer_identifier
        gdfs.append(gdf)

    if gdfs:
        all_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs="EPSG:2154")

        if return_all:
            return all_gdf
        else:
            keep_columns = ['pva', 'res', 'echelle', 'date_vol', 'year_range']
            return all_gdf[keep_columns]
    else:
        return None

if False: 
    #!curl "https://data.geopf.fr/wfs/ows?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetCapabilities" -o capabilities.xml # type: ignore

    import requests
    import xml.etree.ElementTree as ET

    # 1) Fetch Capabilities
    resp = requests.get(
        "https://data.geopf.fr/wfs/ows",
        params={
            "SERVICE": "WFS",
            "VERSION": "2.0.0",
            "REQUEST": "GetCapabilities"
        }
    )
    resp.raise_for_status()

    # 2) Parse and extract typeNames
    ns = {
        "wfs": "http://www.opengis.net/wfs/2.0",
        "ows": "http://www.opengis.net/ows/1.1"
    }
    root = ET.fromstring(resp.content)

    type_names = [
        ft.find("wfs:Name", ns).text
        for ft in root.findall(".//wfs:FeatureType", ns)
    ]
    print("Available typeNames:")
    for name in type_names:
        if 'ORTHO' in name:

            print(" -", name)
