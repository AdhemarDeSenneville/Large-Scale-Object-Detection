import requests
from tqdm import tqdm
from shapely.geometry import box
from shapely import Polygon
from pystac_client import Client
import numpy as np
import planetary_computer
import io
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds
from dateutil.parser import isoparse
from rio_tiler.io import Reader
from .utils.geometry import reproject_bounds

def download(url, verbose=False):
    """
    Downloads data from a given URL and returns it as a NumPy array.
    Args:
        url (str): The URL to download the data from.
        verbose (bool, optional): If True, displays a progress bar during the download. Defaults to False.
    Returns:
        list: A list containing the downloaded images as a list of NumPy arrays.
        list: A list of MemoryFile objects containing the downloaded images.
    Raises:
        Exception: If the date of the image cannot be parsed to be included in the image suffix tag.
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024 * 1024  # 1 MB
    buffer = io.BytesIO()

    if verbose:
        t = tqdm(total=total_size, unit='B', unit_scale=True)
        for data in response.iter_content(block_size):
            buffer.write(data)
            t.update(len(data))
        t.close()
    else:
        for data in response.iter_content(block_size):
            buffer.write(data)

    buffer.seek(0)
    
    with rasterio.open(buffer) as src:

        band = src.read(1)
        meta = src.meta.copy()
        transform = src.transform
        crs = src.crs

    return band, meta, transform, crs

def download_bbox(url, bounds, max_size=512):
    """
    Downloads only the spatial bounding box from a COG using HTTP Range requests.

    Args:
        url (str): URL to the COG file (e.g., Sentinel-2 band URL).
        bounds (tuple): (minx, miny, maxx, maxy) in image CRS (usually EPSG:4326 for STAC assets).
        max_size (int): Maximum pixel size for output image (preserves aspect ratio).

    Returns:
        tuple: (band_data, meta, transform, crs)
    """
    with Reader(url) as cog:
        # WGS84 bounds (required by Planetary Computer for some reasons)
        img = cog.part(bounds, max_size=max_size, dst_crs=cog.crs, bounds_crs="EPSG:4326")
        band = img.data[0]

        meta = cog.dataset.meta.copy()
        meta.update({
            "height": band.shape[0],
            "width": band.shape[1],
            "transform": img.transform,
            "count": 1
        })
        return band, meta, meta["transform"], cog.crs


def get_sentinel2_image(cloud_cover=10, date_range=("2024-01-01", "2024-03-01"), bbox_delta=0.009, verbose=False, api='microsoft', bbox=None):
    
    if api == 'microsoft':
        API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
        client = Client.open(API_URL, modifier=planetary_computer.sign_inplace)
    elif api == 'element84':
        API_URL = "https://earth-search.aws.element84.com/v1"
        client = Client.open(API_URL)
    collection = "sentinel-2-l2a"
    
    
    if not isinstance(bbox, Polygon):
        bbox = box(*bbox)
    
    # Search for images containing the bouding box using the STAC API
    search = client.search(
        collections=[collection],
        intersects=bbox,
        query={"eo:cloud_cover": {"lt": cloud_cover}},
        datetime=f"{date_range[0]}/{date_range[1]}"
    )
    

    items = search.item_collection()
    if not items or len(items) == 0:
        print("No suitable images found.")
        return None, None
    
    if verbose:
        print(f"Found {len(items)} images matching the criteria.")
        for item in items:
            print(f"Image ID: {item.id}, Cloud Cover: {item.properties['eo:cloud_cover']}%, Date: {item.properties['datetime']}")

    items = sorted(items, key=lambda it: it.properties['eo:cloud_cover'])
    items = [items[0]]

    memfile_list = []
    rgb_list = []

    for item in items:
        assets = item.assets
        
        # Get RGB bands download links (blue, green, red)
        if api == 'microsoft': # Microsoft api is usually faster than element84
            blue_href = assets["B02"].href
            green_href = assets["B03"].href
            red_href = assets["B04"].href
        elif api == 'element84':
            blue_href = assets["blue"].href
            green_href = assets["green"].href
            red_href = assets["red"].href


        if bbox2 is None:
            # Estimate file sizes before downloading
            if verbose:
                    estimated_sizes = {}
                    for color, href in zip(["blue", "green", "red"], [blue_href, green_href, red_href]):
                        response = requests.head(href)
                        size = int(response.headers.get("Content-Length", 0)) / (1024 * 1024)
                        estimated_sizes[color] = size
                    
                    total_size = sum(estimated_sizes.values())
                    print(f"Estimated download size: {total_size:.2f} MB (Blue: {estimated_sizes['blue']:.2f} MB, Green: {estimated_sizes['green']:.2f} MB, Red: {estimated_sizes['red']:.2f} MB)")
            b, meta_b, transform, crs = download(blue_href, verbose=verbose)
            g, _, _, _ = download(green_href, verbose=verbose)
            r, _, _, _ = download(red_href, verbose=verbose)
        else:
            b, meta_b, transform, crs = download_bbox(blue_href, bbox.bounds, max_size=1024)
            g, _, _, _ = download_bbox(green_href, bbox.bounds, max_size=1024)
            r, _, _, _ = download_bbox(red_href, bbox.bounds, max_size=1024)

        
        rgb = np.stack([r, g, b], axis=0)

        meta_b.update({
            "count": 3,
            "dtype": rgb.dtype,
            "driver": "GTiff",
            "transform": transform,
            "crs": crs
        })
        memfile = MemoryFile()
        with memfile.open(**meta_b) as dst:
            dst.write(rgb[0], 1)
            dst.write(rgb[1], 2)
            dst.write(rgb[2], 3)
            try:
                date = isoparse(item.properties["datetime"])
                dst.update_tags(
                    Title="Sentinel-2 RGB Composite",
                    CloudCover=item.properties["eo:cloud_cover"],
                    Date=item.properties["datetime"],
                    Suffix=f'_{date.year}_{date.month:02d}_{date.day:02d}',
                    Platform=item.properties.get("platform", "Sentinel-2")
                )
            except Exception as e:
                print(f"Error parsing date for item {item.id}: {e}")
                dst.update_tags(
                    Title="Sentinel-2 RGB Composite",
                    CloudCover=item.properties["eo:cloud_cover"],
                    Date=item.properties["datetime"],
                    Platform=item.properties.get("platform", "Sentinel-2")
                )
        memfile_list.append(memfile)
        rgb_list.append(rgb)

    return rgb_list, memfile_list

def save_image(memfile, path):
    """
    Save the image from the MemoryFile to a specified path.
    Args:
        memfile (MemoryFile): The MemoryFile containing the image data.
        path (str): The path where the image will be saved.
    """
    with memfile.open() as src:
        if src.tags().get('Suffix'):
            with rasterio.open(path[:-4] + src.tags().get('Suffix') + '.tif', 'w', **src.meta) as dst:
                
                dst.write(src.read())
                dst.update_tags(**src.tags())
        else:
            with rasterio.open(path, 'w', **src.meta) as dst:
                dst.write(src.read())
                dst.update_tags(**src.tags())




if __name__ == "__main__":
    latitude, longitude = 51.482694952724614, 46.20856383098548
    cloud_cover = 30
    date_range = ("2024-01-01", "2024-01-13")
    delta_x = 0.03
    delta_y = 0.03
    bbox_delta = [delta_x, delta_y]

    bbox = (longitude - bbox_delta[0], latitude - bbox_delta[1], longitude + bbox_delta[0], latitude + bbox_delta[1])
    rgb, memfile = get_sentinel2_image(latitude, longitude, cloud_cover, date_range, verbose=True, bbox=bbox)

    