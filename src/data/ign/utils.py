import math
import requests

def is_bbox_within_tile(bbox, tile):
    tile_x, tile_y, tile_width, tile_height = tile
    bbox_x, bbox_y, bbox_width, bbox_height = bbox

    return (tile_x <= bbox_x < tile_x + tile_width and
            tile_y <= bbox_y < tile_y + tile_height and
            tile_x <= bbox_x + bbox_width <= tile_x + tile_width and
            tile_y <= bbox_y + bbox_height <= tile_y + tile_height)


def get_square_coordinates(lat, lon, size_m=700):
    R = 6378137 # Earth's radius in meters

    delta_lat = (size_m / 2) / R * (180 / math.pi)
    delta_lon = (size_m / 2) / (R * math.cos(math.radians(lat))) * (180 / math.pi)

    # Corner coordinates
    top_left = (lat + delta_lat, lon - delta_lon)
    top_right = (lat + delta_lat, lon + delta_lon)
    bottom_left = (lat - delta_lat, lon - delta_lon)
    bottom_right = (lat - delta_lat, lon + delta_lon)

    return {
        "top_left": top_left,
        "top_right": top_right,
        "bottom_left": bottom_left,
        "bottom_right": bottom_right
    }

def download_wms_image(lat, lon, size_m, cfg):
    """
    Downloads an image from a WMS (Web Map Service) server based on the given latitude, 
    longitude, and image size in meters.

    Parameters:
        lat (float): Latitude of the center point.
        lon (float): Longitude of the center point.
        size_m (float): Size of the requested image in meters.

    Returns:
        - "data" : The image data.
        - "meta_data" (dict): The coordinates of the requested bounding box.
    """

    coords = get_square_coordinates(lat, lon, size_m)
    bbox = f"{coords['bottom_left'][0]},{coords['bottom_left'][1]},{coords['top_right'][0]},{coords['top_right'][1]}"

    layer = cfg.get('LAYER', 'ORTHOIMAGERY.ORTHOPHOTOS.BDORTHO')
    style = cfg.get('STYLE', 'normal')
    format = cfg.get('FORMAT', 'image/png')
    crs = cfg.get('CRS', 'EPSG:4326')
    width = cfg.get('WIDTH', 256)
    height = cfg.get('HEIGHT', 256)

    url = (
        f"https://data.geopf.fr/wms-r?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
        f"&LAYERS={layer}&STYLES={style}&FORMAT={format}&CRS={crs}"
        f"&BBOX={bbox}&WIDTH={width}&HEIGHT={height}"
    )

    response = requests.get(url)
    
    return {
        "data": response.content,
        "meta_data": coords,
    }