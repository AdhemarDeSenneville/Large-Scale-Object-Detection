import math
import requests
import numpy as np

import io
from PIL import Image, UnidentifiedImageError

from ..utils_api import get_bbox_coordinates

class IGNDataset:
    def __init__(self, cfg = {
            'resolution': 0.5, # meters per pixel
            'pixel_width': 1000, # pixels
            'pixel_height': 1000, # pixels
            'layer': 'ORTHOIMAGERY.ORTHOPHOTOS.BDORTHO', # see: https://geoservices.ign.fr/services-web-experts-ortho
            'style': 'normal',
            'format': 'image/png',
            'crs': 'EPSG:2154',
        }
    ):
        self.cfg = cfg
    
    def geo_index_gps_positions(self, detections, format = 'old'):
        print("[INFO] geo_index_gps_positions useless for IGN API dataset")
        return detections

    def get_rgb_image(self, x, y, window_size = None):

        cfg = self.cfg.copy()
        

        if window_size is not None:
            cfg['pixel_width'] = window_size
            cfg['pixel_height'] = window_size
        
        image =  np.array(download_image(x, y, cfg)['data'])
    
        print(image.shape)

        return image


def download_image(x = None, y = None, cfg = {}):

    # Using API: https://geoservices.ign.fr/documentation/services/services-geoplateforme/telechargement

    layer = cfg.get('layer', 'ORTHOIMAGERY.ORTHOPHOTOS.BDORTHO')
    style = cfg.get('style', 'normal')
    format = cfg.get('format', 'image/png')
    crs = cfg.get('crs', 'EPSG:4326')

    resolution = cfg.get('resolution', 0.2)

    bbox_epsg = cfg.get('bbox_epsg', None) # xywh
    if bbox_epsg:
        pixel_width = int(bbox_epsg[2]/resolution) 
        pixel_height = int(bbox_epsg[3]/resolution)
        
        #bbox = f"{coords['bottom_left'][0]},{coords['bottom_left'][1]},{coords['top_right'][0]},{coords['top_right'][1]}"
        coords = {
            "top_left": (bbox_epsg[0], bbox_epsg[1]),
            "top_right": (bbox_epsg[0] + bbox_epsg[2], bbox_epsg[1]),
            "bottom_left": (bbox_epsg[0], bbox_epsg[1] + bbox_epsg[3]),
            "bottom_right": (bbox_epsg[0] + bbox_epsg[2], bbox_epsg[1] + bbox_epsg[3])
        }
        bbox = f"{bbox_epsg[0]},{bbox_epsg[1]},{bbox_epsg[0]+bbox_epsg[2]},{bbox_epsg[1]+bbox_epsg[3]}"
    else:
        pixel_width = cfg.get('pixel_width', 256)
        pixel_height = cfg.get('pixel_height', pixel_width)

        delta_x_metre = resolution * pixel_width
        delta_y_metre = resolution * pixel_height

        print('HERE', delta_x_metre, delta_y_metre)

        if crs not in ['EPSG:4326', 'EPSG:2154']:
            raise ValueError(f"CRS {crs} not supported")

        #print(x, )
        coords = get_bbox_coordinates(x, y, delta_x_metre, delta_y_metre, crs)


        bbox = f"{coords['bottom_left'][0]},{coords['bottom_left'][1]},{coords['top_right'][0]},{coords['top_right'][1]}"


        print('HERE',crs, bbox)

    url = (
        f"https://data.geopf.fr/wms-r?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
        f"&LAYERS={layer}&STYLES={style}&FORMAT={format}&CRS={crs}"
        f"&BBOX={bbox}&WIDTH={pixel_width}&HEIGHT={pixel_height}"
    )

    response = requests.get(url)
    try:
        print(url)
        image = Image.open(io.BytesIO(response.content))
        return {
            "data": image,
            "meta_data": coords,
        }
    except UnidentifiedImageError:
        print(f"[ERROR] Failed to load image from WMS. Status: {response.status_code}")
        print(f"URL: {url}")
        return None


if __name__ == "__main__":

    #center_x = 48.5687097375828
    #center_y = 4.36987399941624

    center_x = 801_063 
    center_y = 6_830_709

    cfg = {
        'resolution': 0.5, # meters per pixel
        'pixel_width': 256, # pixels
        'pixel_height': 256, # pixels
        'layer': 'ORTHOIMAGERY.ORTHOPHOTOS.BDORTHO', # see: https://geoservices.ign.fr/services-web-experts-ortho
        'style': 'normal',
        'format': 'image/png',
        #'crs': 'EPSG:4326', 
        'crs': 'EPSG:2154',
    }

    response = download_image(x = center_x, y = center_y, cfg = cfg)

    image = response['data']
    bbox = response['meta_data']

    print('test', bbox)

    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.show()

    from PIL.ExifTags import TAGS
    # Extract metadata (EXIF)
    exif_data = image._getexif()

    # Print metadata if available
    if exif_data:
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            print(f"{tag:25}: {value}")
    else:
        print("No EXIF metadata found.")

