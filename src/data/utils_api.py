import io
import math
import requests
import time

from typing import Optional, Dict, Tuple
from owslib.wms import WebMapService
from PIL import Image
from pyproj import Transformer, CRS


def check_lon_lat(lon, lat):
    in_eu  = (-25 <= lon <= 45) and (34 <= lat <= 72)
    # if you swap, does it land in Europe?
    swapped = (-25 <= lat <= 45) and (34 <= lon <= 72)
    if swapped:
        print('WARNING : have you swapped')
    return in_eu, swapped


class OrthoCropApi:
    def __init__(
            self,
            num_retries: int = 10,
            delay_between_retries: float = 2,
            delay: float = 0.0,
        ):
        
        self.delay = delay
        self.delay_between_retries = delay_between_retries
        self.num_retries = num_retries

        self.prev_time = 0.0
    
    def get(self, *args, **kwargs):
        for attempt in range(self.num_retries):
            try:
                self.delay_sleep()
                return self.get_once(*args, **kwargs)
            except Exception as e:
                print(f"[ERROR] Attempt {attempt + 1}/{self.num_retries} failed: {e}")
                time.sleep(self.delay_between_retries)
        raise RuntimeError("Failed to fetch map image after multiple attempts.")

    def delay_sleep(self):
        
        if self.delay > 0:
            current_time = time.time()
            if current_time - self.prev_time < self.delay:
                time.sleep(self.delay - (current_time - self.prev_time))
            self.prev_time = current_time

class OrthoCropWebApi(OrthoCropApi):
    """
    A simple client to request orthophoto crops from a WMS server.
    """
    def __init__(
        self,
        wms_url: str,
        epsg_query: str,
        wms_version: str = '1.3.0',
        layer: str = None,
        style: str = '',
        img_format: str = 'image/geotiff',
        verbose = False,
        token: str = None,
        **kwargs
    ):
        """
        Initialize the WMS client with explicit parameters.

        Parameters
        ----------
        wms_url : str
            URL to the WMS endpoint
        layer : str, optional
            Name of the layer to request. If omitted, defaults to first available.
        style : str, optional
            Style to use for the layer (default: '').
        img_format : str, optional
            Image MIME type to request (default: 'image/geotiff').
        """
        # Store parameters
        self.wms_url = wms_url
        self.wms_version = wms_version
        self.epsg_query = epsg_query

        self.layer = layer
        self.style = style
        self.img_format = img_format
        self.token = token

        # Initialize WMS connection
        self.wms = WebMapService(self.wms_url, version=wms_version)

        self.wms_args = {
            'service': 'WMS',
            'version': '1.3.0',
            'request': 'GetMap',
        }
        self.headers = {"token": self.token} if self.token else {}

        # Choose default layer if none provided
        if self.layer is None:
            available_layers = list(self.wms.contents)
            self.layer = available_layers[0]
            print(f"[INFO] Using layer: {self.layer} out of {available_layers}")
        
        if verbose:
            print('Layers :', list(self.wms.contents))
            print('Crs Options :',self.wms.contents[self.layer].crsOptions)
        
        super().__init__(**kwargs)

    def get_once(
        self,
        bbox: tuple,
        size: tuple,
        layer: str = None,
    ):
        """
        Fetch a map image for a given bounding box and output size.

        Parameters
        ----------
        bbox : tuple
            Bounding box in the form (minx, miny, maxx, maxy) in the target CRS.
        size : tuple
            Output image dimensions as (width, height) in pixels.
        epsg : str
            CRS code, e.g. 'EPSG:3035' or 'EPSG:2154'.
        """

        # Build WMS GetMap parameters
        layer = layer or self.layer

        if self.wms_version == '1.3.0':
            request_args = {
                'layers': [layer],
                'styles': [layer],
                'srs': self.epsg_query,
                'bbox': bbox,
                'size': size,
                'format': self.img_format,
                'transparent': False
            }
        elif self.wms_version == '1.3.0':
            request_args = {
                'layers': layer,
                'styles': layer,
                'crs': self.epsg_query,
                'bbox': bbox,
                'size': size,
                'format': self.img_format,
                'transparent': False
            }
        else:
            raise NotImplementedError
        
        if 'tiff' in self.img_format.lower():

            response = self.wms.getmap(**request_args)
            data = response.read()
        else:
            #print(request_args)
            #response = requests.get(self.wms_url, params = {**self.wms_args, **request_args}, headers=self.headers)
            #response.raise_for_status()
            #data = response.content
            bbox_n = []
            bbox_n.append(min(bbox[0], bbox[2]))
            bbox_n.append(min(bbox[1], bbox[3]))
            bbox_n.append(max(bbox[0], bbox[2]))
            bbox_n.append(max(bbox[1], bbox[3]))

            url = (
                f"{self.wms_url}?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
                f"&LAYERS={layer}&STYLES={self.style}&FORMAT={self.img_format}&CRS={self.epsg_query}"
                f"&BBOX={bbox_n[0]},{bbox_n[1]},{bbox_n[2]},{bbox_n[3]}&WIDTH={size[0]}&HEIGHT={size[1]}"
            )
            #print(url)
            response = requests.get(url, headers=self.headers)
            data = response.content

        if 'tiff' in self.img_format.lower():
            from rasterio.io import MemoryFile
            with MemoryFile(data) as memfile:
                ds = memfile.open()
                img = ds.read()
                meta = ds.profile.copy()
            return {'image': img, 'meta_data': meta}
        else:
            #print(data)
            return {
                'image': Image.open(io.BytesIO(data)),
                'meta_data': {'size': size, 'bbox': bbox, 'epsg': self.epsg_query}
            }
    
    def post_process_tiff_image():
        pass
            
            


        #except Exception as e:
        #    logging.error(f"[ERROR] WMS GetMap failed: {e}")
        #    return None


def get_bbox_coordinates(
    x: float,
    y: float,
    delta_x_metre: float,
    delta_y_metre: float,
    crs: str
) -> Dict[str, Tuple[float, float]]:
    """
    Calculate the coordinates of the bounding box corners centered at (x, y),
    with specified width and height in meters, for a given coordinate reference system (CRS).

    Parameters:
    ----------
    x : float
        X coordinate (longitude in degrees for EPSG:4326, or easting in meters for projected CRSs).
    y : float
        Y coordinate (latitude in degrees for EPSG:4326, or northing in meters for projected CRSs).
    delta_x_metre : float
        Width of the bounding box in meters.
    delta_y_metre : float
        Height of the bounding box in meters.
    crs : str
        Coordinate Reference System (e.g., 'EPSG:4326', 'EPSG:2154', 'EPSG:3035').
    """


    if crs in ["EPSG:4326"]:
        units = 'angles'
    elif crs in ["EPSG:2154", "EPSG:3035", "EPSG:28992", 'EPSG:25832', 'EPSG:25829']:
        units = 'metre'
    else:
        crs = CRS.from_user_input(crs)
        unit = crs.axis_info[0].unit_name.lower()
        if 'degree' in unit:
            units =  'angles'
        elif 'metre' in unit:
            units =  'metre'
        else:
            units =  ''


    if units == 'angles':
        # From meters to degrees
        earth_radius = 6378137
        delta_lat = (delta_y_metre / 2) / earth_radius * (180 / math.pi)
        delta_lon = (delta_x_metre / 2) / (earth_radius * math.cos(math.radians(y))) * (180 / math.pi)

        # Calculate corner coordinates
        top_left = (y + delta_lat, x - delta_lon)
        top_right = (y + delta_lat, x + delta_lon)
        bottom_left = (y - delta_lat, x - delta_lon)
        bottom_right = (y - delta_lat, x + delta_lon)

    elif units == 'metre':
        # Calculate corner coordinates
        top_left = (x - delta_x_metre/2, y + delta_y_metre/2)
        top_right = (x + delta_x_metre/2, y + delta_y_metre/2)
        bottom_left = (x - delta_x_metre/2, y - delta_y_metre/2)
        bottom_right = (x + delta_x_metre/2, y - delta_y_metre/2)

    else:
        raise ValueError(f"CRS {crs} not supported")

    return {
        "top_left": top_left,
        "top_right": top_right,
        "bottom_left": bottom_left,
        "bottom_right": bottom_right
    }

def centre_format_to_bbox(
    cx: float,
    cy: float,
    wx: float = 1000,
    yx: Optional[float] = None,
    resolution: float = 1.0,
    pixel_size: bool = True,
    epsg_in: str = '',
    epsg_out: str = ''
) -> Dict[str, Tuple[float, float]]:
    """
    Computes a bounding box around a center point, optionally transforming from one CRS to another.

    Parameters:
    ----------
    cx : float
        Center X coordinate (longitude or easting).
    cy : float
        Center Y coordinate (latitude or northing).
    wx : float
        Width in meters or pixels (default: 1000).
    yx : float, optional
        Height in meters or pixels (if None, defaults to same as wx).
    resolution : float
        Resolution in meters per pixel (used only if pixel_size is True).
    pixel_size : bool
        If True, wx and yx are treated as pixel dimensions and multiplied by resolution.
    epsg_in : str
        EPSG code of the input coordinates, e.g. 'EPSG:4326'.
    epsg_out : str
        EPSG code of the output coordinates. If empty or same as input, no transform is applied.
    """
    if yx is None:
        yx = wx

    # Convert to metres if dimensions are in pixels
    if pixel_size:
        wx *= resolution
        yx *= resolution

    # Transform center point to output CRS if needed
    if epsg_in and epsg_out and epsg_in != epsg_out:
        transformer = Transformer.from_crs(epsg_in, epsg_out, always_xy=True)
        cx, cy = transformer.transform(cx, cy)


    return get_bbox_coordinates(cx, cy, wx, yx, epsg_out)

#

# Why not using that object for cropping by constructing the crop from x, y ,z tile (less margine on the choice of the zoom)
# and then us a buffer on the tiles we know we will still need to use (cause we know the tiling patern)
# each tile store oll the x y z tils needed
# when no tile will need a tile remoove from the buffer 
# this is the format of inference that will ask the server the less load

# simpler solution is using a 
# 1 hop of 0.5 on 2* 2* 256 tiles 
# 2 or eaven 0.75 on 4* 4* 256 (1024 * 1024)
# then you only store the bottom overlapping thile but it souds like useless optimization

class OrthoPyramidApi:
    def __init__(self):
        pass
    
    def get(self):
        pass

class OrthoPyramidWebApi(OrthoPyramidApi): # TODO TO TEST

    # https://support.plexearth.com/hc/en-us/articles/6325794324497-Understanding-Zoom-Level-in-Maps-and-Imagery
    def __init__(
        self,
        base_url: str,
        layer: str,
        epsg_query: str = 'EPSG:3857',
        style: str = 'default',
        tile_matrix_set: str = 'GoogleMapsCompatible',
        service: str = 'WMTS',
        request: str = 'GetTile',
        version: str = '1.0.0',
        img_format: str = 'image/jpeg',
        tile_size: int = 256,
        verbose: bool = False
    ):
        self.base_url = base_url
        self.layer = layer
        self.style = style
        self.tile_matrix_set = tile_matrix_set
        self.service = service
        self.request = request
        self.version = version
        self.img_format = img_format
        self.tile_size = tile_size
        self.epsg_query = epsg_query
        self.verbose = verbose

        # Default GoogleMapsCompatible resolutions (EPSG:3857, meters per pixel)
        self.resolutions = {
            z: 156543.03392804097 / (2 ** z) for z in range(0, 20)
        }

        self.origin = (-20037508.342789244, 20037508.342789244)  # Top-left corner in EPSG:3857

    def get_tile_indices(self, x: float, y: float, zoom: int) -> Tuple[int, int]:
        res = self.resolutions[zoom]
        origin_x, origin_y = self.origin
        tile_x = int((x - origin_x) / (res * self.tile_size))
        tile_y = int((origin_y - y) / (res * self.tile_size))
        return tile_x, tile_y

    def get_tile_bbox(self, tile_x, tile_y, zoom):
        res = self.resolutions[zoom]
        tile_size = self.tile_size
        origin_x, origin_y = self.origin

        minx = origin_x + tile_x * tile_size * res
        maxx = origin_x + (tile_x + 1) * tile_size * res
        maxy = origin_y - tile_y * tile_size * res
        miny = origin_y - (tile_y + 1) * tile_size * res

        return (minx, miny, maxx, maxy)

    def get_tile(self, x: float, y: float, zoom: int):
        # Transform point to tile CRS

        tile_x, tile_y = self.get_tile_indices(x, y, zoom)

        url = (
            f"{self.base_url}?SERVICE={self.service}&REQUEST={self.request}&VERSION={self.version}"
            f"&LAYER={self.layer}&STYLE={self.style}&FORMAT={self.img_format}"
            f"&TILEMATRIXSET={self.tile_matrix_set}&TILEMATRIX={zoom}"
            f"&TILEROW={tile_y}&TILECOL={tile_x}"
        )

        if self.verbose:
            print(f"[INFO] Tile request URL: {url}")

        response = requests.get(url)
        response.raise_for_status()
        return {
                'image': Image.open(io.BytesIO(response.content)),
                'meta_data': {
                    'tile_col': tile_x,
                    'tile_row': tile_y,
                    'zoom': zoom,
                    'crs': self.epsg_query,
                    'resolution': self.resolutions[zoom],
                    'tile_bbox': self.get_tile_bbox(tile_x, tile_y, zoom)  # Optional: compute geographic extent (minx, miny, maxx, maxy)
                }
            }