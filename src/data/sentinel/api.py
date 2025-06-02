from ..utils_api import OrthoCropApi

from .downloader import *
import cv2
from PIL import Image

class OrthoCropSentinelApi(OrthoCropApi):
    """
    A simple client to request orthophoto crops from a WMS server.
    """
    def __init__(
        self,
        epsg_query: str = "EPSG:4326",
        api_backend: str = 'microsoft',
        #img_format: str = 'image/geotiff',
        collection: str = "sentinel-2-l2a",
        date_range=("2024-01-01", "2024-03-01"),
        min_cloud_cover = 10,
        verbose = False,
        **kwargs
    ):

        assert epsg_query == "EPSG:4326"
        self.epsg_query = "EPSG:4326"


        self.date_range = date_range 
        self.api_backend = api_backend
        self.min_cloud_cover = min_cloud_cover
        
        if self.api_backend == 'microsoft':
            self.API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
            self.client = Client.open(self.API_URL, modifier=planetary_computer.sign_inplace)
        elif self.api_backend == 'element84':
            self.API_URL = "https://earth-search.aws.element84.com/v1"
            self.client = Client.open(self.API_URL)


        self.collection = [collection]

        if verbose:
            print('Collection: ',self.collection)
            print('api_backend: ',self.api_backend)

        super().__init__(**kwargs)

    def get_once(
        self,
        bbox: tuple,
        size: tuple,
        layer: str = None,
    ):
        image_array = self.get_sentinel2_image(bbox)

        image = self.pre_process(image_array)

        image = cv2.resize(image, size)

        # if shape no equal size re size

        return {
            'image': Image.fromarray(image),
            'meta_data': {'size': size, 'bbox': bbox, 'epsg': self.epsg_query}
        }
        
    def get_sentinel2_image(self, bbox, verbose=False):
        
        lat_min, lon_min, lat_max, lon_max = bbox
        # swap into (minx, miny, maxx, maxy) = (lon_min, lat_min, lon_max, lat_max)
        bbox = box(lon_min, lat_min, lon_max, lat_max)
        
        # Search for images containing the bouding box using the STAC API
        search = self.client.search(
            collections=self.collection,
            intersects=bbox,
            query={"eo:cloud_cover": {"lt": self.min_cloud_cover}},
            datetime=f"{self.date_range[0]}/{self.date_range[1]}"
        )
        

        items = search.item_collection()
        if not items or len(items) == 0:
            print("No suitable images found.")
            return None
        
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
            if self.api_backend == 'microsoft': # Microsoft api is usually faster than element84
                blue_href = assets["B02"].href
                green_href = assets["B03"].href
                red_href = assets["B04"].href
            elif self.api_backend == 'element84':
                blue_href = assets["blue"].href
                green_href = assets["green"].href
                red_href = assets["red"].href



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

        # return rgb_list, memfile_list
        return rgb_list[0]


    def pre_process(self, img, min = 1000, max = 3500):
        img = img.copy()
        img = img.astype(np.float32)
        img = np.clip(img, min, max) - min
        img = (img / (max - min)) * 255.0
        return img.astype(np.uint8).transpose((1, 2, 0))