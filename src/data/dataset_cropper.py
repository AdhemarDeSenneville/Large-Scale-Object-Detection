
from __future__ import annotations
import numpy as np
import io
import logging

from typing import Optional


from pyproj import Transformer
from owslib.wms import WebMapService
from PIL import Image

from .utils_api import centre_format_to_bbox, OrthoCropApi
from .utils_annotation import convert_bbox, convert_anything_with_transform
from .utils_image import convert_image
from .utils_crs import crs_units, format_crs

class OrthoCropDataset:
    def __init__(
            self, 
            api: OrthoCropApi,
            resolution: float,
            epsg_in: str,
            format_bbox_in = 'xyxy_list_single',
            format_image_out = 'cv2',
            window_size = 256,
    ):
        
        self.api = api

        self.resolution = resolution
        self.epsg_in = format_crs(epsg_in)
        self.format_bbox_in = format_bbox_in
        self.format_image_out = format_image_out
        self.window_size = window_size
        

    def geo_index_gps_positions(self, detections, format='old'):
        print("[INFO] geo_index_gps_positions is unused for that dataset.")
        return detections
    
    def get_image_from_bbox(
            self,
            bbox,
            size = None,
            resolution = None,
            epsg_in = None,
            format_bbox_in = None,
            format_image_out = None,
            save_path = None,
            layer = None
    ):
        
        epsg_in          = epsg_in          or self.epsg_in
        format_bbox_in   = format_bbox_in   or self.format_bbox_in
        format_image_out = format_image_out or self.format_image_out
        #layer            = layer            or self.api.layer
        

        
        if epsg_in != self.api.epsg_query:

            transformer = Transformer.from_crs(epsg_in, self.api.epsg_query, always_xy=True)
            def trasform_(x, y):
                x, y = transformer.transform(x, y)
                return x, y


            bbox = convert_anything_with_transform(
                bbox,
                transform = trasform_,
                object_in = 'bbox',
                format_in = format_bbox_in,
                format_out = 'xyxy_list_single',
            )
        else:
            bbox = convert_bbox(bbox, format_in = format_bbox_in, format_out = 'xyxy_list_single')


        if size is None:
            resolution       = resolution       or self.resolution

            if crs_units(self.api.epsg_query) == 'metre':
                size = (int(abs(bbox[2]-bbox[0])/resolution), int(abs(bbox[3] - bbox[1])/resolution))
            else:
                raise NotImplementedError
            
        import pdb#; pdb.set_trace()

        image = self.api.get(
            bbox,
            size,
            layer
        )
        
        import pdb

        if save_path is not None:
            image['image'].save(save_path)
            return True
        else:
            #image = convert_image(image, format_in = 'pil', format_out = format_image_out)
            return image


    def get_image_from_centre(
            self, 
            centre: tuple,
            window_size: Optional[tuple | int] = None,
            resolution = None,
            epsg_in = None,
            **kwarg, # to verify
        ):
        
        x, y = centre
        resolution       = resolution       or self.resolution
        epsg_in          = epsg_in          or self.epsg_in
        window_size = window_size or self.window_size

        if isinstance(window_size, int):
            window_size = (window_size, window_size)


        # Get bbox from center point using your logic
        bbox = centre_format_to_bbox(
            cx=x,
            cy=y,
            wx=window_size[0], # 
            yx=window_size[1],
            resolution=resolution,
            pixel_size=True,
            epsg_in=epsg_in,
            epsg_out=self.api.epsg_query,
        )

        #bbox = f"{coords['bottom_left'][0]},{coords['bottom_left'][1]},{coords['top_right'][0]},{coords['top_right'][1]}"

        bbox = [bbox['bottom_left'][0],bbox['bottom_left'][1],bbox['top_right'][0],bbox['top_right'][1]]


        return self.get_image_from_bbox(
            bbox,
            size = window_size,
            epsg_in = self.api.epsg_query,
            **kwarg,
        )

