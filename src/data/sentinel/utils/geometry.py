from shapely.geometry import box
import numpy as np
import io
from math import cos, radians
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from rasterio.crs import CRS
from pyproj import Transformer

def crop_image_to_bbox(
    image_path,
    input_bbox,
    output_path,
    source_crs: str = 'EPSG:4326'  # Default to GPS coordinates
):
    with rasterio.open(image_path) as src:
        image_crs = src.crs
        image_bounds = src.bounds
        pixel_width, pixel_height = src.res

        image_crs_obj = CRS.from_user_input(image_crs)
        source_crs_obj = CRS.from_user_input(source_crs)

        # Transform GPS bbox to image CRS
        if source_crs_obj != image_crs_obj:
            reprojected_bbox = transform_bounds(
                source_crs_obj, image_crs_obj, *input_bbox
            )
        else:
            reprojected_bbox = input_bbox

        left, bottom, right, top = reprojected_bbox

        # Convert coordinates to pixel offsets
        col_off = int((left - image_bounds.left) / pixel_width)
        row_off = int((image_bounds.top - top) / pixel_height)
        width = int((right - left) / pixel_width)
        height = int((top - bottom) / pixel_height)

        if width <= 0 or height <= 0:
            raise ValueError("Invalid crop size: width and height must be > 0")

        # Clamp offsets
        col_off = max(0, min(col_off, src.width - width))
        row_off = max(0, min(row_off, src.height - height))

        window = Window(col_off, row_off, width, height)
        data = src.read(window=window)

        profile = src.profile.copy()
        profile.update(
            height=height,
            width=width,
            transform=src.window_transform(window)
        )

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data)

        print(f"Cropped image saved to: {output_path}")


def delta_km_to_deg(lat, delta_x_km, delta_y_km):
    delta_lat_deg = delta_y_km / 111.32
    delta_lon_deg = delta_x_km / (111.32 * cos(radians(lat)))
    return (delta_lon_deg, delta_lat_deg)

def reproject_bounds(bounds, src_crs, dst_crs):
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    minx, miny = transformer.transform(bounds[0], bounds[1])
    maxx, maxy = transformer.transform(bounds[2], bounds[3])
    return (minx, miny, maxx, maxy)