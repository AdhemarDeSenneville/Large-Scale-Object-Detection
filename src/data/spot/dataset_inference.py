import json
import math
from shapely.geometry import Point, Polygon
import rasterio
from rasterio.windows import Window
import numpy as np
from pyproj import Transformer
import cv2
import random
from tqdm import tqdm
import geopandas as gpd
from collections import defaultdict
from os.path import join
import os

from .utils import process_image_data
from .utils import draw_bounding_boxes, get_random_points_in_polygon, convert_to_dota_format, signed_distance
from ..utils_annotation import get_bbox, get_center, get_obb, get_segmentation

from env import PATH_METHANIZERS_2_TRAIN_JSON

class TiledDataset:
    def __init__(self, json_path, mode = 'inference'):

        with open(json_path, 'r', encoding='utf-8') as f:
            self.tiles = json.load(f)
        for tile in self.tiles:
            tile["polygon"] = Polygon(tile["corners"])
            tile["transform_mat"] = np.array(tile["transform"]).reshape(3,3)
            
        if mode == 'train':
            with open(PATH_METHANIZERS_2_TRAIN_JSON, 'r') as f:
                train_data = json.load(f)

            self.images_known = train_data['images']
            self.annotations_known = train_data['annotations']

            transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
            self.images_coord = [transformer.transform(img['excel_data']["Longitude (\u00b0E)"],img['excel_data']["Latitude (\u00b0N)"]) for img in self.images_known]
            self.images_annotations = []
            for img in self.images_known:
                img_id = img['id']

                self.images_annotations.append([
                    ann for ann in self.annotations_known if ann['image_id'] == img_id
                ])
            
            random.seed(42)
            self.val_train_split = [random.random() < 0.2 for _ in self.images_known]  # 20% validation, 80% training

            # self.bg_images_coord is a list same size as self.images_coord of EPSG:2154 coordinnates randomely selected in france 
            
            france_shp_path = "/home/adhemar/Bureau/datasets/France/gadm41_FRA_shp/gadm41_FRA_0.shp"
            france_gdf = gpd.read_file(france_shp_path)
            france_gdf = france_gdf.to_crs(epsg=2154)
            france_polygon = france_gdf.unary_union
            random.seed(42)
            bg_points = get_random_points_in_polygon(france_polygon, len(self.images_coord) + 400, seed=42)
            self.bg_images_coord = [(pt.x, pt.y) for pt in bg_points]


    def generate_dataset(self, save_path):
        
        save_images = join(save_path, 'images')
        save_images_annotated = join(save_path, 'images_annotated')
        save_coco_annotations = join(save_path, 'annotations')
        os.makedirs(save_images, exist_ok=True)
        os.makedirs(save_images_annotated, exist_ok=True)
        os.makedirs(save_coco_annotations, exist_ok=True)

        save_train_images = join(save_path, 'train', 'images')
        save_train_annotations = join(save_path, 'train', 'labels')
        os.makedirs(save_train_images, exist_ok=True)
        os.makedirs(save_train_annotations, exist_ok=True)

        save_val_images = join(save_path, 'val', 'images')
        save_val_annotations = join(save_path, 'val', 'labels')
        os.makedirs(save_val_images, exist_ok=True)
        os.makedirs(save_val_annotations, exist_ok=True)

        manualy_filtred = [
            10, 16, 41, 44, 48, 83,
            106, 124, 141, 183, 186,
            188, 195
        ]

        count = 0
        for coord, anns, is_val in tqdm(zip(self.images_coord, self.images_annotations, self.val_train_split)):
            count+=1

            #if count >= 6:
            #    break

            if count in manualy_filtred:
                continue
            
            tile, dist  = self.find_tile_covering_or_closest(*coord)
            
            if dist<0:
                image_name = f"{count:04d}.png"
                text_name = f"{count:04d}.txt"

                data, window_col_off, window_row_off = self.read_window_from_tile(tile, *coord, 1000)
                pixel_annotations = self.convert_anns_from_tile(tile, anns, window_col_off, window_row_off)

                image = process_image_data(data)
                #print(pixel_annotations)
                #print(image.shape)

                image_annotated = draw_bounding_boxes(image, pixel_annotations)

                cv2.imwrite(join(save_images_annotated, image_name), image_annotated)
                cv2.imwrite(join(save_images, image_name), image)

                dota_label = convert_to_dota_format(pixel_annotations) # TODO
                if is_val:

                    with open(join(save_val_annotations, text_name), 'a') as f:
                        f.write(dota_label)
                    
                    cv2.imwrite(join(save_val_images, image_name), image)
                else:

                    with open(join(save_train_annotations, text_name), 'a') as f:
                        f.write(dota_label)

                    cv2.imwrite(join(save_train_images, image_name), image)

                #if count==3:
                #    return data, pixel_annotations
            else:
                print("Not Found at :", coord)

        print(count, "Now Background Tiles")
        manualy_filtred_bg = []
        count = 0
        for coord in tqdm(self.bg_images_coord):
            count+=1

            #if count >= 6:
            #    break

            if count in manualy_filtred_bg:
                continue
            
            tile, dist  = self.find_tile_covering_or_closest(*coord)
            
            if dist<0:
                image_name = f"bg_{count:04d}.png"
                text_name = f"bg_{count:04d}.txt"

                data, window_col_off, window_row_off = self.read_window_from_tile(tile, *coord, 1000)
                image = process_image_data(data)
                cv2.imwrite(join(save_images, image_name), image)

                if count < 400:

                    with open(join(save_val_annotations, text_name), 'a') as f:
                        f.write('')
                    
                    cv2.imwrite(join(save_val_images, image_name), image)
                else:

                    with open(join(save_train_annotations, text_name), 'a') as f:
                        f.write('')

                    cv2.imwrite(join(save_train_images, image_name), image)

                #if count==3:
                #    return data, pixel_annotations
            else:
                print("Not Found at :", coord)

        print(count)


    def geo_index_gps_positions(self, positions, format = 'old'):

        if format == 'old':
            from collections import defaultdict
            tile_map = defaultdict(list)

            for ann in positions:
                # ann is a list with one dict, so we do:
                x, y = ann['position']
                tile, dist = self.find_tile_covering_or_closest(x, y)
                tile_key = tile["file"]
                tile_map[tile_key].append(ann)

            
            sorted_annotations = []
            for tile_key in sorted(tile_map.keys()):
                # tile_map[tile_key] is a list of annotations belonging to tile_key
                sorted_annotations.extend(tile_map[tile_key])
        
            return sorted_annotations
        
        elif format == 'geojson':
            from collections import defaultdict
            tile_map = defaultdict(list)

            for ann in positions:
                # ann is a list with one dict, so we do:
                x, y = ann['geometry']['coordinates']
                tile, dist = self.find_tile_covering_or_closest(x, y)
                tile_key = tile["file"]
                tile_map[tile_key].append(ann)

            
            sorted_annotations = []
            for tile_key in sorted(tile_map.keys()):
                # tile_map[tile_key] is a list of annotations belonging to tile_key
                sorted_annotations.extend(tile_map[tile_key])
        
            return sorted_annotations
        
        elif format == 'geopandas':
            import geopandas as gpd
            from collections import defaultdict
            tile_map = defaultdict(list)

            # Iterate through each row in the GeoDataFrame
            for idx, row in positions.iterrows():
                # Extract x, y from the geometry
                x, y = row.geometry.x, row.geometry.y

                tile, dist = self.find_tile_covering_or_closest(x, y)
                tile_key = tile["file"]
                tile_map[tile_key].append(row)

            # Build a list of rows in sorted tile order
            sorted_rows = []
            for tile_key in sorted(tile_map.keys()):
                sorted_rows.extend(tile_map[tile_key])

            # Convert the list of rows back into a GeoDataFrame
            sorted_annotations = gpd.GeoDataFrame(
                sorted_rows, 
                columns=positions.columns, 
                crs=positions.crs
            )

            return sorted_annotations
            # position is a frame with geometry position


    def convert_anns_from_tile(self, tile, anns, window_col_off, window_row_off):
        
        pixel_annotations = []
        for annotation in anns:

            epsg2154_segmentation = np.array(annotation['epsg2154_segmentation']).reshape(-1, 2)
            transform_mat = tile["transform_mat"]

            homogeneous_coords = np.hstack((epsg2154_segmentation, np.ones((epsg2154_segmentation.shape[0], 1))))
            pix_segmentation = (np.linalg.inv(transform_mat) @ homogeneous_coords.T).T[:, :2]
            pix_segmentation[:, 0] -= window_col_off
            pix_segmentation[:, 1] -= window_row_off


            segm_segmentation = get_segmentation(pix_segmentation, annotation['category_id'] - 1)
            segm_bbox = get_bbox(pix_segmentation, annotation['category_id'] - 1)
            segm_obb = get_obb(pix_segmentation, annotation['category_id'] - 1)
            center_xy = get_center(pix_segmentation, annotation['category_id'] - 1)


            pixel_annotations.append({
                'segmentation': segm_segmentation,
                'bbox': segm_bbox['bbox_xyxy'],
                'center': center_xy,
                'bboxes': segm_bbox,
                'obbs': segm_obb,
                'category_id': annotation['category_id'],
            })

        return pixel_annotations


    def find_tile_covering_or_closest(self, x: float, y: float):
        point = Point(x, y)

        best_tile = None
        best_dist = math.inf

        for tile in self.tiles:
            polygon = tile["polygon"]
            dist = signed_distance(point, polygon)
            if dist < best_dist:
                best_dist = dist
                best_tile = tile

            # Optional shortcut: if dist == 0, the point lies exactly on the boundary
            #   -> we can break early if desired.

        return best_tile, best_dist


    def read_window_from_tile(
        self, tile, x: float, y: float, window_size: int, nodata_fill=None
    ):
        transform_mat_inv = np.linalg.inv(tile["transform_mat"])
        
        map_coords = np.array([x, y, 1], dtype=np.float64)
        pix_coords = transform_mat_inv @ map_coords  # [col, row, 1]
        col_center, row_center, _ = pix_coords

        # Round to nearest integer pixel coordinate
        col_center = int(round(col_center))
        row_center = int(round(row_center))

        # Create a Window (rasterio)
        half = window_size // 2
        window_col_off = col_center - half
        window_row_off = row_center - half

        # The Window object: (col_off, row_off, width, height)
        window = Window(window_col_off, window_row_off, window_size, window_size)

        # Now, open the image and read the data. 
        data = None
        with rasterio.open(tile["file"]) as src:
            # Read the window
            # If the window is partially outside the dataset, you can read with boundless=True
            # and specify fill_value if desired. For example:
            data = src.read(window=window, boundless=True, fill_value=nodata_fill)

        return data, window_col_off, window_row_off


    def get_image_window(self, x: float, y: float, window_size: int):
        """
        Public method that:
          1) Finds the tile covering or closest to the point (x, y).
          2) Reads a window of `window_size x window_size` around that point.
          3) Returns the windowed data as a NumPy array (bands, window_size, window_size).
        """
        tile, distance = self.find_tile_covering_or_closest(x, y)
        if distance >= 0:
            return None
        
        data, _, _ = self.read_window_from_tile(tile, x, y, window_size)
        return data, tile, distance


    def get_rgb_image(self, x, y, window_size=1000):
        
        img_array, _, _ = self.get_image_window(x, y, window_size)
        return process_image_data(img_array, out_bgr = False)


    def get_annotated_image(self, detec, crop):
        pass # TODO