import os
import json
import geopandas as gpd
import numpy as np
from shapely.wkt import loads
import rasterio
from rasterio.windows import Window
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch.utils.data import Sampler, BatchSampler
import math

from .utils import process_image_data


class CoverageTileDataset(Dataset):
    def __init__(self, folder_path, selected_big_tiles = None):
        self.folder_path = folder_path
        self.tile_folders = []
        self.catalog = []

        # Gather big tile folders and build a flattened list of sub-tiles
        big_tile_dirs = [
            f for f in os.listdir(folder_path)
            if os.path.isdir(os.path.join(folder_path, f))
        ]
        big_tile_dirs.sort()

        if selected_big_tiles is not None:
            big_tile_dirs = [big_tile_dirs[i] for i in selected_big_tiles]

        cumulative_idx = 0
        for big_tile_idx, tile_dir_name in enumerate(big_tile_dirs):
            tile_dir_path = os.path.join(folder_path, tile_dir_name)
            shp_path = os.path.join(tile_dir_path, "tiles_shape.shp")
            if not os.path.isfile(shp_path):
                continue  # skip if no shapefile

            gdf = gpd.read_file(shp_path)
            n_sub_tiles = len(gdf)

            for i in range(n_sub_tiles):
                self.catalog.append({
                    'big': big_tile_idx,
                    'sub': i,
                })

            self.tile_folders.append(tile_dir_path)
            cumulative_idx += n_sub_tiles
            
        
        # We will cache the currently loaded "big tile" to avoid re-opening repeatedly
        self._current_big_tile_idx = None
        self._current_gdf = None
        self._current_metadata = None
        self._current_raster = None

    def plot_big_tiles(self):
        fig, ax = plt.subplots()
        
        # Iterate all big tile folders
        for i, folder in enumerate(self.tile_folders):
            shp_path = os.path.join(folder, "tiles_shape.shp")
            if not os.path.isfile(shp_path):
                continue
            gdf = gpd.read_file(shp_path)
            
            # We could compute the bounding box or centroid of all sub-tiles
            # For simplicity, let's get the centroid of the union of all polygons
            # (Be mindful of memory if you have many polygons)
            union_poly = gdf.unary_union
            center = union_poly.centroid
            
            ax.plot(center.x, center.y, 'ro')
            ax.text(center.x, center.y, str(i), color='black', fontsize=8)

        ax.set_title("Big Tiles Centers")
        plt.show()

    def __len__(self):
        return len(self.catalog)

    def _load_big_tile(self, big_tile_idx):
        """
        If we need a new big tile, load its shapefile, metadata, and raster.
        Cache them for subsequent calls.
        """
        if big_tile_idx == self._current_big_tile_idx:
            return False

        # Else load new tile
        tile_path = self.tile_folders[big_tile_idx]
        shp_path = os.path.join(tile_path, "tiles_shape.shp")
        self._current_gdf = gpd.read_file(shp_path)

        metadata_path = os.path.join(tile_path, "tile_metadata.json")
        with open(metadata_path, "r") as f:
            self._current_metadata = json.load(f)

        jp2_path = os.path.join(tile_path, self._current_metadata["file"])
        self._current_raster = rasterio.open(jp2_path)
        self._current_big_tile_idx = big_tile_idx

        return True

    def __getitem__(self, idx):
        """
        Return a dictionary with:
          {
            "image": ...,
            "tile": row of shapefile,
            "transform": 3x3 transform (optional),
            "big_tile_idx": ...,
            "sub_tile_idx": ...
          }
        """
        record = self.catalog[idx]
        big_tile_idx = record['big']
        sub_tile_idx = record['sub']

        # Load the big tile data if not already loaded
        is_new = self._load_big_tile(big_tile_idx)

        # Now read the sub-tile
        tile_row = self._current_gdf.iloc[sub_tile_idx]
        polygon = loads(tile_row["pixels"])
        minx, miny, maxx, maxy = polygon.bounds

        window = Window(
            col_off=minx, 
            row_off=miny, 
            width=(maxx - minx), 
            height=(maxy - miny)
        )
        image_data = self._current_raster.read(window=window)
        image_data = process_image_data(image_data)

        # Build correct transform
        transform = np.array(self._current_metadata["transform"]).reshape(3,3)
        transform[0,2] += minx * transform[0,0]
        transform[1,2] += miny * transform[1,1]

        output = {
            "image": image_data,
            "transform": transform,
            "is_new": is_new,
            "big_tile_idx": big_tile_idx,
            "sub_tile_idx": sub_tile_idx,
        }
        return output
    

class OneTileAtATimeBatchSampler(BatchSampler):
    """
    Yields batches from exactly one big tile at a time,
    splitting large tile groups into multiple smaller batches.
    """
    def __init__(self, dataset, max_batch_size=32, drop_last=False):
        """
        :param dataset: the TileDataset
        :param max_batch_size: max number of sub-tiles per batch
        :param drop_last: whether to drop the last incomplete batch in each tile
        """
        self.dataset = dataset
        self.max_batch_size = max_batch_size
        self.drop_last = drop_last
        self.grouped_indices = {}
        for ds_idx, record in enumerate(self.dataset.catalog):
            big_idx = record['big']
            if big_idx not in self.grouped_indices:
                self.grouped_indices[big_idx] = []
            self.grouped_indices[big_idx].append(ds_idx)

        self.big_tile_ids = sorted(self.grouped_indices.keys())

    def __iter__(self):
        """
        Yields lists (batches) of dataset indices.
        Each batch only has indices from one big tile.
        """
        for big_idx in self.big_tile_ids:
            indices_for_tile = self.grouped_indices[big_idx]
            # Split into chunks of size self.max_batch_size
            num_chunks = math.ceil(len(indices_for_tile) / self.max_batch_size)
            for i in range(num_chunks):
                start = i * self.max_batch_size
                end = start + self.max_batch_size
                batch_indices = indices_for_tile[start:end]

                if self.drop_last and len(batch_indices) < self.max_batch_size:
                    # skip incomplete batch
                    continue

                yield batch_indices

    def __len__(self):
        # Count total
        count = 0
        for big_idx in self.big_tile_ids:
            n = len(self.grouped_indices[big_idx])
            c = math.ceil(n / self.max_batch_size)
            if self.drop_last:
                c = n // self.max_batch_size
            count += c
        return count

