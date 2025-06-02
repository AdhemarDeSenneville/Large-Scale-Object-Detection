import geopandas as gpd


class SubTileIterator:
    '''
    Iterator class for generating sub-tiles from larger tiles based on a given configuration.
    This class divides large geospatial tiles into smaller sub-tiles using a sliding window approach.

    It iterates through the tiles in a shapefile and generates bounding boxes for each sub-tile.
    '''

    def __init__(self, config):


        self.real_tile_size = 25_000
        self.tile_size = config['dataset']['tile_size']
        self.sub_tile_size = config['sub_tile_size']
        self.hop_size = config['hop_size']
        self.sub_tiles_per_tile_len = (self.tile_size - self.sub_tile_size) / self.hop_size
        self.sub_tiles_per_tile = self.sub_tiles_per_tile_len **2

        self.gdf_tiles = gpd.read_file(config['dataset']['tiles_shapefile_path'])
        
        # Print the important variables
        print("Configuration and Derived Values:")
        print(f"  Real Tile Size: {self.real_tile_size}")
        print(f"  Tile Size: {self.tile_size}")
        print(f"  Sub Tile Size: {self.sub_tile_size}")
        print(f"  Hop Size: {self.hop_size}")
        print(f"  Sub Tiles per Tile (Length): {self.sub_tiles_per_tile_len}")
        print(f"  Sub Tiles per Tile (Total): {self.sub_tiles_per_tile}")
        print(f"  Number of Tiles in Shapefile: {len(self.gdf_tiles)}\n")
        
        self.current_tile_index = -1
        self.current_sub_tile_index = 0
        self.current_tile_bounds = None
        self.current_tile_jp2_folder = None

    def __iter__(self):
        return self
        
    def __len__(self):
        # Total number of sub-tiles across all tiles
        return len(self.gdf_tiles) * self.sub_tiles_per_tile

    def reset(self):
        # Reset the iterator to the beginning
        self.current_tile_index = -1
        self.current_sub_tile_index = 0
        self.current_tile_bounds = None
        self.current_tile_jp2_folder = None

    def __next__(self):
        
        if self.current_sub_tile_index >= self.sub_tiles_per_tile or self.current_tile_index == -1:
            self.current_sub_tile_index = 0
            self.current_tile_index += 1

            
            if self.current_tile_index == len(self.gdf_tiles):
                raise StopIteration

            new_tile = True

            # Get the current tile geometry
            current_tile = self.gdf_tiles.iloc[self.current_tile_index]
            self.current_tile_bounds  = current_tile.geometry.bounds
            self.current_tile_jp2_folder = current_tile.NOM
            self.current_tile_region = current_tile.region
        else:
            new_tile = False

        tile_bounds = self.current_tile_bounds
        # Calculate the sub-tile grid
        sub_tile_col_index = self.current_sub_tile_index % self.sub_tiles_per_tile_len
        sub_tile_row_index = self.current_sub_tile_index // self.sub_tiles_per_tile_len

        # Calculate sub-tile bounds in geographic coordinates
        sub_tile_cartesian_len = (tile_bounds[2] - tile_bounds[0]) * self.sub_tile_size / self.tile_size
        sub_tile_cartesian_hop = (tile_bounds[2] - tile_bounds[0]) * self.hop_size / self.tile_size
        sub_tile_cartesian_minx = tile_bounds[0] + sub_tile_col_index * sub_tile_cartesian_hop
        sub_tile_cartesian_miny = tile_bounds[1] + sub_tile_row_index * sub_tile_cartesian_hop
        sub_tile_cartesian_maxx = sub_tile_cartesian_minx + sub_tile_cartesian_len
        sub_tile_cartesian_maxy = sub_tile_cartesian_miny + sub_tile_cartesian_len

        cartesian_bbox = (
            sub_tile_cartesian_minx, 
            sub_tile_cartesian_miny, 
            sub_tile_cartesian_maxx, 
            sub_tile_cartesian_maxy,
            )

        # Calculate the pixel bounds within the current tile
        sub_tile_col_pixel_start = sub_tile_col_index * self.hop_size
        sub_tile_row_pixel_start = sub_tile_row_index * self.hop_size
        sub_tile_col_pixel_end = sub_tile_col_pixel_start + self.sub_tile_size
        sub_tile_row_pixel_end = sub_tile_row_pixel_start + self.sub_tile_size

        # Convention for images
        sub_tile_row_pixel_start_ = self.tile_size - sub_tile_row_pixel_end
        sub_tile_row_pixel_end_ = self.tile_size - sub_tile_row_pixel_start

        pixel_bbox = (
            sub_tile_col_pixel_start,    #round(sub_tile_col_pixel_start),
            sub_tile_row_pixel_start_,    #round(sub_tile_row_pixel_start_),
            sub_tile_col_pixel_end,    #round(sub_tile_col_pixel_end),
            sub_tile_row_pixel_end_,    #round(sub_tile_row_pixel_end_),
        )

        # Move to the next sub-tile
        self.current_sub_tile_index += 1
        

        # Return the sub-tile bounding box, folder of JP2, and pixel bounding box
        return {
            "cartesian_bbox": cartesian_bbox,
            "pixel_bbox": pixel_bbox,
            "sub_tile_col_index": sub_tile_col_index,
            "sub_tile_row_index": sub_tile_col_index,
            "tile_index": self.current_tile_index,
            "tile_size": self.tile_size,
            "tile_new": new_tile,
            "tile_jp2_folder": self.current_tile_jp2_folder,
            "tile_region": self.current_tile_region,
            "tile_bounds": tile_bounds,
        }