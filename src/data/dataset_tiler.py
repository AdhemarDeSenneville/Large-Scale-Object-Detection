import os
import json
import numpy as np
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt

import geopandas as gpd
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
import folium
import geopandas as gpd
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.wkt import loads
from pyproj import CRS

from .utils_annotation import convert_segmentation_to_bbox, convert_anything_with_lerp, convert_anything_to_segmentation
from .utils_image import convert_image
import time

def signed_distance(tile, reference_polygon):
    d = tile.distance(reference_polygon.boundary)

    # If distance == 0, they touch or intersect => signed distance = 0
    if d == 0:
        return 0.0
    
    # Check if centroid is inside the polygon
    if reference_polygon.contains(tile.centroid):
        return -d
    else:
        return d

def densify_box(minx, miny, maxx, maxy, step=10):
    """Return a densified box polygon in the source CRS."""
    # Sample points along each edge
    xs = np.linspace(minx, maxx, step)
    ys = np.linspace(miny, maxy, step)

    top = [(x, maxy) for x in xs]
    right = [(maxx, y) for y in ys[::-1]]
    bottom = [(x, miny) for x in xs[::-1]]
    left = [(minx, y) for y in ys]

    ring = top + right + bottom + left + [top[0]]
    return Polygon(ring)

class TileSlidingGenerator():
    
    def __init__(
            self,
            zones_to_map,
            zones_with_pixels,
            epsg,
            folder,
            tiling_config: dict,
        ):

        self.zones_to_map = zones_to_map.to_crs(epsg)
        self.zones_with_pixels = zones_with_pixels.to_crs(epsg)
        self.epsg = epsg

        self.folder = folder
        self.folder_data = os.path.join(folder, 'data')
        self.folder_meta = os.path.join(folder, 'meta')
        
        os.makedirs(self.folder, exist_ok=True)
        os.makedirs(self.folder_data, exist_ok=True)
        os.makedirs(self.folder_meta, exist_ok=True)

        self.tile_size_metre = tiling_config['tile_size_metre']
        self.hop_fraction = tiling_config['hop_fraction']
        self.hop_size_metre = self.hop_fraction * self.tile_size_metre

        self.tiles = None

    def construct_tiling(self):
        raise NotImplementedError("construct_tiling() must be implemented in subclass")

    def get_trad_of(self, annotation_objects_path = None, compute_mode = 'math'):

        if annotation_objects_path is None:
            from env import PATH_METHANIZERS_2_TRAIN_JSON
                # Load the COCO labels JSON file
            annotation_objects_path = PATH_METHANIZERS_2_TRAIN_JSON
        
        with open(annotation_objects_path, 'r') as f:
            coco_data = json.load(f)

        coco_data['annotations'][:5]
        object_sizes_metres = {}

        for annotation in coco_data['annotations']:
            if annotation['category_id'] == 1:
                image_id = annotation['image_id']
                segm = annotation['epsg2154_segmentation']

                xs = segm[::2]
                ys = segm[1::2]

                w = max(xs) - min(xs)
                h = max(ys) - min(ys)
                object_sizes_metres[image_id] = (w, h)

        # Compute the characteristic length (sqrt of the area)


        # Compute the mean dimensions of the bounding boxes in pixels
        mean_width_pix = np.mean([dim[0] for dim in object_sizes_metres.values()])
        mean_height_pix = np.mean([dim[1] for dim in object_sizes_metres.values()])
        print(f"Mean Width:  ({mean_width_pix:.2f} metre)")
        print(f"Mean Height: ({mean_height_pix:.2f} metre)")
        #assert abs((config['resolution'] / 0.2 +0.5 )% 1 -0.5) < 0.000001

        def is_bbox_within_tile(bbox, tile):

            tile_x, tile_y, tile_width, tile_height = tile
            bbox_x, bbox_y, bbox_width, bbox_height = bbox

            return (tile_x <= bbox_x < tile_x + tile_width and
                    tile_y <= bbox_y < tile_y + tile_height and
                    tile_x <= bbox_x + bbox_width <= tile_x + tile_width and
                    tile_y <= bbox_y + bbox_height <= tile_y + tile_height)

        def get_uncut_probability(sub_tile_size, hop_fraction, object_sizes_metres, n_simu = 50):
            #compute_mode = 'math'
            # Every thing is in metres
            probabilities = []
            for i, values in object_sizes_metres.items():
                width, height = values

                if compute_mode == 'simulation':
                    hop_size = int(sub_tile_size * hop_fraction)

                    count_within_tile = 0
                    for i_simu in range(n_simu):
                        #print(i_simu, 'NEW SIMU')
                        detected = False
                        tile_x = np.random.randint(0, sub_tile_size)
                        tile_y = np.random.randint(0, sub_tile_size)

                        #print('x',0, tile_x, tile_x + width, sub_tile_size)
                        #print('y',0, tile_y, tile_y + height, sub_tile_size)
                        
                        for x in range(0, sub_tile_size, round(hop_size)):
                            for y in range(0, sub_tile_size, round(hop_size)):
                                if is_bbox_within_tile((tile_x, tile_y, width, height), (x, y, sub_tile_size, sub_tile_size)):
                                    detected = True
                                    break
                        
                        count_within_tile += int(detected)
                
                    probability = count_within_tile / n_simu

                elif compute_mode == 'math':
                    pw = (width - (1-hop_fraction)*sub_tile_size)/(sub_tile_size)
                    pw = max(0,pw)
                    pw = min(1,pw)

                    ph = (height - (1-hop_fraction)*sub_tile_size)/(sub_tile_size)
                    ph = max(0,ph)
                    ph = min(1,ph)

                    probability = 1 - pw * ph - ph - pw
                
                else:
                    raise NotImplementedError
                
                probabilities.append(probability)



            mean_probability = np.mean(probabilities)
            #print(f"\nMean Probability: {100*mean_probability:.2f}% (of being insinde a tile)")

            return mean_probability
        # Generate values for hop fraction
        hop_fractions = np.linspace(0.5, 1.0, 101)
        sub_tile_size = self.tile_size_metre

        probabilities = []
        npasses = []
        for hop_fraction in tqdm(hop_fractions):

            prob = get_uncut_probability(sub_tile_size, hop_fraction, object_sizes_metres,  n_simu= 100)
            npass = (1 / hop_fraction)**2
            probabilities.append(prob)
            npasses.append(npass)

        # Plot both Probability and Number of Passes on the same graph
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Plot Probability on the left y-axis
        ax1.set_xlabel("Hop Length Fraction")
        ax1.set_ylabel("Probability of Detection", color='tab:blue')
        ax1.plot(hop_fractions, probabilities, marker='o', color='tab:blue', label="Probability of not Cutting the Object")
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # Create a second y-axis for Number of Passes
        ax2 = ax1.twinx()
        ax2.set_ylabel("Number of Passes", color='tab:red')
        ax2.plot(hop_fractions, npasses, marker='s', color='tab:red', linestyle='dashed', label="Inferences per Pixel")
        ax2.tick_params(axis='y', labelcolor='tab:red')

        # Optimal hop length selection (e.g., intersection or balance point)
        optimal_hop_length = hop_fractions[np.argmax(np.array(probabilities) - np.array(npasses)/max(npasses))]
        ax1.axvline(optimal_hop_length, color='gray', linestyle='dotted', label=f'Optimal Hop Length: {optimal_hop_length:.2f}')

        # Titles and Legends
        fig.suptitle("Optimization of Hop Length for Detection Performance")
        fig.tight_layout()
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.savefig(os.path.join(self.folder_meta, 'tilling_design_choice_2.pdf'))
        plt.show()

        # Find optimal hop by your chosen criterion

        probabilities_cut = 1 - np.array(probabilities)
        #opt_idx = np.argmax(1 - probabilities_cut - npasses/np.max(npasses)) - 4
        opt_idx = np.argmin(np.abs(np.array(hop_fractions) - self.hop_fraction))
        opt_prob, opt_pass = probabilities_cut[opt_idx], npasses[opt_idx]
        opt_hop = hop_fractions[opt_idx]

        plt.figure(figsize=(6, 6))
        plt.scatter(probabilities_cut, npasses, s=6)
        for k, (p, n, h) in enumerate(zip(probabilities_cut, npasses, hop_fractions)):
            if k%10 != 0:
                continue
            plt.text(p, n, f"{h:.2f}", fontsize=9, va='bottom', ha='left')

        # highlight the optimal point
        plt.scatter([opt_prob], [opt_pass], s=100, marker='*', edgecolor='k')
        plt.text(opt_prob, opt_pass, f"               choose hop={opt_hop:.2f}", fontsize=10, va='center')

        plt.xlabel("Probability of Cutting the Object")
        plt.ylabel("Inferences per Pixel")
        plt.title("Probability vs. Inferences")
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_meta, 'tilling_design_choice.pdf'))
        plt.show()


        print(f"Choose Hop Length: {opt_hop:.4f}")
        print(f"Choose Probability of Detection: {opt_prob:.4f}")
        print(f"Choose Number of Passes: {opt_pass:.4f}")

    def get_map_base(self, m):
        
        # Define CRS
        crs_4326 = CRS.from_epsg(4326)

        bounds = self.zones_to_map.to_crs(crs_4326).total_bounds  # (minx, miny, maxx, maxy)
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]]) 
        
        # Add France border to Folium map
        folium.GeoJson(
            self.zones_to_map.to_crs(crs_4326),
            name="Border Zone to Map",
            style_function=lambda feature: {
                "color": "black",
                "weight": 2,
                "fillOpacity": 0.1,
            },
            tooltip="Zone to Map"
        ).add_to(m)


        # Save to HTML file
        path_save = os.path.join(self.folder_meta, "tiles_visualization.html")
        m.save(path_save)

        print('map saved at :', path_save)



class TileSlidingGeneratorShapes(TileSlidingGenerator):

    # If you have acces to a cropp api over differente regions heach one with a differente cropp api
    # Crop API are supossed to be able to go outside of the bounds

    def __init__(self, *args, **kwargs):

        epsg = kwargs.get("epsg")
        crs = CRS.from_user_input(epsg)
        unit_name = crs.axis_info[0].unit_name.lower()
        if 'metre' not in unit_name and 'meter' not in unit_name:
            raise ValueError(f"This class only supports EPSG codes in meters, but got: {unit_name}")

        self.signed_distance_treshold_margine = kwargs.pop('signed_distance_treshold_margine', 0)


        super().__init__(*args, **kwargs)

    def construct_tiling(self):

        meta_meta_data = []
        dataset_len = 0

        for idx, zone_with_pixels in tqdm(enumerate(self.zones_with_pixels.itertuples())):
            min_x, min_y, max_x, max_y = zone_with_pixels.geometry.bounds  # (minx, miny, maxx, maxy)
            name = f"{int(min_x)}_{int(min_y)}_{int(max_x)}_{int(max_y)}"
            folder_name = os.path.join(self.folder_data, name)
            os.makedirs(folder_name, exist_ok=True)
            shp_path = os.path.join(folder_name, f"tiles_shape.shp")
            metadata_path = os.path.join(folder_name, f"tile_metadata.json")

            #geojson_path = os.path.join(folder_name, f"tiles.geojson")
            #gdf_tiles.to_file(geojson_path, driver="GeoJSON")
            
            tile_list = []

            intersecting_zones_with_pixels = [
                other.geometry #zone_with_pixels.geometry.intersection(other.geometry)
                for i, other in enumerate(self.zones_with_pixels.itertuples())
                if zone_with_pixels.geometry.intersects(other.geometry) and i != idx
            ]
            
            xs = np.arange(min_x, max_x, self.hop_size_metre)
            ys = np.arange(min_y, max_y, self.hop_size_metre)

            
            for x in tqdm(xs):
                for y in ys:
                    tile = box(x, y, x + self.tile_size_metre, y + self.tile_size_metre)

                    # .contains() for strict inclusion # intersects
                    if zone_with_pixels.geometry.intersects(tile) and self.zones_to_map.geometry.intersects(tile).any(): 
                        current_distance = signed_distance(tile, zone_with_pixels.geometry)
                        other_distances = [signed_distance(tile, other_poly) for other_poly in intersecting_zones_with_pixels]

                        if current_distance < min(other_distances, default=float('inf')) + self.signed_distance_treshold_margine :
                            tile_list.append({
                                "geometry": tile,
                            })
                            dataset_len += 1


            gdf_tiles = gpd.GeoDataFrame(tile_list, geometry="geometry",  crs=self.epsg)

            try:
                gdf_tiles.to_file(shp_path)
            except:
                gdf_tiles.to_file(shp_path, index=False)

            union_tiles = unary_union([tile["geometry"] for tile in tile_list])
            metadata = {
                "name": name,
                "file": os.path.join('data',name),
                "bounds": zone_with_pixels.geometry.bounds,
                "union_tiles":union_tiles.wkt, # union of all tile_results geometry??,  # 
            }
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)

            meta_meta_data.append(metadata)
        
        with open(os.path.join(self.folder_meta, "metadata.json"), "w") as f:
            json.dump(meta_meta_data, f, indent=4)
        
        self.len = dataset_len


    def get_map(self):

        # Define CRS
        crs_4326 = CRS.from_epsg(4326)  # Target CRS (WGS84)
        
        m = folium.Map(zoom_start=6)

        with open(os.path.join(self.folder_meta, "metadata.json"), "r") as f:
            meta_meta_data = json.load(f)


        # Loop through tiles and plot them in Folium
        for i, dico in enumerate(meta_meta_data):

            # Bound
            gdf_bounds = gpd.GeoSeries(densify_box(*dico["bounds"]), crs=self.epsg).to_crs(crs_4326)
            bounds_latlon = [(y, x) for x, y in gdf_bounds.geometry[0].exterior.coords]
            folium.GeoJson(gdf_bounds, color="red", weight=2, tooltip=f"Tile {i} Bounds").add_to(m)
            for latlon in bounds_latlon[:-1]:
                folium.CircleMarker(latlon, color="red", radius=3, fill=True, fill_color="red").add_to(m)

            # Compute tile center and transform to EPSG:4326
            polygon = box(*dico["bounds"])
            center = gpd.GeoSeries([polygon.centroid], crs=self.epsg).to_crs(crs_4326).geometry[0]
            folium.Marker(
                location=[center.y, center.x],
                icon=folium.DivIcon(html=f'<div style="font-size:12px;font-weight:bold;">{i}</div>'),
            ).add_to(m)
            

            union_geom = loads(dico["union_tiles"])  # Convert WKT back to Shapely geometry
            gdf_union = gpd.GeoSeries([union_geom], crs=self.epsg).to_crs(crs_4326)

            # Check if union is MultiPolygon or Polygon and extract coordinates properly
            union_shapes = []
            
            if isinstance(gdf_union.geometry[0], MultiPolygon):
                for poly in gdf_union.geometry[0].geoms:  # Iterate over each Polygon in MultiPolygon
                    union_shapes.append([(y, x) for x, y in poly.exterior.coords])
            elif isinstance(gdf_union.geometry[0], Polygon):
                union_shapes.append([(y, x) for x, y in gdf_union.geometry[0].exterior.coords])

            # Add union of all tiles in green with 30% transparency
            for shape in union_shapes:
                folium.Polygon(
                    locations=shape,
                    color="green",  # Border color
                    weight=3,  # Border thickness
                    fill=True,  # Enable fill color
                    fill_color="green",  # Inside fill color
                    fill_opacity=0.3,  # Transparency level (30%)
                    tooltip="Union of Tiles"
                ).add_to(m)


        super().get_map_base(m)


    def tile_iterator(self, dataset_cropper = None):
        
        resolution = dataset_cropper.resolution
        size = (self.tile_size_metre//resolution, self.tile_size_metre//resolution)

        metadata_path = os.path.join(self.folder_meta, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, "r") as f:
            zones_meta = json.load(f)

        for zone_meta in zones_meta:
            name = zone_meta["name"]
            zone_bounds = tuple(zone_meta["bounds"])
            # folder where you wrote that zone's shapefile
            zone_folder = os.path.join(self.folder_data, name)
            shp_path = os.path.join(zone_folder, "tiles_shape.shp")

            if not os.path.exists(shp_path):
                raise FileNotFoundError(f"Shapefile for zone '{name}' not found at {shp_path}")

            # read all tile polygons for this zone
            gdf = gpd.read_file(shp_path)

            for row in gdf.itertuples():

                if dataset_cropper is None:
                    yield {
                        "zone":     name,
                        "geometry": row.geometry
                    }
                else:
                    bbox = list(row.geometry.bounds)
                    print(bbox)
                    data = dataset_cropper.get_image_from_bbox(
                        bbox,
                        epsg_in = self.epsg, # WARNING CHANGE
                        size = size,
                        format_bbox_in = 'xyxy_list_single',
                        format_image_out = None,
                    )
                    yield {
                        "zone":     name,
                        "geometry": row.geometry,
                        "bbox_epsg_image": bbox,
                        "bbox_pix_image": size,
                        "image":    data['image']
                    }


    def run(
            self, 
            dataset_cropper, 
            model, 
            path_submission = '/home/adhemar/Bureau/METHAN/code/results/test_denmark_1',
            time_between_saves = 1*60*60,
            ):
        


        os.makedirs(path_submission, exist_ok=True)


        annotation_type = model.annotation_type
        annotation_format = model.annotation_format
        image_format = model.image_format
        #annotation_type = 'bbox'
        #annotation_format = 'xyxy_list_single'

        results = []
        number_saves = 0
        prev_time = time.time()

        for i, tile in enumerate(tqdm(self.tile_iterator(dataset_cropper), total = self.len)):

            image = convert_image(
                tile['image'],
                format_in = 'pil',
                format_out = image_format,
            )

            output = model([image])[0]

            annotations = output['annotations'] #    'boxes': Tensor of shape [N, k]
            scores = output['scores']       #    'labels': Tensor of shape [N]
            labels = output['classes']      #    'scores': Tensor of shape [N]

            # Filter out low score predictions (optional)
            #keep_idx = scores >= score_threshold
            #positions = positions[keep_idx]
            #scores = scores[keep_idx]
            #labels = labels[keep_idx]
            
            for annotation, score, label in zip(annotations, scores, labels):

                detection_segmentation = convert_anything_to_segmentation(
                    annotation,
                    object_in = annotation_type,
                    format_in = annotation_format,
                    format_out = 'pts_array_single'
                )

                detection_segmentation = convert_anything_with_lerp(
                    detection_segmentation,
                    bbox_pix_image = tile['bbox_pix_image'],
                    bbox_epsg_image = tile['bbox_epsg_image'],
                    object_in = 'segmentation',
                    format_in = 'pts_array_single',
                    format_out = 'flat_list_multi',
                    format_in_bbox_epsg_image = 'xyxy_list_single',
                )

                detection_bbox = convert_segmentation_to_bbox(
                    detection_segmentation,
                    format_in= 'flat_list_multi',
                    format_out= 'xywh_list_single'

                )

                result = {
                    'category_id': int(label),
                    'segmentation': detection_segmentation,
                    'bbox': detection_bbox, 
                    'score': float(score)
                }
                results.append(result)
        


            if time.time() - prev_time > time_between_saves and len(results) != 0:
                path_prediction = os.path.join(path_submission, f'predictions_{number_saves:04d}.json')
                with open(path_prediction, 'w') as f:
                    json.dump(results, f, indent=4)

                results = []
                number_saves += 1
                prev_time = time.time()

        if len(results) != 0:
            path_prediction = os.path.join(path_submission, f'predictions_{number_saves:04d}.json')
            with open(path_prediction, 'w') as f:
                json.dump(results, f, indent=4)       





class TileSlidingGeneratorPyramidSimple(TileSlidingGenerator): # TODO Test Test Test
    
    
    def __init__(self, *args, **kwargs):

        zoom_level = kwargs['tiling_config'].pop('zoom_level')
        tiles_per_image = kwargs['tiling_config'].pop('tiles_per_image')  # e.g. 3 means 3x3 tile
        hop_tile_count = kwargs['tiling_config'].pop('hop_tile_count')  # e.g. 1, 2, etc.
        tile_pixel_size = kwargs['tiling_config'].pop('tile_pixel_size') 
        

        resolution_m_per_px = 156543.03392 / (2 ** zoom_level)
        single_tile_m = resolution_m_per_px * tile_pixel_size

        # Compute final tile size and hop
        tile_size_metre = single_tile_m * tiles_per_image
        hop_size_metre = single_tile_m * hop_tile_count

        self.signed_distance_treshold_margine = kwargs.pop('signed_distance_treshold_margine', 0)

        tiling_config = kwargs.get("tiling_config", {})
        if 'tile_size_metre' in tiling_config:
            warnings.warn("Overwriting user-provided 'tile_size_metre' from pyramid parameters.")
        if 'hop_fraction' in tiling_config:
            warnings.warn("Overwriting user-provided 'hop_fraction' from pyramid parameters.")


        print(f"[TileSlidingGeneratorPyramidSimple] Computed values from pyramid:")
        print(f"  → tile_size_metre = {tile_size_metre:.2f} m")
        print(f"  → hop_size_metre  = {hop_size_metre:.2f} m (hop_fraction = {hop_size_metre / tile_size_metre:.2f})")
        print(f"  → resolution      = {resolution_m_per_px:.2f} m / pix")

        tiling_config['tile_size_metre'] = tile_size_metre
        tiling_config['hop_fraction'] = hop_size_metre / tile_size_metre
        kwargs['tiling_config'] = tiling_config


        self.zoom_level = zoom_level
        self.tiles_per_image = tiles_per_image
        self.hop_tile_count = hop_tile_count
        self.tile_pixel_size = tile_pixel_size
        self.single_tile_m = single_tile_m

        self.origin_shift = np.pi * 6378137

        super().__init__(*args, **kwargs)

    def construct_tiling(self):
        meta_meta_data = []

        z = self.zoom_level
        tile_m = self.single_tile_m
        block_size = self.tiles_per_image
        hop_px   = self.hop_tile_count

        # for each input zone
        for idx, zone in enumerate(self.zones_with_pixels.itertuples()):
            minx, miny, maxx, maxy = zone.geometry.bounds

            # convert zone bounds to tile indices
            tx_min = int((minx + self.origin_shift) / tile_m)
            tx_max = int((maxx + self.origin_shift) / tile_m)
            ty_min = int((self.origin_shift - maxy) / tile_m)
            ty_max = int((self.origin_shift - miny) / tile_m)

            tile_list = []

            # slide over blocks of tiles_per_image
            for tx0 in tqdm(range(tx_min, tx_max + 1, hop_px), desc="X Loop"):
                for ty0 in range(ty_min, ty_max + 1, hop_px):
                    # collect sub-tiles in this block
                    subtiles = [
                        [tx0, ty0, z],
                        [tx0 + block_size - 1, ty0, z],
                        [tx0, ty0 + block_size - 1, z],
                        [tx0 + block_size - 1, ty0 + block_size - 1, z]
                    ]
                    geom_parts = []
                    for i in range(tx0, tx0 + block_size):
                        for j in range(ty0, ty0 + block_size):
                            # compute this single tile’s bbox in EPSG:3857
                            bx_min = i    * tile_m - self.origin_shift
                            bx_max = (i+1)* tile_m - self.origin_shift
                            by_max = self.origin_shift - j    * tile_m
                            by_min = self.origin_shift - (j+1)* tile_m

                            geom_parts.append(box(bx_min, by_min, bx_max, by_max))

                    # merge the subtiles to one polygon
                    block_geom = unary_union(geom_parts)

                    # only keep if it actually intersects the zone
                    if zone.geometry.intersects(block_geom):
                        tile_list.append({
                            "geometry": block_geom,
                            "subtiles": subtiles
                        })

            # write out per-zone shapefile + metadata
            name = f"{idx}_{int(minx)}_{int(miny)}_{int(maxx)}_{int(maxy)}"
            folder = os.path.join(self.folder_data, name)
            os.makedirs(folder, exist_ok=True)

            # shapefile of tile blocks
            gdf = gpd.GeoDataFrame(tile_list, geometry="geometry", crs=self.epsg)
            gdf.to_file(os.path.join(folder, "tiles_shape.shp"))

            union_tiles = unary_union([tile["geometry"] for tile in tile_list])
            meta = {
                "name": name,
                "subtile_count": sum(len(t["subtiles"]) for t in tile_list),
                "file": os.path.join('data',name),
                "bounds": zone.geometry.bounds,
                "union_tiles":union_tiles.wkt,
            }
            with open(os.path.join(folder, "tile_metadata.json"), "w") as f:
                json.dump(meta, f, indent=2)

            meta_meta_data.append(meta)

        # global metadata
        with open(os.path.join(self.folder_meta, "metadata.json"), "w") as f:
            json.dump(meta_meta_data, f, indent=2)

    def tile_iterator(self):
        """
        Generator that yields tile metadata one by one from the metadata.json file.
        """
        metadata_path = os.path.join(self.folder_meta, "metadata.json")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, "r") as f:
            meta_meta_data = json.load(f)

        for tile_meta in meta_meta_data:

            zone_folder = os.path.join(self.folder_data, tile_meta['file'])
            shp_path = os.path.join(zone_folder, "tiles_shape.shp")
            gdf = gpd.read_file(shp_path)

            for _, row in gdf.iterrows():
                yield {
                    "zone":         tile_meta["name"],
                    "zone_bounds":  tile_meta["bounds"],
                    "geometry":     row["geometry"],
                    "subtiles":     row["subtiles"]
                }

    def get_map(self):

        # Define CRS
        crs_4326 = CRS.from_epsg(4326)  # Target CRS (WGS84)
        
        m = folium.Map(zoom_start=6)

        with open(os.path.join(self.folder_meta, "metadata.json"), "r") as f:
            meta_meta_data = json.load(f)


        # Loop through tiles and plot them in Folium
        for i, dico in enumerate(meta_meta_data):

            # Bound
            gdf_bounds = gpd.GeoSeries(densify_box(*dico["bounds"]), crs=self.epsg).to_crs(crs_4326)
            bounds_latlon = [(y, x) for x, y in gdf_bounds.geometry[0].exterior.coords]
            folium.GeoJson(gdf_bounds, color="red", weight=2, tooltip=f"Tile {i} Bounds").add_to(m)
            for latlon in bounds_latlon[:-1]:
                folium.CircleMarker(latlon, color="red", radius=3, fill=True, fill_color="red").add_to(m)

            # Compute tile center and transform to EPSG:4326
            polygon = box(*dico["bounds"])
            center = gpd.GeoSeries([polygon.centroid], crs=self.epsg).to_crs(crs_4326).geometry[0]
            folium.Marker(
                location=[center.y, center.x],
                icon=folium.DivIcon(html=f'<div style="font-size:12px;font-weight:bold;">{i}</div>'),
            ).add_to(m)
            

            union_geom = loads(dico["union_tiles"])  # Convert WKT back to Shapely geometry
            gdf_union = gpd.GeoSeries([union_geom], crs=self.epsg).to_crs(crs_4326)

            # Check if union is MultiPolygon or Polygon and extract coordinates properly
            union_shapes = []
            
            if isinstance(gdf_union.geometry[0], MultiPolygon):
                for poly in gdf_union.geometry[0].geoms:  # Iterate over each Polygon in MultiPolygon
                    union_shapes.append([(y, x) for x, y in poly.exterior.coords])
            elif isinstance(gdf_union.geometry[0], Polygon):
                union_shapes.append([(y, x) for x, y in gdf_union.geometry[0].exterior.coords])

            # Add union of all tiles in green with 30% transparency
            for shape in union_shapes:
                folium.Polygon(
                    locations=shape,
                    color="green",  # Border color
                    weight=3,  # Border thickness
                    fill=True,  # Enable fill color
                    fill_color="green",  # Inside fill color
                    fill_opacity=0.3,  # Transparency level (30%)
                    tooltip="Union of Tiles"
                ).add_to(m)


        super().get_map_base(m)
        

class TileSlidingGeneratorLocalSuperTiles(TileSlidingGenerator):

    def __init__(self, *args, tile_list, **kwargs):
        self.tile_list = tile_list

        raise NotImplementedError

        super().__init__(*args, **kwargs)
    

    def construct_tiling(self):
    
        tile_size = 1500# 750 # 1500
        hop_size = 1230 # 525 # 750
        tile_size_pix = 1000 # 500 # 1000
        hop_size_pix = 820 # 350 # 500



        tile_list_plus = []

        for dico_choose, dico in tqdm(enumerate(tile_list)):

            #try:
            if True:
                tile_results = []
                corners_polygon = Polygon(dico["corners"])
                bounds = dico["bound"]

                min_x, min_y, max_x, max_y = corners_polygon.bounds
                xs = np.arange(min_x, max_x, hop_size)
                ys = np.arange(min_y, max_y, hop_size)

                xs_pix = np.arange(0, dico["width"], hop_size_pix)
                ys_pix = np.arange(0, dico["height"], hop_size_pix)

                intersecting_tile_list = [
                    Polygon(other_dico["corners"]) for i, other_dico in enumerate(tile_list)
                    if Polygon(other_dico["corners"]).intersects(corners_polygon) and i != dico_choose
                ]

                for x, x_pix in zip(xs, xs_pix):
                    for y, y_pix in zip(ys, ys_pix):
                        tile = box(x, y, x + tile_size, y + tile_size)
                        tile_pix = box(x_pix, y_pix, x_pix + tile_size_pix, y_pix + tile_size_pix)

                        # .contains() for strict inclusion
                        if corners_polygon.intersects(tile) and france_gdf.intersects(tile).any(): 
                            current_distance = signed_distance(tile, corners_polygon)
                            other_distances = [signed_distance(tile, other_poly) for other_poly in intersecting_tile_list]

                            if current_distance < min(other_distances, default=float('inf')):
                                tile_results.append({
                                    "geometry": tile,
                                    "pixels": tile_pix,
                                })

                union_tiles = unary_union([tile["geometry"] for tile in tile_results])

                name = f"{int(bounds[0][0])}_{int(bounds[0][1])}"
                folder_name = os.path.join(path_save_metadata, name)
                os.makedirs(folder_name, exist_ok=True)
                geojson_path = os.path.join(folder_name, f"tiles.geojson")
                shp_path = os.path.join(folder_name, f"tiles_shape.shp")
                metadata_path = os.path.join(folder_name, f"tile_metadata.json")

                gdf_tiles = gpd.GeoDataFrame(tile_results, geometry="geometry",  crs="EPSG:2154")
                #gdf_tiles.to_file(geojson_path, driver="GeoJSON")
                gdf_tiles.to_file(shp_path)

                metadata = {
                    "name": name,
                    "file": dico["file"],
                    "corners": dico["corners"],
                    "bounds": dico["bound"],
                    "transform": dico["transform"],
                    "union_tiles":union_tiles.wkt, # union of all tile_results geometry??,  # 
                }
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=4)

                tile_list_plus.append(metadata)