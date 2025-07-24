

from pyproj import Transformer

def add_positions(detections):

    transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)

    for detec in detections:
        bboxe = detec['bbox']
        x = bboxe[0] + (bboxe[2] / 2)
        y = bboxe[1] + (bboxe[3] / 2)

        lon, lat = transformer.transform(x, y)
        detec['position'] = (x, y)
        detec['position_gps'] = (lon, lat)
        detec['human_feedback'] = {}


class Detection:
    def __init__(self, big_tile_idx, sub_tile_idx, category_id, segmentation, bbox, score, inside_list=None):
        self.big_tile_idx = big_tile_idx
        self.sub_tile_idx = sub_tile_idx
        self.category_id = category_id
        self.segmentation = segmentation
        self.bbox = bbox
        self.score = score
        self.inside_list = [Detection(**inside) for inside in inside_list] if inside_list else []

    def get_position(self):
        # Return the centroid of the bounding box
        x_min, y_min, width, height = self.bbox
        x_center = x_min + width / 2
        y_center = y_min + height / 2
        return x_center, y_center

    def annotate_image(self, image, image_bbox):
        # Placeholder for annotation logic (e.g., drawing bbox on the image)
        pass

    def export_to_dict(self):
        # Convert object to dictionary format
        return {
            "big_tile_idx": self.big_tile_idx,
            "sub_tile_idx": self.sub_tile_idx,
            "category_id": self.category_id,
            "segmentation": self.segmentation,
            "bbox": self.bbox,
            "score": self.score,
            "inside_list": [inside.export_to_dict() for inside in self.inside_list]
        }

    def export_to_shp_row(self):
        # Placeholder for exporting to a shapefile row format
        pass