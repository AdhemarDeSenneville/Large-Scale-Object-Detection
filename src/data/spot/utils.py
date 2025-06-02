import numpy as np
import cv2
from shapely.geometry import Point, Polygon
import random

def process_image_data(image_data, out_bgr = True):

    image = image_data[:3] / 1024
    image = np.clip(np.power(image, 1/1.2),0,1)
    image = (image * 255).astype(np.uint8)
    image = image.transpose(1, 2, 0)

    if out_bgr:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image


def draw_bounding_boxes(image_original, annotations):
    
    image = image_original.copy()
    for annotation in annotations:
        bbox = annotation['bbox']
        obb = annotation['obbs']['obb_segm']
        cv2.rectangle(
            image, 
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (0, 0, 255), 
            1
        )
        
        obb_pts = np.array(obb, dtype=np.int32).reshape(4, 2)
        cv2.polylines(image, [obb_pts], isClosed=True, color=(0, 255, 0), thickness=1) 
    return image


def get_random_points_in_polygon(polygon, n_points, seed=42):
    """
    Generate `n_points` random points within a (multi-)polygon using rejection sampling.
    
    :param polygon:   A Shapely (Multi)Polygon within which we want random points
    :param n_points:  Number of random points to generate
    :param seed:      Random seed for reproducibility
    :return:          List of Shapely Point objects
    """
    random.seed(seed)
    points = []
    
    # Get bounding box of the polygon
    minx, miny, maxx, maxy = polygon.bounds
    
    # Keep sampling until we have enough points
    while len(points) < n_points:
        print(len(points), end='\r')
        # Generate a random point within the bounding box
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        candidate = Point(x, y)
        
        # Check if the candidate point is actually within France
        if candidate.within(polygon):
            points.append(candidate)
    return points


def signed_distance(point: Point, reference_polygon: Polygon) -> float:
    dist = point.distance(reference_polygon.boundary)
    return -dist if reference_polygon.contains(point) else dist


def convert_to_dota_format(annotations):
    """
    Convert COCO-style annotation to DOTA format.

    Args:
        annotation (dict): COCO annotation with 'segmentation' or 'bbox'.
        category_mapping (dict): Mapping from category_id to category name.

    Returns:
        str: Annotation in DOTA format.
    """
    final_file = ''
    for annotation in annotations:
        category_id_to_name = {
            1: 'all',
            2: 'tank',
            3: 'pile'
        }

        # Extract COCO annotation data
        category_id = annotation["category_id"]
        category_name = category_id_to_name[category_id]
        difficult = 0
        obb = annotation['obbs']['obb_segm']

        box = [obb[0], obb[1], obb[2], obb[3], obb[4], obb[5], obb[6], obb[7]]

        # Convert to DOTA format string
        dota_label = " ".join(map(str, box)) + f" {category_name} {difficult}\n"
        final_file += dota_label
    return final_file

