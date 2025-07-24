
from pyproj import Transformer 

from shapely.geometry import box, Point
from shapely.strtree import STRtree

import geopandas as gpd
from geopy.distance import distance 
import numpy as np
import ast

def get_scores_inside(all_new_detections, all_old_detections, filtered_position, polygone = None):
    true_positive_detections = [all_new_detections[detec_idx] for detec_idx in filtered_position['detected_known_new']]
    false_positive_detections = [all_new_detections[detec_idx] for detec_idx in filtered_position['detected_unknown_new']]
    false_negative_detections = [all_old_detections[detec_idx] for detec_idx in filtered_position['undetected_known_old']]

    if polygone:
        true_positive_detections = [detec for detec in true_positive_detections if polygone.contains(Point(detec['position']))]
        false_positive_detections = [detec for detec in false_positive_detections if polygone.contains(Point(detec['position']))]
        false_negative_detections = [detec for detec in false_negative_detections if polygone.contains(Point(detec['position']))]

    true_positive_scores = [detec['global_score'] for detec in true_positive_detections]
    false_positive_scores = [detec['global_score'] for detec in false_positive_detections]
    false_negative_scores = [detec['global_score'] for detec in false_negative_detections]
    

    thresholds = sorted(set(true_positive_scores + false_positive_scores), reverse=True)
    tp_counts = np.array([sum(score >= t for score in true_positive_scores) for t in thresholds])
    fp_counts = np.array([sum(score >= t for score in false_positive_scores) for t in thresholds])
    
    return {
        'true_positive_scores': true_positive_scores, 
        'false_positive_scores': false_positive_scores, 
        'false_negative_scores': false_negative_scores,
        'thresholds': thresholds,
        'tp_counts': tp_counts,
        'fp_counts': fp_counts,
        }


def load_points_from_geojson(points_geojson, filtering = False): # TODO better control
    """Load points from a GeoJSON file and return as a list of (longitude, latitude) tuples."""
    transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
    gdf = gpd.read_file(points_geojson)
    points_list = []
    all_detec_biodigesters = []

    for _, row in gdf.iterrows():
        x_center, y_center = row.geometry.x, row.geometry.y

        if filtering and ast.literal_eval(row['human_feedback'])['Human_Check_Spot_2023'] == 'True':
            
            points_list.append(Point(x_center, y_center))
            
            all_detec_biodigesters.append({
                'position': (x_center, y_center),
                'gps': transformer.transform(x_center, y_center),
                'global_score': 1
            })
        
    return all_detec_biodigesters, points_list

def load_points_from_results(all_detec_biodigesters):
    transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)

    points_list = []
    for ann in all_detec_biodigesters:
        bboxe = ann['bbox']
        x_center = bboxe[0] + (bboxe[2] / 2)
        y_center = bboxe[1] + (bboxe[3] / 2)
        points_list.append(Point(x_center, y_center))

        ann['position'] = (x_center, y_center)
        ann['gps'] = transformer.transform(x_center, y_center)

    return all_detec_biodigesters, points_list




def print_detections(filtered_position):

    for key, value in filtered_position.items():
        print(f'{key:<20} :',len(value))



def fast_detection_filtering(
        all_new_detections, # list of shapely.geometry points 
        all_old_detections, # list of shapely.geometry points 
        threshold = 100,
        verbose = False,
        ):
    
    filtered_position = {
        'detected_unknown_new': set(range(len(all_new_detections))),
        'undetected_known_old': set(range(len(all_old_detections))),
        'detected_known_new': set(),
        'detected_known_old': set(),
    }

    # STRtree
    tree = STRtree(all_new_detections)

    for idx_old, detec_old in enumerate(all_old_detections):
        
        idx_close_new_detecs = tree.query(detec_old.buffer(threshold))
        
        for idx_new in idx_close_new_detecs: # Here is the opptimisation select only close candidate
            
            dist = detec_old.distance(all_new_detections[idx_new])

            if dist <= threshold:
                filtered_position['detected_known_new'].add(idx_new)
                filtered_position['detected_known_old'].add(idx_old)

                if verbose: print(f"Sites {idx_new} and {idx_old} are within {threshold}m: Distance = {dist:.2f}m")

    filtered_position['undetected_known_old'] = filtered_position['undetected_known_old'] - filtered_position['detected_known_old']
    filtered_position['detected_unknown_new'] = filtered_position['detected_unknown_new'] - filtered_position['detected_known_new']

    if verbose: print_detections(filtered_position)
    
    return filtered_position



def get_gps_position(results, results_gt, threshold = 100, verbose = False):

    biodigestors_results = [res[0] for res in results]
    biodigestors_results_gt = [res[0] for res in results_gt]
    # Create EPSG transformer
    transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)

    for ann in biodigestors_results:
        bboxe = ann['bbox']
        x_center = bboxe[0] + (bboxe[2] / 2)
        y_center = bboxe[1] + (bboxe[3] / 2)
        ann['gps'] = transformer.transform(x_center, y_center)

    for ann in biodigestors_results_gt:
        bboxe = ann['bbox']
        x_center = bboxe[0] + (bboxe[2] / 2)
        y_center = bboxe[1] + (bboxe[3] / 2)
        ann['gps'] = transformer.transform(x_center, y_center)

    gps_position = {
        'undetected_known': set(),
        'detected_known': set(),
        'detected_unknown': set(),
        'detected_known_gt': set(),
    }

    distances = [] 
    for i in range(len(biodigestors_results)):

        for j in range(len(biodigestors_results_gt)):  # Avoid self-comparison
            coord = biodigestors_results[i]['gps']  # GPS coordinate (lat, lon)
            coord_gt = biodigestors_results_gt[j]['gps']

            # Compute distance in meters
            dist = distance(coord, coord_gt).m


            # Print if within threshold
            if dist <= threshold:

                if verbose:
                    print(f"Sites {i} and {j} are within {threshold}m: Distance = {dist:.2f}m")
                distances.append(dist)
                gps_position['detected_known'].add(i)
                gps_position['detected_known_gt'].add(j)
            else:
                gps_position['undetected_known'].add(j)
                gps_position['detected_unknown'].add(i)

    gps_position['undetected_known'] = gps_position['undetected_known'] - gps_position['detected_known_gt']
    gps_position['detected_unknown'] = gps_position['detected_unknown'] - gps_position['detected_known']
    
    

    #gps_position['detected_known_gt'] = [biodigestors_results_gt[i] for i in gps_position['detected_known_gt']]
    gps_position['detected_known'] = [results[i] for i in gps_position['detected_known']]
    gps_position['undetected_known'] = [results_gt[j] for j in gps_position['undetected_known']]
    gps_position['detected_unknown'] = [results[i] for i in gps_position['detected_unknown']]

    
    if verbose:
        print(f"{len(gps_position['detected_known_gt']) = }") # type: ignore
        print(f"{len(gps_position['detected_known']) = }") # type: ignore
        print(f"{len(gps_position['undetected_known']) = }") # type: ignore
        print(f"{len(gps_position['detected_unknown']) = }") # type: ignore
    
    return gps_position, distances
