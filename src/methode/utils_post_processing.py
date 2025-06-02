import torch
import torchvision.ops as ops
from collections import defaultdict
from collections import Counter
from tqdm import tqdm
from lsnms import nms, wbc

from shapely.geometry import box, Point
from shapely.strtree import STRtree


from .utils_proba import get_score

def make_shapely_box(b):
    x, y, w, h = b
    return box(x, y, x + w, y + h)

def count_categories(results, title, verbose):
    category_counts = Counter(r['category_id'] for r in results)
    if verbose:
        print(title, {cat: category_counts.get(cat, 0) for cat in [1, 2, 3]})

def apply_nms_per_class(results, iou_threshold):
    if len(results) == 0:
        return []

    # Organize boxes by class
    classwise_results = {1:[], 2:[], 3:[]}
    for res in results:
        classwise_results[res['category_id']].append(res)

    filtered_results = []

    for boxes, thresh in zip(classwise_results.values(), iou_threshold.values()):
        print('NMS on Boxes:', len(boxes))
        if len(boxes) == 0:
            continue

        # Convert to tensor
        box_tensors = torch.tensor([r['bbox'] for r in boxes], dtype=torch.float32)
        scores = torch.tensor([r['score'] for r in boxes], dtype=torch.float32)

        # Convert [x_min, y_min, width, height] -> [x1, y1, x2, y2]
        box_tensors[:, 2] += box_tensors[:, 0]
        box_tensors[:, 3] += box_tensors[:, 1]

        # Apply NMS
        #keep_indices = ops.nms(box_tensors, scores, thresh).numpy() Slow NMS
        keep_indices = nms(box_tensors, scores, thresh)

        # Append kept results
        filtered_results.extend([boxes[i] for i in keep_indices])

    return filtered_results

def apply_part_based_score(results, threshold, methode, skip, verbose):


    # 1) Keep track of an ID -> index mapping so we can attach each result to the correct biodigester
    #  We can just rely on the index order in results_biodigester and biodigester_polygons
    all_detec_biodigesters = [res for res in results if res['category_id'] == 1]
    for detec in all_detec_biodigesters:
        detec['inside_list'] = []

    all_detec_subparts     = [res for res in results if res['category_id'] != 1]
    biodigester_polygons = [make_shapely_box(bd['bbox']) for bd in all_detec_biodigesters]
    tree = STRtree(biodigester_polygons)
    poly_index_to_biodigester = dict(enumerate(all_detec_biodigesters))

    # 2) For each "rest" detection, find which biodigester(s) contain its center
    if verbose: print('Linking Parts with Boi-digesters...')
    for det in tqdm(all_detec_subparts):
        cx = det['bbox'][0] + det['bbox'][2] / 2
        cy = det['bbox'][1] + det['bbox'][3] / 2
        center_pt = Point(cx, cy)
        candidate_polygons = tree.query(center_pt)
        
        for idx in candidate_polygons:
            # Access the actual Shapely polygon
            poly = biodigester_polygons[idx]
            if poly.contains(center_pt):
                # 4c) The index of 'poly' in biodigester_polygons 
                biodi = poly_index_to_biodigester[idx]
                biodi['inside_list'].append(det)
    
    if verbose: print('Computing Scores...')
    results_selected = []
    totat_total_score = 0
    for bd in tqdm(all_detec_biodigesters):

        tank_confidence = [detec['score'] for detec in bd['inside_list'] if detec['category_id'] == 2]
        pile_confidence = [detec['score'] for detec in bd['inside_list'] if detec['category_id'] == 3]
        bd_confidence = bd['score']

        score = get_score(bd_confidence, tank_confidence, pile_confidence, methode)
        bd['global_score'] = score

        totat_total_score += score

        if score >= threshold or skip:
            results_selected.append(bd)
    
    return results_selected


def post_process_results(
        results,
        threshold = 0.1,
        methode = 'proba_histogram',
        iou_threshold = {1: 0.3, 2: 0.1, 3: 0.5},
        pre_score_threshold = 0.05,
        pre_biodigestor_threshold = 0.05,
        verbose = True,
        skip = False,
    ):


    count_categories(results, f'Befor Pre Processing ', verbose)
    
    if verbose: print('Apply pre thresholding...')
    results = [res for res in results if res['score'] > pre_score_threshold]
    count_categories(results, f'After pre threshold = {pre_score_threshold}', verbose)
    
    if verbose: print('Apply biodigestor pre thresholding...')
    results = [res for res in results if (res['score'] > pre_biodigestor_threshold and res['category_id'] == 1) or res['category_id'] != 1]
    count_categories(results, f'After pre biodigestor threshold = {pre_biodigestor_threshold}', verbose)

    if verbose: print('Apply NMS...')
    results = apply_nms_per_class(results, iou_threshold)
    count_categories(results, f'Afted NMS IoU  = {iou_threshold}', verbose)

    if verbose: print('Apply part based scoring...')
    results_selected = apply_part_based_score(results, threshold, methode, skip, verbose)
    if verbose: print('Number of Detections:',len(results_selected))

    return results_selected