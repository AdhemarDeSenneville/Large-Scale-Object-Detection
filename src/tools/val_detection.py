import os
import json
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torchvision.ops as ops
from collections import defaultdict
from collections import Counter

from ..methode.utils_proba import detection_function
from ..data.utils_annotation import segmentation_to_bbox, pix_to_epsg
from .utils import count_categories, torch_nms_per_class


# -- -- -- -- -- -- -- -- -- #
#                            #
#     filter_annotations     #
#                            #
# -- -- -- -- -- -- -- -- -- #


def test_filter_annotations(
        results,
        threshold = 0.5,
        methode = 'proba_poisson',
        iou_threshold = {1: 0.3, 2: 0.1, 3: 0.5},
        pre_score_threshold = 0.05,
        pre_biodigestor_threshold = 0.2,
        verbose = False,
        skip = False,
    ):    

    count_categories(results, f'Befor pre threshold = {pre_score_threshold}', verbose)
    results = [res for res in results if res['score'] > pre_score_threshold]
    
    count_categories(results, f'Befor pre threshold biodigestor = {pre_biodigestor_threshold}', verbose)
    results = [res for res in results if (res['score'] > pre_biodigestor_threshold and res['category_id'] == 1) or res['category_id'] != 1]

    count_categories(results, f'Befor NMS IoU  = {iou_threshold}', verbose)
    annotations_by_image = defaultdict(list)
    for ann in results:
        annotations_by_image[ann['image_id']].append(ann)

    
    results_selected = defaultdict(list)
    for image_id, anns in annotations_by_image.items():
        results = torch_nms_per_class(anns, iou_threshold)

        count_categories(results, f'Befor Detection :', verbose)
        results_biodigester = [res for res in results if res['category_id'] == 1]
        results_rest = [res for res in results if res['category_id'] != 1]

        for biodigester in tqdm(results_biodigester, disable= ~verbose): # , verbose = verbose
            selected, total = detection_function(biodigester, results_rest, methode = methode)
            selected[0]['global_score'] = total
            if total >= threshold or skip:
                results_selected[image_id].append(selected)

        if verbose:
            print('Number of sites detected:', len(results_selected))
    
    return results_selected


# -- -- -- -- -- -- -- -- -- #
#                            #
#    generate_annotations    #
#                            #
# -- -- -- -- -- -- -- -- -- #


def test_generate_annotations(
        model, 
        dataset,
        path_submission,
        annotation_type="bbox", 
        score_threshold=0.05,
        bgr_to_rgb = True
    ):
    """
    Evaluate the model on the dataset (images in `image_dir`) with annotations in `ann_file`
    using pycocotools. Prints out mAP and AP per class.
    """

    # Create save directory if it does not exist
    if not os.path.exists(path_submission):
        os.makedirs(path_submission)

    results = []
    for img_id, item in tqdm(enumerate(dataset)):
        
        # Load image
        if bgr_to_rgb:
            img_pil = item["image"] # TODO
        else:
            img_pil = item["image"] # TODO

        # Inference
        with torch.no_grad():
            outputs = model(img_pil) 

        # Filter out low score predictions 
        positions = outputs['positions'] #    'boxes': Tensor of shape [N, 4]
        scores = outputs['scores']       #    'labels': Tensor of shape [N]
        labels = outputs['classes']      #    'scores': Tensor of shape [N]
        keep_idx = scores >= score_threshold
        positions = positions[keep_idx]
        scores = scores[keep_idx]
        labels = labels[keep_idx]

        # Convert to format
        for box, score, label in zip(positions, scores, labels): # tqdm(

            if annotation_type == "bbox":  # Json serializable
                x1, y1, x2, y2 = box 

                segmentation = [float(x1), float(y1), float(x1), float(y2), 
                    float(x2), float(y2), float(x2), float(y1)]
            
            if annotation_type == "obb":  # Json serializable
                # For oriented bounding boxes (obb)
                x, y, width, height, angle = box
                c, s = np.cos(angle), np.sin(angle)
                dx, dy = width / 2, height / 2
                
                # Compute the four corners relative to the box center
                corners = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])
                rotation_matrix = np.array([[c, -s], [s, c]])
                rotated_corners = (rotation_matrix @ corners.T).T + [x, y]

                segmentation = rotated_corners.flatten().tolist()


            segmentation = pix_to_epsg(segmentation, item['meta_data'])

            result = {
                'image_id': int(img_id),
                'category_id': int(label),
                'segmentation': [segmentation],
                'bbox': segmentation_to_bbox(segmentation), 
                'score': float(score)
            }
            results.append(result)


    if len(results) == 0:
        print("No detections were made; cannot compute mAP.")
        return

    path_prediction = os.path.join(path_submission, 'predictions.json')
    with open(path_prediction, 'w') as f:
        json.dump(results, f, indent=4)

    return results


def check(save_dir, id = 0, image_dir = None):

    # Load the JSON files
    path_ground_truth = os.path.join(save_dir, 'ground_truth.json')
    path_prediction = os.path.join(save_dir, 'predictions.json')
    with open(path_ground_truth, 'r') as f:
        gt_data = json.load(f)
    with open(path_prediction, 'r') as f:
        pred_data = json.load(f)

    # Select the first image_id
    first_image_id = gt_data['annotations'][id]['image_id']
    gt_segmentations = [(ann['segmentation'], ann['category_id']) for ann in gt_data['annotations'] if ann['image_id'] == first_image_id]
    pred_segmentations = [(res['segmentation'], res['category_id']) for res in pred_data if res['image_id'] == first_image_id]

    # Create the plot
    cmap = cm.get_cmap('tab10')
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
    plt.figure(figsize=(6, 6))

    if image_dir:
        image_name = next(img['file_name'] for img in gt_data['images'] if img['id'] == first_image_id)
        img_path = os.path.join(image_dir, image_name)
        img = plt.imread(img_path)
        plt.imshow(img)

    # Plot ground truth segmentations
    for segmentation, category_id in gt_segmentations:
        for poly in segmentation:
            x = poly[0::2]
            y = poly[1::2]
            min_x, min_y = min(min(x), min_x), min(min(y), min_y)
            max_x, max_y = max(max(x), max_x), max(max(y), max_y)
            #plt.fill(x, y, alpha=0.2, edgecolor='black', color=cmap(category_id % 10), linewidth=1, label=f'GT: Category {category_id}')
            plt.plot(x+ [x[0]], y + [y[0]], linestyle='-', color=cmap(category_id % 10), linewidth=1, label=f'Pred: Category {category_id}')

    # Plot predicted segmentations with dotted lines
    for segmentation, category_id in pred_segmentations:
        for poly in segmentation:
            x = poly[0::2]
            y = poly[1::2]
            min_x, min_y = min(min(x), min_x), min(min(y), min_y)
            max_x, max_y = max(max(x), max_x), max(max(y), max_y)
            plt.plot(x + [x[0]], y + [y[0]], linestyle='--', color=cmap(category_id % 10), linewidth=1, label=f'Pred: Category {category_id}')

    # Customize the plot
    plt.title(f'Segmentations for Image ID: {first_image_id}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().invert_yaxis() # Invert Y-axis for correct orientation
    #plt.legend(loc='upper right')
    plt.grid(False)
    plt.axis('equal')
    plt.axis([min_x - 10, max_x + 10, max_y + 10, min_y - 10])
    plt.show()