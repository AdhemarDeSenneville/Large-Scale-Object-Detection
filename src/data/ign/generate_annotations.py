
import os
import json
import torch
import numpy as np
from tqdm import tqdm
import torch
from collections import Counter

from ..utils_annotation import pix_to_epsg, segmentation_to_bbox


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

            if annotation_type == "bbox":
                x1, y1, x2, y2 = box 

                segmentation = [
                    float(x1), float(y1), float(x1), float(y2), float(x2), float(y2), float(x2), float(y1)
                ]
            
            if annotation_type == "obb":
                # For oriented bounding boxes (obb), calculate the vertices from the box.
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
