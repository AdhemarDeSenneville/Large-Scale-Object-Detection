from .dataset_coverage import CoverageTileDataset, OneTileAtATimeBatchSampler
from .utils import segmentation_to_bbox
import json
import os
import numpy as np  
from tqdm import tqdm
import time
import torch


def segmentation_to_bbox(segmentation):
    xs = segmentation[0::2]  # Extract x-coordinates
    ys = segmentation[1::2]  # Extract y-coordinates
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    width = x_max - x_min
    height = y_max - y_min
    return [x_min, y_min, width, height]


def generate_annotation(model, dataset_folder_path, path_submission, annotation_type = 'obb', score_threshold = 0.05):

    #folder_path = "/home/adhemar/Bureau/datasets/France/metadata"
    dataset = CoverageTileDataset(dataset_folder_path, [190, 191]) # , [0,1] #, 205, 206
    batch_sampler = OneTileAtATimeBatchSampler(dataset, max_batch_size=16)
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler)

    results = []
    #annotation_type = "obb"
    #score_threshold = 0.05
    path_submission = "/home/adhemar/Bureau/METHAN/code/results/france_test_2_rgb"
    number_saves = 0

    os.makedirs(path_submission, exist_ok = True)

    start_time = time.time()

    i_break = 0
    for batch in tqdm(loader):
        i_break += 1
        if i_break == 20:
            pass
            #break

        end_time = time.time()  # End time

        #print(f"Bach LOADING Time: {end_time - start_time:.4f} seconds") 

        if batch['is_new'].any() and len(results) != 0:
            path_prediction = os.path.join(path_submission, f'predictions_{number_saves:04d}.json')
            with open(path_prediction, 'w') as f:
                json.dump(results, f, indent=4)

            results = []
            number_saves += 1

        with torch.no_grad():
            #print(img_pil.shape)
            outputs = model(batch['image']) 
        
        #set_trace()

        big_tile_idx = batch['big_tile_idx'][0]
        for transform, sub_tile_idx, output in zip(batch['transform'], batch['sub_tile_idx'], outputs):

            positions = output['positions'] #    'boxes': Tensor of shape [N, 4]
            scores = output['scores']       #    'labels': Tensor of shape [N]
            labels = output['classes']      #    'scores': Tensor of shape [N]

            # Filter out low score predictions (optional)
            keep_idx = scores >= score_threshold
            positions = positions[keep_idx]
            scores = scores[keep_idx]
            labels = labels[keep_idx]

            # Convert to format
            for box, score, label in zip(positions, scores, labels): # tqdm(

                if annotation_type == "bbox":
                    x1, y1, x2, y2 = box 

                    segmentation = [
                        float(x1), 
                        float(y1), 
                        float(x1), 
                        float(y2), 
                        float(x2), 
                        float(y2), 
                        float(x2), 
                        float(y1)
                    ]
                
                if annotation_type == "obb":
                    # For oriented bounding boxes (obb), calculate the vertices from the box.
                    x, y, width, height, angle = box
                    c, s = np.cos(angle), np.sin(angle)
                    #print(angle)
                    # Define the half-width and half-height
                    dx, dy = width / 2, height / 2
                    
                    # Compute the four corners relative to the box center
                    corners = np.array([
                        [-dx, -dy],
                        [dx, -dy],
                        [dx, dy],
                        [-dx, dy]
                    ])
                    
                    # Rotation matrix
                    rotation_matrix = np.array([[c, -s], [s, c]])
                    
                    # Rotate and shift the corners to the correct position
                    rotated_corners = (rotation_matrix @ corners.T).T + [x, y]
                    segmentation = rotated_corners.flatten().tolist()

                segmentation = np.array(segmentation).reshape(-1, 2)
                homogeneous_coords = np.hstack((segmentation, np.ones((segmentation.shape[0], 1))))
                segmentation = (transform @ homogeneous_coords.T).T[:, :2]
                segmentation = segmentation.flatten().tolist()
                

                result = {
                    "big_tile_idx": int(big_tile_idx),
                    "sub_tile_idx": int(sub_tile_idx),
                    'category_id': int(label),
                    'segmentation': [segmentation],
                    'bbox': segmentation_to_bbox(segmentation), 
                    'score': float(score)
                }
                results.append(result)
        

        start_time = time.time()  # Start time

    path_prediction = os.path.join(path_submission, f'predictions_{number_saves:04d}.json')
    with open(path_prediction, 'w') as f:
        json.dump(results, f, indent=4)

    results = []
    number_saves += 1