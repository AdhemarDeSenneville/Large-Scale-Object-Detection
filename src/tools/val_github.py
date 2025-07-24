import argparse
import os
import json
import torch
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

from ..data.utils_annotation import *
from ..data.utils_image import *


#from pycocotools import mask as maskUtils
#from torchvision import transforms

metric_names = [
    'AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
    'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large'
]

def segmentation_to_bbox(segmentation):
    xs = segmentation[0::2]  # Extract x-coordinates
    ys = segmentation[1::2]  # Extract y-coordinates
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    width = x_max - x_min
    height = y_max - y_min
    return [x_min, y_min, width, height]

def segmentation_to_bbox_segmentation(segmentation):
    xs = segmentation[0][0::2]  # Extract x-coordinates
    ys = segmentation[0][1::2]  # Extract y-coordinates
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [[
        x_min, y_min,
        x_max, y_min,
        x_max, y_max,
        x_min, y_max,
    ]]

def segmentation_to_obb_segmentation(segmentation):
    # Convert segmentation to a NumPy array (Nx2 shape)
    points = np.array(segmentation, dtype=np.float32).reshape(-1, 2)

    # Get the minimum area rectangle
    rect = cv2.minAreaRect(points)
    (cx, cy), (w, h), angle = rect

    # Get the 4 corner points of the rotated bounding box
    box = cv2.boxPoints(rect)  # Returns 4 points
    box = np.intp(box)  # Convert to integer

    # (cx, cy, w, h, angle), 
    #print('OTHER',angle/180)

    return [box.flatten().tolist()]

def generate_results(
        model,
        path_to_outs,
        path_to_imgs, 
        path_to_json,
        path_to_vpv = None,
        score_threshold = 0,
    ):

    path_to_model_outputs = os.path.join(path_to_outs, 'model_outputs.json')

    if path_to_vpv == True:
        path_to_vpv = os.path.join(path_to_outs, 'vpv_outs')

    os.makedirs(path_to_outs, exist_ok=True)
    os.makedirs(path_to_vpv, exist_ok=True)

    annotation_type = model.annotation_type
    annotation_format = model.annotation_format
    image_format = model.image_format

    cocoGt = COCO(path_to_json)
    cat_ids = cocoGt.getCatIds()
    img_ids = cocoGt.getImgIds()
    model_outputs_list = []    

    for img_id in tqdm(img_ids):
        info = cocoGt.loadImgs([img_id])[0]
        file_name = info['file_name']
        img_path = os.path.join(path_to_imgs, file_name)

        image = convert_image(
            img_path,
            format_out = image_format,
        )

        output = model(image)

        annotations = output['positions'] #    'boxes': Tensor of shape [N, k]
        scores = output['scores']       #    'labels': Tensor of shape [N]
        labels = output['classes']      #    'scores': Tensor of shape [N]
        
        keep_idx = scores >= score_threshold
        annotations = annotations[keep_idx]
        scores = scores[keep_idx]
        labels = labels[keep_idx]

        vpv_segm_list = []
        vpv_cls_list = []

        for annotation, score, label in zip(annotations, scores, labels):

            detection_segmentation = convert_anything_to_segmentation(
                annotation,
                object_in = annotation_type,
                format_in = annotation_format,
                format_out = 'flat_list_multi'
            )

            detection_bbox = convert_segmentation_to_bbox(
                detection_segmentation,
                format_in= 'flat_list_multi',
                format_out= 'xywh_list_single'
            )

            result = {
                'image_id': int(img_id),
                'category_id': int(label), 
                'segmentation': detection_segmentation,
                'bbox': detection_bbox, 
                'score': float(score)
            }
            model_outputs_list.append(result)

            vpv_segm_list.append(detection_segmentation)
            vpv_cls_list.append(int(label))
        
        if path_to_vpv is not None:
            save_path_vpv = os.path.join(path_to_vpv, file_name[:-3] +'svg')
            write_svg(
                vpv_segm_list,
                vpv_cls_list,
                object_in = 'segmentation',
                format_in = 'flat_list_multi',
                save_path = save_path_vpv,
                overwrite = True,
            )

            
    with open(path_to_model_outputs, 'w') as f:
        json.dump(model_outputs_list, f, indent=4)
    
    return path_to_vpv, path_to_model_outputs

def eval_coco(
        path_to_metrics,
        path_to_json_gt,
        path_to_json_pred,
    ):
    os.makedirs(path_to_metrics, exist_ok=True)
    path_results = os.path.join(path_to_metrics, 'eval_coco.json')

    # Load the JSON file
    #with open(path_to_json_gt, 'r') as f:
    #    gt_data = json.load(f)
    
    #with open(path_to_json_pred, 'r') as f:
    #    pred_data = json.load(f)

    """

    keys_to_keep = ['id', 'image_id', 'category_id', 'segmentation', 'bbox', 'iscrowd', 'area']
    annotations = gt_data.get("annotations", [])
    for ann in annotations:
        keys_to_remove = [key for key in ann.keys() if key not in keys_to_keep]
        for key in keys_to_remove:
            ann.pop(key, None)
        #if annotation_type == 'obb':
        #    ann['segmentation'] = segmentation_to_obb_segmentation(ann['segmentation'])
        #if annotation_type == 'bbox':
        #    ann['segmentation'] = segmentation_to_bbox_segmentation(ann['segmentation'])

    keys_to_keep_images = ['id', 'file_name', 'width', 'height']
    images = gt_data.get("images", [])
    for img in images:
        keys_to_remove = [key for key in img.keys() if key not in keys_to_keep_images]
        for key in keys_to_remove:
            img.pop(key, None)
    gt_data['annotations'] = annotations
    gt_data['images'] = images


    # Step 2: Save the data as JSON files (in COCO format)
    with open(path_ground_truth, 'w') as f:
        json.dump(gt_data, f, indent=4)

    with open(path_prediction, 'w') as f:
        json.dump(results, f, indent=4)

    """# Update the JSON data


    # eval
    coco_gt = COCO(path_to_json_gt)
    coco_dt = coco_gt.loadRes(path_to_json_pred)
    
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')  # 'bbox' for bounding box evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


    np.save(os.path.join(path_to_metrics, "precision_iou.npy"), coco_eval.eval['precision'])


    all_results = {
        'overall': {metric: coco_eval.stats[idx] for idx, metric in enumerate(metric_names)},
        'per_class': {}
    }
    for category_id in coco_gt.getCatIds():
        category_name = coco_gt.loadCats(category_id)[0]['name']
        
        per_class_eval = COCOeval(coco_gt, coco_dt, iouType='segm')
        per_class_eval.params.catIds = [category_id]
        per_class_eval.evaluate()
        per_class_eval.accumulate()
        per_class_eval.summarize()
        
        # Extract precision and recall for PR values
        all_precision = coco_eval.eval['precision']
        all_results['per_class'][category_name] = {metric: per_class_eval.stats[idx] for idx, metric in enumerate(metric_names)}
        

    # Step 4: Save the per-class results to a JSON file
    with open(path_results, 'w') as f:
        json.dump(all_results, f, indent=4)

    return all_results

def val(
        model, 
        save_dir,
        image_dir = '/home/adhemar/Bureau/datasets/Methanizers/res_100cm/val/images', 
        ann_file = '/home/adhemar/Bureau/datasets/Methanizers/res_100cm/annotation/val.json',
        annotation_type="bbox", 
        score_threshold=0.05,
        bgr_to_rgb = True,  
        zoomed = False, 
        zoomed_bigger = False,
    ):
    """
    Evaluate the model on the dataset (images in `image_dir`) with annotations in `ann_file`
    using pycocotools. Prints out mAP and AP per class.
    """

    # Create save directory if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Load COCO ground truth
    cocoGt = COCO(ann_file)
    cat_ids = cocoGt.getCatIds()  # category IDs

    # Load all image IDs from the annotation file
    img_ids = cocoGt.getImgIds()

    #transform = transforms.Compose([
    #    transforms.ToTensor(),
    #])
    results = []

    for img_id in tqdm(img_ids):
        info = cocoGt.loadImgs([img_id])[0]
        file_name = info['file_name']
        img_path = os.path.join(image_dir, file_name)

        if not os.path.exists(img_path):
            # skip if the image doesn't exist
            continue

        # Load image
        if bgr_to_rgb:
            img_pil = Image.open(img_path).convert("RGB")
        else:
            img_pil = Image.open(img_path).convert("RGB")
            img_pil = np.array(img_pil)[:, :, ::-1]  # Convert RGB to BGR
        #img_tensor = #transform(img_pil).to(device)


        if zoomed:
            crop_border = 250
            img_pil = np.array(img_pil)[crop_border:-crop_border, crop_border:-crop_border]
        
        if zoomed_bigger:
            # double the size of the image don t crop, use bilinera interpolation
            img_pil = cv2.resize(np.array(img_pil), (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

        # Inference
        with torch.no_grad():
        # forward pass
            outputs = model(img_pil) 
            # outputs might contain:
            #    'boxes': Tensor of shape [N, 4]
            #    'labels': Tensor of shape [N]
            #    'scores': Tensor of shape [N]

        positions = outputs['positions']
        scores = outputs['scores']
        labels = outputs['classes']

        # Filter out low score predictions (optional)
        keep_idx = scores >= score_threshold
        positions = positions[keep_idx]
        scores = scores[keep_idx]
        labels = labels[keep_idx]

        # Convert to format
        for box, score, label in zip(positions, scores, labels): # tqdm(
            
            if annotation_type == "bbox":
                x1, y1, x2, y2 = box 

                if zoomed:
                    # Adjust the coordinates for the zoomed image
                    x1 += crop_border
                    y1 += crop_border
                    x2 += crop_border
                    y2 += crop_border

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

                
                if zoomed:
                    # Adjust the coordinates for the zoomed image
                    x += crop_border
                    y += crop_border
                
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

            result = {
                'image_id': int(img_id),
                'category_id': int(label),  # must match COCO category ID
                #'segmentation': maskUtils.frPyObjects([segmentation], 1000, 1000),
                'segmentation': [segmentation],
                'bbox': segmentation_to_bbox(segmentation), 
                'score': float(score)
            }
            results.append(result)

    # If no results found, just return
    if len(results) == 0:
        print("No detections were made; cannot compute mAP.")
        return
    



    path_ground_truth = os.path.join(save_dir, 'ground_truth.json')
    path_prediction = os.path.join(save_dir, 'predictions.json')
    path_results = os.path.join(save_dir, 'results.json')

    # Load the JSON file
    with open(ann_file, 'r') as f:
        gt_data = json.load(f)
    
    keys_to_keep = ['id', 'image_id', 'category_id', 'segmentation', 'bbox', 'iscrowd', 'area']
    annotations = gt_data.get("annotations", [])
    for ann in annotations:
        keys_to_remove = [key for key in ann.keys() if key not in keys_to_keep]
        for key in keys_to_remove:
            ann.pop(key, None)
        if annotation_type == 'obb':
            ann['segmentation'] = segmentation_to_obb_segmentation(ann['segmentation'])
        if annotation_type == 'bbox':
            ann['segmentation'] = segmentation_to_bbox_segmentation(ann['segmentation'])

    keys_to_keep_images = ['id', 'file_name', 'width', 'height']
    images = gt_data.get("images", [])
    for img in images:
        keys_to_remove = [key for key in img.keys() if key not in keys_to_keep_images]
        for key in keys_to_remove:
            img.pop(key, None)

    # Update the JSON data
    gt_data['annotations'] = annotations
    gt_data['images'] = images


    # Step 2: Save the data as JSON files (in COCO format)
    with open(path_ground_truth, 'w') as f:
        json.dump(gt_data, f, indent=4)

    with open(path_prediction, 'w') as f:
        json.dump(results, f, indent=4)


    # eval
    coco_gt = COCO(path_ground_truth)
    coco_dt = coco_gt.loadRes(path_prediction)
    
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')  # 'bbox' for bounding box evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


    np.save(os.path.join(save_dir, "precision_iou.npy"), coco_eval.eval['precision'])


    all_results = {
        'overall': {metric: coco_eval.stats[idx] for idx, metric in enumerate(metric_names)},
        'per_class': {}
    }
    for category_id in coco_gt.getCatIds():
        category_name = coco_gt.loadCats(category_id)[0]['name']
        
        per_class_eval = COCOeval(coco_gt, coco_dt, iouType='segm')
        per_class_eval.params.catIds = [category_id]
        per_class_eval.evaluate()
        per_class_eval.accumulate()
        per_class_eval.summarize()
        
        # Extract precision and recall for PR values
        all_precision = coco_eval.eval['precision']


        all_results['per_class'][category_name] = {metric: per_class_eval.stats[idx] for idx, metric in enumerate(metric_names)}
        

    # Step 4: Save the per-class results to a JSON file
    with open(path_results, 'w') as f:
        json.dump(all_results, f, indent=4)

    return all_results


def check(save_dir, id = 0, image_dir = None):

    path_ground_truth = os.path.join(save_dir, 'ground_truth.json')
    path_prediction = os.path.join(save_dir, 'predictions.json')

    with open(path_ground_truth, 'r') as f:
        gt_data = json.load(f)

    with open(path_prediction, 'r') as f:
        pred_data = json.load(f)

    # Select the first image_id
    first_image_id = id #gt_data['annotations'][id]['image_id']

    # Extract ground truth segmentations and their category IDs
    gt_segmentations = [(ann['segmentation'], ann['category_id']) for ann in gt_data['annotations'] if ann['image_id'] == first_image_id]

    # Extract predicted segmentations and their category IDs
    pred_segmentations = [(res['segmentation'], res['category_id']) for res in pred_data if res['image_id'] == first_image_id]

    # Set a color map to differentiate categories
    cmap = cm.get_cmap('tab10')
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')

    # Create the plot
    plt.figure(figsize=(6, 6))

    if image_dir:
        image_name = next(img['file_name'] for img in gt_data['images'] if img['id'] == first_image_id)
        img_path = os.path.join(image_dir, image_name)
        img = plt.imread(img_path)
        plt.imshow(img)

    # Plot ground truth segmentations
    for segmentation, category_id in gt_segmentations:
        for poly in segmentation:
            x = poly[0::2]  # x-coordinates
            y = poly[1::2]  # y-coordinates
            min_x, min_y = min(min(x), min_x), min(min(y), min_y)
            max_x, max_y = max(max(x), max_x), max(max(y), max_y)
            #plt.fill(x, y, alpha=0.2, edgecolor='black', color=cmap(category_id % 10), linewidth=1, label=f'GT: Category {category_id}')
            plt.plot(x+ [x[0]], y + [y[0]], linestyle='-', color=cmap(category_id % 10), linewidth=1, label=f'Pred: Category {category_id}')

    # Plot predicted segmentations with dotted lines
    for segmentation, category_id in pred_segmentations:
        for poly in segmentation:
            x = poly[0::2]  # x-coordinates
            y = poly[1::2]  # y-coordinates
            min_x, min_y = min(min(x), min_x), min(min(y), min_y)
            max_x, max_y = max(max(x), max_x), max(max(y), max_y)
            plt.plot(x + [x[0]], y + [y[0]], linestyle='--', color=cmap(category_id % 10), linewidth=1, label=f'Pred: Category {category_id}')

    # Customize the plot
    plt.title(f'Segmentations for Image ID: {first_image_id}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().invert_yaxis()  # Invert Y-axis for correct orientation
    #plt.legend(loc='upper right')
    plt.grid(False)  # Disable grid
    plt.axis('equal')
    plt.axis([min_x - 10, max_x + 10, max_y + 10, min_y - 10])
    plt.show()