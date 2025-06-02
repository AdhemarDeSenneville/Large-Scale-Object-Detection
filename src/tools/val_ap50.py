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
from ..data.utils_annotation import segmentation_to_bbox, segmentation_to_obb_segmentation, segmentation_to_bbox_segmentation

metric_names = [
    'AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
    'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large'
]


def val(
        model, 
        save_dir,
        image_dir = '/home/adhemar/Bureau/datasets/Methanizers/res_100cm/val/images', 
        ann_file = '/home/adhemar/Bureau/datasets/Methanizers/res_100cm/annotation/val.json',
        annotation_type="bbox", 
        score_threshold=0.05,
        bgr_to_rgb = True
    ):
    """
    Evaluate the model on the val dataset (images in `image_dir`) with annotations in `ann_file`
    using pycocotools. Prints out mAP and AP per class.
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load COCO ground truth
    cocoGt = COCO(ann_file)
    cat_ids = cocoGt.getCatIds()
    img_ids = cocoGt.getImgIds()

    results = []
    for img_id in tqdm(img_ids):
        info = cocoGt.loadImgs([img_id])[0]
        file_name = info['file_name']
        img_path = os.path.join(image_dir, file_name)

        if not os.path.exists(img_path): # skip if the image doesn't exist
            continue

        # Load image
        if bgr_to_rgb:
            img_pil = Image.open(img_path).convert("RGB")
        else:
            img_pil = Image.open(img_path).convert("RGB")
            img_pil = np.array(img_pil)[:, :, ::-1]  # Convert RGB to BGR
        
        with torch.no_grad():
            outputs = model(img_pil) 
            # outputs contain:
            #    'boxes': Tensor of shape [N, 4]
            #    'labels': Tensor of shape [N]
            #    'scores': Tensor of shape [N]

        # Filter out low score predictions
        positions = outputs['positions']
        scores = outputs['scores']
        labels = outputs['classes']
        keep_idx = scores >= score_threshold
        positions = positions[keep_idx]
        scores = scores[keep_idx]
        labels = labels[keep_idx]

        # Convert to format
        for box, score, label in zip(positions, scores, labels): # tqdm(
            
            if annotation_type == "bbox": # Json serializable
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
            
            if annotation_type == "obb":  # Json serializable
                
                x, y, width, height, angle = box
                c, s = np.cos(angle), np.sin(angle)
                dx, dy = width / 2, height / 2
                
                corners = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])
                rotation_matrix = np.array([[c, -s], [s, c]])
                rotated_corners = (rotation_matrix @ corners.T).T + [x, y]

                segmentation = rotated_corners.flatten().tolist()

            result = { # Json serializable
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
    
    path_ground_truth = os.path.join(save_dir, 'ground_truth.json')
    path_prediction = os.path.join(save_dir, 'predictions.json')
    path_results = os.path.join(save_dir, 'results.json')

    # Load the JSON file
    with open(ann_file, 'r') as f:
        gt_data = json.load(f)
    
    # Step 1: Remove unnecessary keys from the ground truth data
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

    gt_data['annotations'] = annotations
    gt_data['images'] = images

    # Step 2: Save the data as JSON files (in COCO format)
    with open(path_ground_truth, 'w') as f:
        json.dump(gt_data, f, indent=4)
    with open(path_prediction, 'w') as f:
        json.dump(results, f, indent=4)

    # Step 3: Eval
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
        all_precision = coco_eval.eval['precision']
        all_results['per_class'][category_name] = {metric: per_class_eval.stats[idx] for idx, metric in enumerate(metric_names)}
        

    # Step 4: Save the per-class results to a JSON file
    with open(path_results, 'w') as f:
        json.dump(all_results, f, indent=4)
    return all_results


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

    # Set a color map to differentiate categories
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
    plt.gca().invert_yaxis()
    plt.grid(False)
    plt.axis('equal')
    plt.axis([min_x - 10, max_x + 10, max_y + 10, min_y - 10])
    plt.show()