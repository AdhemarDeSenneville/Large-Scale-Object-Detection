from collections import Counter
import torch
import torchvision.ops as ops

def count_categories(results, title, verbose):
    category_counts = Counter(r['category_id'] for r in results)
    if verbose:
        print(title, {cat: category_counts.get(cat, 0) for cat in [1, 2, 3]})


def is_center_inside(bbox_all, bbox_check):
    """ Check if the center (x, y) is inside the bbox_all (x1, y1, x2, y2). """

    cx = bbox_check[0] + bbox_check[2] / 2
    cy = bbox_check[1] + bbox_check[3] / 2

    x, y, w, h = bbox_all
    return x <= cx <= x + w and y <= cy <= y + h


def torch_nms_per_class(results, iou_threshold):
    if len(results) == 0:
        return []

    # Organize boxes by class
    classwise_results = {1:[], 2:[], 3:[]}
    for res in results:
        classwise_results[res['category_id']].append(res)

    filtered_results = []

    for boxes, thresh in zip(classwise_results.values(), iou_threshold.values()):
        if len(boxes) == 0:
            continue

        # Convert to tensor
        box_tensors = torch.tensor([r['bbox'] for r in boxes], dtype=torch.float32)
        scores = torch.tensor([r['score'] for r in boxes], dtype=torch.float32)

        # Convert [x_min, y_min, width, height] -> [x1, y1, x2, y2]
        box_tensors[:, 2] += box_tensors[:, 0]
        box_tensors[:, 3] += box_tensors[:, 1]

        # Apply NMS
        keep_indices = ops.nms(box_tensors, scores, thresh).numpy()

        # Append kept results
        filtered_results.extend([boxes[i] for i in keep_indices])

    return filtered_results