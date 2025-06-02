import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass

def segmentation_to_mask(segmentation, image_shape):
    # Convert flat segmentation list into polygon coordinates
    x = segmentation[0::2]
    y = segmentation[1::2]
    polygon = np.array(list(zip(x, y)), dtype=np.int32)

    # Create binary mask from polygon
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], color=1)
    return mask


def compute_overlap(mask_object, mask_place):
    # Intersection and union
    intersection = np.logical_and(mask_object, mask_place).sum()
    total = mask_object.sum()
    
    if total == 0:
        return 0.0 
    
    return intersection / total


def crop_masked_region(
        mask, 
        image, 
        padding=10,
        min_size=None,
        max_size=None
    ):
    x, y, w, h = mask['bbox']

    # Initial crop region with padding
    x_min = max(x - padding, 0)
    y_min = max(y - padding, 0)
    x_max = min(x + w + padding, image.shape[1])
    y_max = min(y + h + padding, image.shape[0])

    crop_w = x_max - x_min
    crop_h = y_max - y_min

    # Adjust crop to satisfy min_size
    if min_size is not None:
        if crop_w < min_size:
            extra = min_size - crop_w
            x_min = max(x_min - extra // 2, 0)
            x_max = min(x_max + (extra - extra // 2), image.shape[1])
        if crop_h < min_size:
            extra = min_size - crop_h
            y_min = max(y_min - extra // 2, 0)
            y_max = min(y_max + (extra - extra // 2), image.shape[0])

    # Adjust crop to satisfy max_size
    if max_size is not None:
        if (x_max - x_min) > max_size:
            center_x = (x_min + x_max) // 2
            x_min = max(center_x - max_size // 2, 0)
            x_max = x_min + max_size
            if x_max > image.shape[1]:
                x_max = image.shape[1]
                x_min = x_max - max_size
        if (y_max - y_min) > max_size:
            center_y = (y_min + y_max) // 2
            y_min = max(center_y - max_size // 2, 0)
            y_max = y_min + max_size
            if y_max > image.shape[0]:
                y_max = image.shape[0]
                y_min = y_max - max_size

    cropped_image = image[y_min:y_max, x_min:x_max]
    cropped_mask = mask['segmentation'][y_min:y_max, x_min:x_max]

    return cropped_image, cropped_mask


def show_masks(
        image, 
        masks,
        colors = None,
        labels = None,
        colors_contour = None,
        verbose = True, 
        save_path = None, 
        mask_alpha = 0.5,
        show_contour = False
):
    # Create an RGB copy of the image
    img = image.copy()
    if img.ndim == 2 or img.shape[2] == 1:
        img = np.stack([img]*3, axis=-1)

    # Create an overlay image
    overlay = np.zeros_like(img, dtype=np.uint8)

    if colors is None:
        colors = [np.array([random.randint(0, 255) for _ in range(3)], dtype=np.uint8) for __ in range(len(masks))]
    
    if colors_contour is None:
        colors_contour = ['red'] * len(masks)

    for idx, mask in enumerate(masks):
        m = mask['segmentation']
        color = colors[idx]
        overlay[m] = overlay[m] * (1-mask_alpha) + color * mask_alpha

    # Blend original image with overlay
    blended = img.copy()
    alpha = 0.5
    mask_area = overlay.sum(axis=2) > 0
    blended[mask_area] = (1 - alpha) * img[mask_area] + alpha * overlay[mask_area]

    plt.figure(figsize=(10, 10))
    plt.imshow(blended.astype(np.uint8))
    if show_contour:
        for idx, mask in enumerate(masks):
            plt.contour(mask['segmentation'], colors=colors_contour[idx], linewidths=0.5)
    
    if labels is not None:
        for idx, mask in enumerate(masks):
            m = mask['segmentation']
            if np.any(m):
                y, x = center_of_mass(m)  # Note: returns float coordinates
                plt.text(
                    x, y, labels[idx],
                    color='white', fontsize=12,
                    ha='center', va='center',
                    bbox=dict(facecolor='black', alpha=0.5, pad=1)
                )
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    if verbose:
        plt.show()
    else:
        plt.close()

def show_masks_2(
        image, 
        image_masks,
        verbose=True, 
        save_path=None, 
        mask_alpha=0.5,
):
    img = image.copy()
    if img.ndim == 2 or img.shape[2] == 1:
        img = np.stack([img]*3, axis=-1)

    # Create overlay
    overlay = np.zeros_like(img, dtype=np.float32)
    unique_ids = np.unique(image_masks)
    unique_ids = unique_ids[unique_ids != 0]  # Skip background (id = 0)

    for uid in unique_ids:
        color = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.uint8)
        mask = image_masks == uid
        for c in range(3):
            overlay[:, :, c][mask] = color[c]

    # Blend
    blended = img.astype(np.float32) * (1 - mask_alpha) + overlay * mask_alpha
    blended = blended.astype(np.uint8)

    # Show / Save
    plt.figure(figsize=(10, 10))
    plt.imshow(blended)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    if verbose:
        plt.show()
    else:
        plt.close()