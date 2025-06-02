import os
import cv2
import numpy as np
from typing import Union, List, Tuple, Optional, Callable
from typing import Literal #, TypeAlias

import warnings

import pdb

# -- -- -- -- -- -- -- -- -- #
#                            #
#  change annotation format  #
#                            #
# -- -- -- -- -- -- -- -- -- #


SegmentationFormat = Literal[ # : TypeAlias
    'pts_array_single', 
    'pts_array_multi', 
    'pts_list_single', 
    'pts_list_multi',
    'flat_array_single', 
    'flat_array_multi', 
    'flat_list_single', 
    'flat_list_multi'
] 

BoxFormat = Literal[ # : TypeAlias
    'xyxy_array_single',
    'xyxy_array_multi',
    'xyxy_list_single',
    'xyxy_list_multi',
    'xywh_array_single',
    'xywh_array_multi',
    'xywh_list_single',
    'xywh_list_multi',
]

OrientedBoxFormat = Literal[ # : TypeAlias
    'cxcywha_array_single',
    'cxcywha_array_multi',
    'cxcywha_list_single',
    'cxcywha_list_multi',
    'pts_array_single',
    'pts_array_multi',
    'pts_list_single',
    'pts_list_multi',
    'flat_array_single',
    'flat_array_multi',
    'flat_list_single',
    'flat_list_multi'
]

PointFormat = Literal[ # : TypeAlias
    'array_single',
    'array_multi',
    'list_single',
    'list_multi'
]

AnyObject = Literal[
    'segmentation', 
    'bbox', 
    'obb', 
    'point'
]

AnyFormat = Union[
    SegmentationFormat, 
    BoxFormat, 
    OrientedBoxFormat, 
    PointFormat
]

def convert_segmentation(
    segmentation: Union[List[float], List[List[float]], np.ndarray, List[np.ndarray]],
    format_in: SegmentationFormat = 'pts_list_single',
    format_out: SegmentationFormat = 'pts_list_single'
) -> Union[List[float], List[List[float]], np.ndarray, List[np.ndarray]]:
    """
    Convert segmentation representations between different formats:
    - flat: flat list of coordinates (e.g., [x1, y1, x2, y2, ...])
    - pts: nested list of (x, y) pairs (e.g., [[x1, y1], [x2, y2], ...])
    - array: numpy array version of the above
    - list: Python list version of the above
    - single: a single polygon
    - multi: a list of polygons

    Args:
        segmentation: Input segmentation to be converted. Can be a flat list, nested list, or numpy array.
        format_in: Format of the input segmentation. Must be in the form '<pts|list>_<array|list>_<single|multi>'.
        format_out: Desired output format. Must be in the form '<pts|list>_<array|list>_<single|multi>'.

    Returns:
        Converted segmentation in the desired format, either as a single polygon or a list of polygons,
        and either in list or array form, flat or nested.
    """
    if format_in == format_out:
        return segmentation


    if format_out is None:
        format_out = format_in
    
    in_form, in_type, in_mult = format_in.split('_')
    out_form, out_type, out_mult = format_out.split('_')

    # Check input type
    assert in_form in ['pts', 'flat'], "Invalid input format. Choose 'pts' or 'flat'."
    assert in_type in ['array', 'list'], "Invalid input type. Choose 'array' or 'list'."
    assert in_mult in ['single', 'multi'], "Invalid input multiplicity. Choose 'single' or 'multi'."
    # Check output type
    assert out_form in ['pts', 'flat'], "Invalid output format. Choose 'pts' or 'flat'."
    assert out_type in ['array', 'list'], "Invalid output type. Choose 'array' or 'list'."
    assert out_mult in ['single', 'multi'], "Invalid output multiplicity. Choose 'single' or 'multi'."

    # Prepare list of polygons based on multiplicity
    if in_mult == 'single':
        segmentation = [segmentation]
    
    
    segmentation_new = []
    for poly in segmentation:
        poly = np.array(poly)
        if in_form == 'flat':
            poly = poly.reshape(-1, 2)
        
        segmentation_new.append(poly)
    
    segmentation = segmentation_new
    # Build output representations
    segmentation_new = []
    for poly in segmentation:
        
        if out_form == 'flat':
            poly = poly.flatten()
        
        if out_type == 'list':
            poly = poly.tolist()
                
        segmentation_new.append(poly)
    # Return single or multi
    if out_mult == 'single':
        return segmentation_new[0]
    elif out_mult == 'multi':
        return segmentation_new


def convert_bbox(
    bbox: Union[List[float], List[List[float]], np.ndarray],
    format_in: BoxFormat = 'xyxy_array_single',
    format_out: BoxFormat = 'xyxy_array_single'
) -> Union[np.ndarray, List[float], List[List[float]]]:
    """
    Convert bounding boxes between 'xyxy' (top-left and bottom-right) and 'xywh' (top-left, width, height)
    formats, with support for both single and multiple bounding boxes, and both array and list outputs.

    Args:
        bbox: Bounding box or list of bounding boxes to convert. Can be a flat list, nested list, or numpy array.
              Example of 'xyxy' single: [x1, y1, x2, y2]
              Example of 'xywh' multi: [[x1, y1, w1, h1], [x2, y2, w2, h2]]
        format_in: Input format, must follow '<xyxy|xywh>_<array|list>_<single|multi>'.
        format_out: Desired output format, same structure as `format_in`.

    Returns:
        Converted bounding box or list of bounding boxes, in the specified output format.
    """  
    if format_in == format_out:
        return bbox


    out_form, out_type, out_dim = format_out.split('_')
    in_form, in_type, in_dim = format_in.split('_')
    
    # Check input type
    assert in_form in ['xyxy', 'xywh'], "Invalid input format. Choose 'xyxy' or 'xywh'."
    assert in_type in ['array', 'list'], "Invalid input type. Choose 'array' or 'list'."
    assert in_dim in ['single', 'multi'], "Invalid input multiplicity. Choose 'single' or 'multi'."
    # Check output type
    assert out_form in ['xyxy', 'xywh'], "Invalid output format. Choose 'xyxy' or 'xywh'."
    assert out_type in ['array', 'list'], "Invalid output type. Choose 'array' or 'list'."
    assert out_dim in ['single', 'multi'], "Invalid output multiplicity. Choose 'single' or 'multi'."

    bbox = np.array(bbox)
    if bbox.ndim == 1:
        bbox = bbox.reshape(-1, 4)

    if in_form == 'xywh':
        bbox[..., 2] += bbox[..., 0]
        bbox[..., 3] += bbox[..., 1]
    
    # hanndle negative width/height
    x1, y1, x2, y2 = bbox[...,0], bbox[...,1], bbox[...,2], bbox[...,3]
    mnx, mny = np.minimum(x1, x2), np.minimum(y1, y2)
    mxx, mxy = np.maximum(x1, x2), np.maximum(y1, y2)
    bbox[...,0], bbox[...,1], bbox[...,2], bbox[...,3] = mnx, mny, mxx, mxy

    if out_form == 'xywh':
        bbox[..., 2] -= bbox[..., 0]
        bbox[..., 3] -= bbox[..., 1]

    if out_dim == 'single':
        bbox = bbox.flatten()

    if out_type == 'list':
        bbox = bbox.tolist()
    
    return bbox


def convert_obb(
    obbs: Union[List[float], List[List[float]], np.ndarray],
    format_in: OrientedBoxFormat = 'cxcywha_list_single',
    format_out: OrientedBoxFormat = 'cxcywha_list_single'
) -> Union[np.ndarray, List[float], List[List[float]]]:
    """
    Convert oriented bounding boxes (OBBs) between:
      - 'cxcywha' (center x, center y, width, height, angle in degrees)
      - 'pts'   (4 corner points as [[x1,y1],…,[x4,y4]])
      - 'flat'  (flattened [x1,y1,x2,y2,…,x4,y4])

    Supports array vs. list, and single vs. multi.
    """
    if format_in == format_out:
        return obbs
    # -- parse & validate formats --
    in_form, in_type, in_mult = format_in.split('_')
    out_form, out_type, out_mult = format_out.split('_')
    forms = {'cxcywha','pts','flat'}
    types = {'array','list'}
    mults = {'single','multi'}
    if in_form not in forms or out_form not in forms:
        raise ValueError(f"format must start with one of {forms}")
    if in_type not in types or out_type not in types:
        raise ValueError(f"format must contain type one of {types}")
    if in_mult not in mults or out_mult not in mults:
        raise ValueError(f"format must contain multiplicity one of {mults}")

    # -- normalize input to a Python list of OBB descriptors --
    raw = obbs if in_mult=='multi' else [obbs]
    obb_list = []
    for item in raw:
        arr = np.array(item, dtype=float)
        if in_form == 'cxcywha':
            cx, cy, w, h, angle = arr.ravel()
        else:
            # segmentation → minimal-area OBB
            pts = arr.reshape(-1, 2).astype(np.float32)
            (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
        obb_list.append((float(cx), float(cy), float(w), float(h), float(angle)))

    # -- build outputs --
    outs = []
    for cx, cy, w, h, angle in obb_list:
        if out_form == 'cxcywha':
            out_arr = np.array([cx, cy, w, h, angle], dtype=float)
        else:
            # build 4 corners of the rotated rect
            theta = np.deg2rad(angle)
            dx = np.array([-w/2,  w/2,  w/2, -w/2], dtype=float)
            dy = np.array([-h/2, -h/2,  h/2,  h/2], dtype=float)
            xs = cx + dx * np.cos(theta) - dy * np.sin(theta)
            ys = cy + dx * np.sin(theta) + dy * np.cos(theta)
            pts = np.stack([xs, ys], axis=1)
            out_arr = pts.flatten() if out_form=='flat' else pts

        # array vs list
        if out_type == 'list':
            out_item = out_arr.tolist()
        else:
            out_item = np.array(out_arr, dtype=float)

        outs.append(out_item)

    # single vs multi
    return outs if out_mult=='multi' else outs[0]


def convert_point(
    point: Union[List[float], List[List[float]], np.ndarray],
    format_in: PointFormat = 'array_single',
    format_out: PointFormat = 'array_single'
) -> Union[np.ndarray, List[float], List[List[float]]]:
    """
    Convert a point or list of points between different formats:
    - array: numpy array
    - list: Python list
    - single: a single point [x, y]
    - multi: a list of points [[x1, y1], [x2, y2], ...]

    Args:
        point: A single point as [x, y] or multiple points as [[x1, y1], [x2, y2], ...]; can be list or np.ndarray.
        format_in: Format of the input point(s), in the form '<array|list>_<single|multi>'.
        format_out: Desired output format, in the same form.

    Returns:
        The point(s) converted to the specified format.
    """
    if format_in == format_out:
        return point

    form_in, in_mult = format_in.split('_')
    form_out, out_mult = format_out.split('_')

    # Check input type
    assert form_in in ['array', 'list'], "Invalid input format. Choose 'array' or 'list'."
    assert in_mult in ['single', 'multi'], "Invalid input multiplicity. Choose 'single' or 'multi'."
    # Check output type
    assert form_out in ['array', 'list'], "Invalid output format. Choose 'array' or 'list'."
    assert out_mult in ['single', 'multi'], "Invalid output multiplicity. Choose 'single' or 'multi'."
    
    arr = np.array(point)
    if in_mult == 'single':
        arr = arr.reshape(1, 2)

    if form_out == 'list':
        arr = arr.tolist()

    if out_mult == 'single':
        arr =  arr[0]
    
    return arr


# -- -- -- -- -- -- -- -- -- #
#                            #
#      geo coordinates       #
#                            #
# -- -- -- -- -- -- -- -- -- #


def convert_anything_with_lerp(
    input: Union[List, np.ndarray],
    bbox_pix_image: Union[List, np.ndarray, Tuple],
    bbox_epsg_image: Union[List, np.ndarray, Tuple],
    object_in: Literal['segmentation', 'bbox', 'point'],
    format_in: Union[SegmentationFormat, BoxFormat, PointFormat],
    format_out: str = None,
    mode: Literal['pix_to_epsg', 'epsg_to_pix'] = 'pix_to_epsg',
    format_in_bbox_pix_image: BoxFormat = 'xyxy_list_single', 
    format_in_bbox_epsg_image: BoxFormat = 'xyxy_list_single', 
) -> Union[List, np.ndarray]:
    """
    Convert objects (segmentation, bounding box, or point) between pixel and EPSG coordinate spaces using linear interpolation.

    Args:
        input: The object(s) to convert (segmentation, bbox, or point). Can be a list or np.ndarray.
        bbox_pix_image: Bounding box in pixel coordinates (either [x_min, y_min, x_max, y_max] or [x, y, w, h]).
        bbox_epsg_image: Bounding box in EPSG coordinates (same structure as bbox_pix_image).
        object_in: Type of object to convert: 'segmentation', 'bbox', or 'point'.
        format_in: Format of the input object (e.g., 'pts_list_single', 'xyxy_list_multi', etc.).
        format_out: Desired output format. If None, the output will use format_in.
        mode: Conversion direction. Must be either 'pix_to_epsg' or 'epsg_to_pix'.
        format_bbox: Format of the bounding box, either 'xyxy' (x1, y1, x2, y2) or 'xywh' (x, y, width, height).

    Returns:
        The converted object in the specified output format.
    """
    if len(bbox_pix_image) == 2:
        bbox_pix_image = [0, 0, bbox_pix_image[0], bbox_pix_image[1]]
        format_in_bbox_pix_image = 'xywh_list_single'

    bbox_pix_image = convert_bbox(
        bbox_pix_image,
        format_in=format_in_bbox_pix_image,
        format_out='xyxy_list_single',
    )

    bbox_epsg_image = convert_bbox(
        bbox_epsg_image,
        format_in=format_in_bbox_epsg_image,
        format_out='xyxy_list_single',
    )

    if format_out is None:
        format_out = format_in
    

    # WARNING work for multiple bboxes points and single segmentation

    if mode not in ['pix_to_epsg', 'epsg_to_pix']:
        raise ValueError("Invalid mode. Choose 'pix_to_epsg' or 'epsg_to_pix'.")

    if object_in not in ['segmentation', 'bbox', 'point']:
        raise ValueError("Invalid object type. Choose 'segmentation', 'bbox', or 'point'.")
    
    format_bbox = 'xyxy'
    if format_bbox == 'xyxy':
        x_geo_min, y_geo_min, x_geo_max, y_geo_max = bbox_epsg_image
        x_pix_min, y_pix_min, x_pix_max, y_pix_max = bbox_pix_image


        w_pix = x_pix_max - x_pix_min
        h_pix = y_pix_max - y_pix_min
        w_geo = x_geo_max - x_geo_min
        h_geo = y_geo_max - y_geo_min
    else: 
        x_geo_min, y_geo_min, w_geo, h_geo = bbox_epsg_image
        x_pix_min, y_pix_min, w_pix, h_pix = bbox_pix_image

        x_geo_max = x_geo_min + w_geo
        y_geo_max = y_geo_min + h_geo
        x_pix_max = x_pix_min + w_pix
        y_pix_max = y_pix_min + h_pix


    if object_in == 'segmentation':

        segmentation = convert_segmentation(input, format_in = format_in, format_out = 'pts_array_single')
        #print(segmentation, segmentation.shape)

        epsg_segmentation = np.zeros_like(segmentation, dtype=np.float32)
        if mode == 'pix_to_epsg':
            epsg_segmentation[:, 0] = x_geo_min + (segmentation[:, 0]) / w_pix * w_geo
            epsg_segmentation[:, 1] = y_geo_max - (segmentation[:, 1]) / h_pix * h_geo  # Flip Y-axis
        elif mode == 'epsg_to_pix':
            epsg_segmentation[:, 0] = (segmentation[:, 0] - x_geo_min) / w_geo * w_pix
            epsg_segmentation[:, 1] = (y_geo_max - segmentation[:, 1]) / h_geo * h_pix

        segmentation = convert_segmentation(epsg_segmentation, format_in = 'pts_array_single', format_out = format_out)
        return segmentation
    
    elif object_in == 'bbox':
        
        bbox = convert_bbox(input, format_in = format_in, format_out = 'xyxy_array_multi')

        epsg_bbox = np.zeros_like(bbox, dtype=np.float32)
        if mode == 'pix_to_epsg':
            epsg_bbox[..., 0] = x_geo_min + (bbox[..., 0]) / w_pix * w_geo
            epsg_bbox[..., 1] = y_geo_max - (bbox[..., 1]) / h_pix * h_geo
            epsg_bbox[..., 2] = x_geo_min + (bbox[..., 2]) / w_pix * w_geo
            epsg_bbox[..., 3] = y_geo_max - (bbox[..., 3]) / h_pix * h_geo
        elif mode == 'epsg_to_pix':
            epsg_bbox[..., 0] = (bbox[..., 0] - x_geo_min) / w_geo * w_pix
            epsg_bbox[..., 1] = (y_geo_max - bbox[..., 1]) / h_geo * h_pix
            epsg_bbox[..., 2] = (bbox[..., 2] - x_geo_min) / w_geo * w_pix
            epsg_bbox[..., 3] = (y_geo_max - bbox[..., 3]) / h_geo * h_pix

        bbox = convert_bbox(epsg_bbox, format_in = 'xyxy_array_multi', format_out = format_out)
        return bbox
    
    elif object_in == 'point':

        point = convert_point(input, format_in = format_in, format_out = 'array_multi')

        epsg_point = np.zeros_like(point, dtype=np.float32)
        if mode == 'pix_to_epsg':
            epsg_point[..., 0] = x_geo_min + (point[..., 0]) / w_pix * w_geo
            epsg_point[..., 1] = y_geo_max - (point[..., 1]) / h_pix * h_geo
        elif mode == 'epsg_to_pix':
            epsg_point[..., 0] = (point[..., 0] - x_geo_min) / w_geo * w_pix
            epsg_point[..., 1] = (y_geo_max - point[..., 1]) / h_geo * h_pix

        point = convert_point(epsg_point, format_in = 'array_multi', format_out = format_out)
        return point
    else:
        raise ValueError("Invalid object type. Choose 'segmentation', 'bbox', or 'point'.")


def convert_anything_with_transform( # Warning TODO VERIFY
    input: Union[List, np.ndarray],
    transform: Callable,
    object_in: Literal['segmentation', 'bbox', 'point'],
    format_in: str,
    format_out: str = None,
    inverted: bool = False,
) -> Union[List, np.ndarray]:
    """
    Convert objects (segmentation, bounding box, or point) between pixel and EPSG coordinate spaces using linear interpolation.

    Args:
        input: The object(s) to convert (segmentation, bbox, or point). Can be a list or np.ndarray.
        bbox_pix_image: Bounding box in pixel coordinates (either [x_min, y_min, x_max, y_max] or [x, y, w, h]).
        bbox_epsg_image: Bounding box in EPSG coordinates (same structure as bbox_pix_image).
        object_in: Type of object to convert: 'segmentation', 'bbox', or 'point'.
        format_in: Format of the input object (e.g., 'pts_list_single', 'xyxy_list_multi', etc.).
        format_out: Desired output format. If None, the output will use format_in.
        mode: Conversion direction. Must be either 'pix_to_epsg' or 'epsg_to_pix'.
        format_bbox: Format of the bounding box, either 'xyxy' (x1, y1, x2, y2) or 'xywh' (x, y, width, height).

    Returns:
        The converted object in the specified output format.
    """

    if format_out is None:
        format_out = format_in

    # WARNING work for multiple bboxes points and single segmentation
    transform = ~transform if inverted else transform

    mapper = {
        'segmentation': (convert_segmentation, 'pts_array_single'),
        'bbox':         (convert_bbox,       'xyxy_array_multi'),
        'point':        (convert_point,      'array_multi'),
    }

    if object_in not in ['segmentation', 'bbox', 'point']:
        raise ValueError("Invalid object type. Choose 'segmentation', 'bbox', or 'point'.")

    convert_function, norm_fmt = mapper[object_in]

    # 2) normalize input → arr of shape (..., 2) (..., 4) for bbox
    pts = convert_function(input, format_in=format_in, format_out=norm_fmt)
    
    orig_shape = pts.shape
    flat_pts   = pts.reshape(-1, 2)
    x_new, y_new = transform(flat_pts[:, 0], flat_pts[:, 1])
    pts = np.stack([x_new, y_new], axis=-1).reshape(orig_shape)

    # 6) denormalize and return
    return convert_function(pts, format_in=norm_fmt, format_out=format_out)


# -- -- -- -- -- -- -- -- -- #
#                            #
#   convert type of format   #
#                            #
# -- -- -- -- -- -- -- -- -- #

def convert_segmentation_to_bbox(
    segmentation: Union[List[float], List[List[float]], List[np.ndarray], np.ndarray],
    format_in: SegmentationFormat = 'pts_list_multi',
    format_out: BoxFormat = 'xyxy_list_single'
) -> Union[List, np.ndarray]:
    """
    Given a polygonal segmentation, compute the minimal 4-point bounding box and
    return it as a segmentation in the desired format.

    Args:
        segmentation: One or more polygons, in any of your supported formats.
        format_in:   How `segmentation` is represented. Must be
                     '<pts|list>_<array|list>_<single|multi>'.
                     Defaults to 'pts_list_multi' (flat list of floats, Python list, multi-poly).
        format_out:  Same format specifier for the returned bbox-segmentation.
                     Defaults to the same as `format_in`.

    Returns:
        A 4-vertex “bbox” polygon (or list of one polygon) in `format_out`.

    Raises:
        TypeError:  If `segmentation` isn’t a list, tuple or ndarray.
        ValueError: If either format string isn’t one of the eight
                    '<pts|list>_<array|list>_<single|multi>' options.
    """

    segmentation = convert_segmentation(
        segmentation,
        format_in=format_in,
        format_out='pts_array_multi'
    )

    all_pts = np.vstack(segmentation)  # shape (total_pts, 2)
    xs, ys = all_pts[:, 0], all_pts[:, 1]
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    return convert_bbox(
        [x_min, y_min, x_max, y_max],
        format_in='xyxy_list_single',
        format_out=format_out
    )


def convert_bbox_to_segmentation(
    bboxes: Union[List[float], List[List[float]], List[np.ndarray], np.ndarray],
    format_in: BoxFormat = 'xyxy_list_single',
    format_out: SegmentationFormat = 'pts_list_multi'
) -> Union[List, np.ndarray]:
    """
    Given one or more bounding boxes, return their 4-vertex polygon segmentation
    in the desired format.

    Args:
        bboxes:     One or more boxes, in any of your supported bbox-formats.
        format_in:  How `bboxes` is represented. Must be
                    '<xyxy|xywh>_<array|list>_<single|multi>'.
                    Defaults to 'xyxy_list_single'.
        format_out: How the returned segmentation should be formatted.
                    Must be one of your eight '<pts|list>_<array|list>_<single|multi>'.
                    Defaults to 'pts_list_multi'.

    Returns:
        A rectangle polygon (or list of rectangles) as segmentation(s) in `format_out`.

    Raises:
        ValueError: If either format string isn’t one of the supported options.
    """

    # 1) normalize all boxes to a list of [x_min,y_min,x_max,y_max]
    bboxes = convert_bbox(
        bboxes,
        format_in=format_in,
        format_out='xyxy_list_multi'
    )
    
    polys = []
    for x_min, y_min, x_max, y_max in bboxes:
        polys.append([
            x_min, y_min,
            x_max, y_min,
            x_max, y_max,
            x_min, y_max
        ])
    
    return convert_segmentation(
        polys,
        format_in='pts_list_multi',
        format_out=format_out
    )


def convert_anything_to_segmentation( # TODO TEST
    obj: Union[List, np.ndarray],
    object_in: Literal['segmentation', 'bbox', 'obb', 'point'],
    format_in: Union[SegmentationFormat, BoxFormat, OrientedBoxFormat, PointFormat],
    format_out: SegmentationFormat = 'pts_list_multi'
) -> Union[List[float], List[List[float]], np.ndarray]:
    
    if object_in == 'segmentation':
        return convert_segmentation(
            obj, 
            format_in=format_in, 
            format_out=format_out
        )
    
    elif object_in == 'bbox':
        return convert_bbox_to_segmentation(
            obj, 
            format_in=format_in, 
            format_out=format_out
        )

    elif object_in == 'obb':
        # 1) get 4 corners as pts_array_multi, 
        corners = convert_obb(
            obj,
            format_in=format_in,
            format_out='pts_array_multi'
        )
        # 2) re‐encode into desired segmentation format
        return convert_segmentation(
            corners,
            format_in='pts_array_multi',
            format_out=format_out
        )
    elif object_in == 'point':
        # 1) normalize into array_multi
        pts = convert_point(
            obj, 
            format_in=format_in, 
            format_out='array_multi'
        )
        # 2) wrap as a single‐vertex polygon [[x, y]]
        return convert_segmentation(
            [pts], 
            format_in='pts_array_multi', 
            format_out=format_out
        )
    else:
        raise ValueError(f"Unknown object type '{object_in}'. Must be one of "
                         "'segmentation', 'bbox', 'obb', or 'point'.")


def convert_segmentation_to_anything( # TODO TEST
    segmentation: Union[List[float], List[List[float]], np.ndarray, List[np.ndarray]],
    format_in: SegmentationFormat = 'pts_list_multi',
    object_out: Literal['segmentation', 'bbox', 'obb', 'point'] = 'segmentation',
    format_out: Union[SegmentationFormat, BoxFormat, OrientedBoxFormat, PointFormat] = None
) -> Union[
    List[float], List[List[float]], np.ndarray, List[np.ndarray]
]:
    """
    Convert a segmentation into another annotation type, using whatever you've already implemented.

    Args:
        segmentation:   Input polygon(s).
        format_in:      How that polygon is represented (one of your SegmentationFormats).
        object_out:     Desired output type: 
                        - 'segmentation' (just re-format)
                        - 'bbox'         (axis-aligned box)
                        - 'obb'          (oriented box)
                        - 'point'        (not supported)
        format_out:     Desired format (SegmentationFormat, BoxFormat, or OrientedBoxFormat).
                        If None, defaults to:
                          segmentation → same as format_in
                          bbox         → 'xyxy_list_single'
                          obb          → 'cxcywha_list_single'
    """
    # defaults
    if format_out is None:
        if object_out == 'segmentation':
            format_out = format_in
        elif object_out == 'bbox':
            format_out = 'xyxy_list_single'
        elif object_out == 'obb':
            format_out = 'cxcywha_list_single'
        else:
            raise ValueError("Must specify format_out for 'point' or supply a different default")

    if object_out == 'segmentation':
        return convert_segmentation(
            segmentation,
            format_in=format_in,
            format_out=format_out
        )

    elif object_out == 'bbox':
        # minimal axis-aligned bbox
        return convert_segmentation_to_bbox(
            segmentation,
            format_in=format_in,
            format_out=format_out  # type: ignore[arg-type]
        )

    elif object_out == 'obb':
        # minimal-area oriented box: get pts → feed into convert_obb
        # first, get points array
        pts = convert_segmentation(
            segmentation,
            format_in=format_in,
            format_out='pts_array_single'
        )
        return convert_obb(
            pts,
            format_in='pts_array_single',
            format_out=format_out  # type: ignore[arg-type]
        )

    elif object_out == 'point':
        raise NotImplementedError(
            "Conversion from segmentation to a single point isn't defined. "
            "You could compute a centroid, but you'd need to add that yourself."
        )

    else:
        raise ValueError(
            f"Unknown target type '{object_out}'. "
            "Choose one of 'segmentation', 'bbox', 'obb', or 'point'."
        )


def convert_anything_to_anything( # TODO TEST
    obj: Union[List, np.ndarray],
    object_in: Literal['segmentation', 'bbox', 'obb', 'point'],
    format_in: Union[SegmentationFormat, BoxFormat, OrientedBoxFormat, PointFormat],
    object_out: Literal['segmentation', 'bbox', 'obb', 'point'],
    format_out: Optional[Union[SegmentationFormat, BoxFormat, OrientedBoxFormat, PointFormat]] = None
) -> Union[List, np.ndarray]:
    """
    Convert any annotation to any other by:
      1) converting input → segmentation (pts_array_multi)
      2) converting that segmentation → desired output

    Relies solely on your:
      - convert_anything_to_segmentation(...)
      - convert_segmentation_to_anything(...)
    """

    # 1) normalization → polygon segmentation
    seg = convert_anything_to_segmentation(
        obj,
        object_in=object_in,
        format_in=format_in,
        format_out='pts_array_multi'
    )

    # 2) polygon → desired type
    return convert_segmentation_to_anything(
        segmentation=seg,
        format_in='pts_array_multi',
        object_out=object_out,
        format_out=format_out
    )



def get_center(segmentation, category_id):
    raise "todo"
    segmentation = pre_process_segmentation(segmentation)

    x = segmentation[:, 0]
    y = segmentation[:, 1]
    center_x = float(np.mean(x))
    center_y = float(np.mean(y))
    
    return [center_x, center_y]


def get_segmentation(segmentation, category_id):
    raise 'TODO'
    segmentation = pre_process_segmentation(segmentation)

    if category_id == 0 or category_id == 2:
        segm = segmentation.flatten().tolist()

    elif category_id == 1:
        
        x = segmentation[:, 0]
        y = segmentation[:, 1]

        # Build system:  [2x_i, 2y_i, 1] [a, b, c]^T = [x_i^2 + y_i^2]
        M = np.column_stack((2*x, 2*y, np.ones(len(segmentation))))
        d = x**2 + y**2

        (center_x, center_y, c), *_ = np.linalg.lstsq(M, d, rcond=None)

        radius = np.sqrt(center_x**2 + center_y**2 + c)

        x_min = float(center_x - radius)
        y_min = float(center_y - radius)
        x_max = float(center_x + radius)
        y_max = float(center_y + radius)
        segm = [
            x_min, y_min,
            x_max, y_min,
            x_max, y_max,
            x_min, y_max,
            ]
    else:
        raise 'Not a recognized class'

    return segm




# -- -- -- -- -- -- -- -- -- #
#                            #
#         category           #
#                            #
# -- -- -- -- -- -- -- -- -- #


def segmentation_to_obb_tank(segmentation):
    

    # compute from the 4 points touching edge of the tank
    # the centre point and the radius

    points = np.array(segmentation, dtype=np.int32).reshape(-1, 2)
    x = points[:, 0]
    y = points[:, 1]
    
    # Build system:  [2x_i, 2y_i, 1] [a, b, c]^T = [x_i^2 + y_i^2]
    M = np.column_stack((2*x, 2*y, np.ones(len(points))))
    d = x**2 + y**2
    
    (center_x, center_y, c), *_ = np.linalg.lstsq(M, d, rcond=None)
    
    radius = np.sqrt(center_x**2 + center_y**2 + c)

    obb = [
        [center_x - radius, center_y - radius],
        [center_x + radius ,center_y - radius],
        [center_x + radius, center_y + radius],
        [center_x - radius, center_y + radius]
     ]
    
    print(f'points = {points}')
    print(f'obb = {obb}') 
    
    # Get the minimum area rectangle
    rect = cv2.minAreaRect(np.array(obb, dtype=np.float32))
    (cx, cy), (w, h), angle = rect

    # Get the 4 corner points of the rotated bounding box
    box = cv2.boxPoints(rect)  # Returns 4 points
    box = np.intp(box)  # Convert to integer # WARNING Integer

    # (cx, cy, w, h, angle), 
    print('TANK ',angle/180)

    return obb #box.tolist()


def convert_to_dota_format(annotation):
    """
    Convert COCO-style annotation to DOTA format.

    Args:
        annotation (dict): COCO annotation with 'segmentation' or 'bbox'.
        category_mapping (dict): Mapping from category_id to category name.

    Returns:
        str: Annotation in DOTA format.
    """
    category_id_to_name = {
        1: "tank",
        2: "pile",
        3: "all"
    }

    # Extract COCO annotation data
    category_id = annotation["category_id"]
    category_name = category_id_to_name[category_id]
    difficult = 0  # Default difficulty level
    obb = annotation["obb"]

    box = [obb[0][0], obb[0][1], obb[1][0], obb[1][1], obb[2][0], obb[2][1], obb[3][0], obb[3][1]]

    # Convert to DOTA format string
    dota_label = " ".join(map(str, box)) + f" {category_name} {difficult}\n"

    return dota_label


def compute_new_short_axis(long_axis, short_axis):
    perp_axis = np.array([-long_axis[1], long_axis[0]])
    perp_unit = perp_axis / np.linalg.norm(perp_axis)
    projection = np.dot(short_axis, perp_unit) * perp_unit
    return projection


# -- -- -- -- -- -- -- -- -- #
#                            #
#       visualisation        #
#                            #
# -- -- -- -- -- -- -- -- -- #

def write_svg(
    objects,
    colors,
    object_in: AnyObject,
    format_in: AnyFormat,
    save_path: str,
    overwrite: bool = True
):
    svg_polygons = []
    for ann, color in zip(objects, colors):
        seg = convert_anything_to_segmentation(
            ann,
            object_in = object_in,
            format_in = format_in,
            format_out = 'flat_list_single'
        )

        pts = []
        for i in range(0, len(seg), 2):
            x_pt = seg[i]
            y_pt = seg[i+1]
            pts.append(f"{x_pt},{y_pt}")
        points_str = " ".join(pts)

        if isinstance(color, int):
            color = COLOR_MAP_INT[color]

        svg_polygons.append({
            "points": points_str,
            "color": color
        })

    if not overwrite and os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Remove the closing </svg> tag
        content = content.rstrip().rsplit("</svg>", 1)[0]

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(content)
            for poly in svg_polygons:
                pts_str = poly["points"]
                color   = poly["color"]
                f.write(f'  <polygon points="{pts_str}" stroke="{color}" stroke-width="0.1" fill="none"/>\n')
            f.write("</svg>\n")
    else:
        if os.path.exists(save_path):
            os.remove(save_path)

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(f'<svg width="1" height="1">\n')
            for poly in svg_polygons:
                pts_str = poly["points"]
                color   = poly["color"]
                f.write(f'  <polygon points="{pts_str}" stroke="{color}" stroke-width="0.1" fill="none"/>\n')
            f.write("</svg>\n")


def write_dota(
    objects,
    classes,
    object_in: AnyObject,
    format_in: AnyFormat,
    save_path: str,
    overwrite: bool = True
):
    """
    Write annotations in DOTA format (1 object per line, 8 coords + label).
    Assumes polygons or OBBs.
    """
    if overwrite and os.path.exists(save_path):
        os.remove(save_path)

    lines = []

    for ann, label in zip(objects, classes):
        # Convert to 4-point polygon (8 coordinates) if needed
        seg = convert_anything_to_segmentation(
            ann,
            object_in=object_in,
            format_in=format_in,
            format_out='flat_list_single'
        )

        if len(seg) != 8:
            raise "Only 4 Points Are Allowed"
        
        if isinstance(label, int):
            warnings.warn("Converting integer label to class name using IDX_TO_CLASS", category=UserWarning)
            label = IDX_TO_CLASS[label]

        coords = " ".join([f"{x:.2f}" for x in seg])
        line = f"{coords} {label} 0"
        lines.append(line)

    with open(save_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def convert_label_segmentation_to_anything(
    segmentation: Union[List, np.ndarray],
    label: Union[int, str],
    format_in: str = 'flat_list_multi',
    object_out: Literal['segmentation', 'bbox', 'obb'] = 'segmentation',
    format_out: Optional[Union[SegmentationFormat, BoxFormat, OrientedBoxFormat, PointFormat]] = None,
) -> Union[List, np.ndarray]:
    
    label_list = ['all','tank','pile']
    
    if isinstance(label, str): 
        assert label in label_list
    
    elif isinstance(label, int):
        assert label in list(range(1,len(label_list)+1))
        label = label_list[label-1]   

    if object_out not in ['segmentation', 'bbox', 'obb']:
        raise ValueError("Invalid object type. Choose 'segmentation', 'bbox', or 'point'.")


    segmentation = convert_segmentation(
        segmentation,
        format_in = format_in,
        format_out = 'pts_array_single',
    )


    x = segmentation[:, 0]
    y = segmentation[:, 1]

    if label == 'tank':

        # Build system:  [2x_i, 2y_i, 1] [a, b, c]^T = [x_i^2 + y_i^2]
        M = np.column_stack((2*x, 2*y, np.ones(len(segmentation))))
        d = x**2 + y**2

        (center_x, center_y, c), *_ = np.linalg.lstsq(M, d, rcond=None)
        radius = np.sqrt(center_x**2 + center_y**2 + c)
        x_min = center_x - radius
        y_min = center_y - radius
        x_max = center_x + radius
        y_max = center_y + radius

        if object_out == 'bbox':
            return convert_bbox(
                [x_min, y_min, x_max, y_max],
                format_in = 'xyxy_list_single',
                format_out = format_out,
            )
        elif object_out == 'obb':
            return convert_obb(
                [center_x, center_y, 2*radius, 2*radius, 0],
                format_in = 'cxcywha_list_single',
                format_out = format_out,
            )
        elif object_out == 'segmentation':
            angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
            seg_x = center_x + radius * np.cos(angles)
            seg_y = center_y + radius * np.sin(angles)
            return convert_segmentation(
                np.stack((seg_x, seg_y), axis=1),  # shape (8, 2)
                format_in = 'pts_array_single',
                format_out = format_out,
            )
    else:
        if object_out == 'bbox':
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            return convert_bbox(
                [x_min, y_min, x_max, y_max],
                format_in = 'xyxy_list_single',
                format_out = format_out,
            )
        elif object_out == 'obb':
            (center_x,center_y),(w,h),angle = cv2.minAreaRect(segmentation.astype(np.float32))
            return convert_obb(
                [center_x, center_y, w, h, angle],
                format_in = 'cxcywha_list_single',
                format_out = format_out,
            )            
        elif object_out == 'segmentation':
            return convert_segmentation(
                segmentation,
                format_in = 'pts_array_single',
                format_out = format_out,
            )


def pre_process_part(
    part_detection
):

    part_detection['segmentation'] = convert_label_segmentation_to_anything(
        segmentation = part_detection['segmentation'],
        label = part_detection['category_id'],
        format_in = 'flat_list_multi',
        object_out = 'segmentation',
    )



CATEGORY_STYLE = {
    "all":     {"color": (232, 23, 237),   "alpha": 0.0, "thickness": 5},
    "tank":    {"color": (255, 69, 0)  ,   "alpha": 0.0, "thickness": 5},
    "pile":    {"color": (30, 173, 255),   "alpha": 0.0, "thickness": 5},
}

COLOR_MAP_INT = {
    1: 'red',
    2: 'green',
    3: 'blue',
}

COLOR_MAP_STR = {
    "all":  (232, 23, 237) ,
    "tank": (255, 69, 0)  ,
    "pile": (30, 173, 255),
}

IDX_TO_CLASS = {
    1: 'all',
    2: 'tank',
    3: 'pile',
}

CLASS_TO_IDX = {
    'all': 1,
    'tank': 2,
    'pile': 3,
}

# -- -- -- -- -- -- -- -- -- #
#                            #
#        depreciated         #
#                            #
# -- -- -- -- -- -- -- -- -- #

'''
def transform_tank_label(segmentation):

    return 


def transform_pile_label(segmentation):
    poly_points = np.array(segmentation, dtype=np.int32).reshape(-1, 2)

    if len(poly_points) != 4:
        return 'poly', poly_points
    
    axis_1 = ((poly_points[0] - poly_points[1]) + (poly_points[3] - poly_points[2]))/4
    axis_2 = ((poly_points[1] - poly_points[2]) + (poly_points[0] - poly_points[3]))/4

    if np.linalg.norm(axis_1) > np.linalg.norm(axis_2):
        axis_long = axis_1
        axis_short = axis_2
    else:
        axis_long = axis_2
        axis_short = axis_1
    
    axis_short = compute_new_short_axis(axis_long, axis_short)

    centre = (poly_points[0] + poly_points[1] + poly_points[2] + poly_points[3])/4

    corrected_poly_points = np.zeros_like(poly_points)
    corrected_poly_points[0] = centre + axis_long + axis_short
    corrected_poly_points[1] = centre + axis_long - axis_short
    corrected_poly_points[2] = centre - axis_long - axis_short
    corrected_poly_points[3] = centre - axis_long + axis_short

    return 'poly', corrected_poly_points


def transform_all_label(segmentation):
    poly_points = np.array(segmentation, dtype=np.int32).reshape(-1, 2)
    return 'poly', poly_points
def get_bbox(segmentation, category_id):
    raise 'TODO'
    segmentation = pre_process_segmentation(segmentation)

    if category_id == 0 or category_id == 2:
        x_min = np.min(segmentation[:, 0])
        y_min = np.min(segmentation[:, 1])
        x_max = np.max(segmentation[:, 0])
        y_max = np.max(segmentation[:, 1])

    elif category_id == 1:
        
        x = segmentation[:, 0]
        y = segmentation[:, 1]

        # Build system:  [2x_i, 2y_i, 1] [a, b, c]^T = [x_i^2 + y_i^2]
        M = np.column_stack((2*x, 2*y, np.ones(len(segmentation))))
        d = x**2 + y**2

        (center_x, center_y, c), *_ = np.linalg.lstsq(M, d, rcond=None)

        radius = np.sqrt(center_x**2 + center_y**2 + c)

        x_min = center_x - radius
        y_min = center_y - radius
        x_max = center_x + radius
        y_max = center_y + radius
    else:
        raise 'Not a recognized class'

    x_min = float(x_min)
    y_min = float(y_min)
    x_max = float(x_max)
    y_max = float(y_max)
    
    width = x_max - x_min
    height = y_max - y_min

    return {
        'bbox_xyxy': [x_min, y_min, x_max, y_max],
        'bbox_xywh': [x_min, y_min, width, height],
        'bbox_segm': [
            x_min, y_min,
            x_max, y_min,
            x_max, y_max,
            x_min, y_max,
            ]
    }


def get_obb(segmentation, category_id):
    raise 'TODO'
    segmentation = pre_process_segmentation(segmentation)

    if category_id == 0 or category_id == 2:

        #print(segmentation)
        #print(type(segmentation))
        #print(segmentation.shape)

        # Get the minimum area rectangle
        rect = cv2.minAreaRect(segmentation)
        (cx, cy), (w, h), angle = rect

        # Get the 4 corner points of the rotated bounding box
        box = cv2.boxPoints(rect)  # Returns 4 points
        box = np.intp(box)  # Convert to integer # WARNING INTEGER

        # (cx, cy, w, h, angle), 
        #print('OTHER',angle/180)

        obb_segm = box.flatten().tolist()
        obb_cxcywha = [cx, cy, w, h, angle]

    elif category_id == 1: # tank
        
        x = segmentation[:, 0]
        y = segmentation[:, 1]

        # Build system:  [2x_i, 2y_i, 1] [a, b, c]^T = [x_i^2 + y_i^2]
        M = np.column_stack((2*x, 2*y, np.ones(len(segmentation))))
        d = x**2 + y**2

        (center_x, center_y, c), *_ = np.linalg.lstsq(M, d, rcond=None)

        radius = np.sqrt(center_x**2 + center_y**2 + c)

        x_min = float(center_x - radius)
        y_min = float(center_y - radius)
        x_max = float(center_x + radius)
        y_max = float(center_y + radius)

        center_x = float(center_x)
        center_y = float(center_y)
        radius   = float(radius)

        obb_segm = [
            x_min, y_min,
            x_max, y_min,
            x_max, y_max,
            x_min, y_max,
            ]
        
        obb_cxcywha = [center_x, center_y, 2*radius, 2*radius, 0]
    else:
        raise 'Not a recognized class'

    
    return {
        'obb_segm': obb_segm,
        'obb_cxcywha': obb_cxcywha
    }

def convert_segmentation_to_obb(
    segmentation: Union[List[float], List[List[float]], List[np.ndarray], np.ndarray],
    format_in: Literal[
        'pts_array_single','pts_array_multi','pts_list_single','pts_list_multi',
        'list_array_single','list_array_multi','list_list_single','list_list_multi'
    ] = 'pts_list_multi',
    format_out: Optional[Literal[
        'pts_array_single','pts_array_multi','pts_list_single','pts_list_multi',
        'list_array_single','list_array_multi','list_list_single','list_list_multi'
    ]] = None
) -> Union[List[float], List[List[float]], List[np.ndarray], np.ndarray]:
    """
    Compute the minimum-area (oriented) bounding box of a polygonal segmentation
    and return its 4 corners as a new segmentation.

    Args:
        segmentation: One or more polygons, in any supported format.
        format_in:   Input format, one of
                     '<pts|list>_<array|list>_<single|multi>'.
                     Defaults to 'pts_list_multi' (flat-pts, Python list, multi-poly).
        format_out:  Output format, same options as format_in.
                     If None, defaults to format_in.

    Returns:
        A single 4‐vertex polygon (or list of one polygon) representing the OBB,
        in the specified format_out.

    Raises:
        TypeError:  If `segmentation` is not list/tuple/ndarray.
        ValueError: If any format string is not '<pts|list>_<array|list>_<single|multi>'.
    """

    # default output = input format
    if format_out is None:
        format_out = format_in
    
    polys = convert_segmentation(
        segmentation, format_in=format_in, format_out='pts_array_multi'
    )
    all_pts = np.vstack(polys).astype(np.float32)  # shape (total_pts, 2)

    # compute minimum-area rotated rectangle
    rect = cv2.minAreaRect(all_pts)       # ((cx,cy),(w,h),angle)
    #box = cv2.boxPoints(rect)             # 4×2 array of corner coordinates

    # return as a single 4-point segmentation
    return convert_segmentation(
        [box],                         # wrap in list for single-polygon API
        format_in='pts_array_multi',
        format_out=format_out
    )




def segmentation_to_obb_old(segmentation):
    # Convert segmentation to a NumPy array (Nx2 shape)
    points = np.array(segmentation, dtype=np.float32).reshape(-1, 2)

    # Get the minimum area rectangle
    rect = cv2.minAreaRect(points)
    (cx, cy), (w, h), angle = rect

    # Get the 4 corner points of the rotated bounding box
    box = cv2.boxPoints(rect)  # Returns 4 points
    box = np.intp(box)  # Convert to integer # WARNING Integer

    # (cx, cy, w, h, angle), 
    print('OTHER',angle/180)

    return box.tolist()


def convert_bbox_old(segmentation):

    points = np.array(segmentation, dtype=np.float32).reshape(-1, 2)
        # Compute the axis-aligned bounding box (AABB)
    xmin = float(np.min(points[:, 0]))
    ymin = float(np.min(points[:, 1]))
    xmax = float(np.max(points[:, 0]))
    ymax = float(np.max(points[:, 1]))

    # Compute width and height
    width = xmax - xmin
    height = ymax - ymin

    return [xmin, ymin, width, height]


def pre_process_segmentation_old(segmentation):

    """
    Known Formats: 
    1D_list - [[x1, y1, x2, y2, ...]]
    2D_list - [[x1, y1], [x2, y2], ...]
    1D_array - [[x1, y1, x2, y2, ...]]
    2D_array - [[x1, y1], [x2, y2], ...]
    """
    if isinstance(segmentation, list):

        if len(segmentation) == 1:
            segmentation = segmentation[0]

        segmentation = np.array(segmentation)
        segmentation = segmentation.reshape(-1, 2)
    else:
        if len(segmentation) == 1:
            segmentation = segmentation.reshape(-1, 2)
    
    return segmentation.astype(np.float32)


def post_process_segmentation_old(segmentation):

    if isinstance(segmentation, np.ndarray):
        return segmentation.flatten().tolist()  # Convert to flattened list
    return segmentation  # If already a list, return as is


def pix_to_epsg_old(segmentation, sub_tile):

    segmentation = pre_process_segmentation(segmentation)

    x_geo_min, y_geo_min, x_geo_max, y_geo_max = sub_tile['cartesian_bbox']
    x_pix_min, y_pix_min, x_pix_max, y_pix_max = sub_tile['pixel_bbox']

    pix_resolution = x_pix_max - x_pix_min
    cart_resolution = x_geo_max - x_geo_min

    #print(x_geo_min, y_geo_min, x_geo_max, y_geo_max)
    #print(x_pix_min, y_pix_min, x_pix_max, y_pix_max)
    #print(pix_resolution)
    #print(cart_resolution)
    #print(segmentation)

    # Convert pixel coordinates to EPSG:2154 coordinates
    epsg_segmentation = np.zeros_like(segmentation, dtype=np.float32)
    epsg_segmentation[:, 0] = x_geo_min + (segmentation[:, 0]) / pix_resolution * cart_resolution
    epsg_segmentation[:, 1] = y_geo_max - (segmentation[:, 1]) / pix_resolution * cart_resolution  # Flip Y-axis


    #print(epsg_segmentation)

    return epsg_segmentation.flatten().tolist()


def bbox_epsg_to_pix_old(bbox, sub_tile):

    x_min_epsg, y_min_epsg, w_epsg, h_epsg = bbox

    x_geo_min, y_geo_min, x_geo_max, y_geo_max = sub_tile["cartesian_bbox"]
    x_pix_min, y_pix_min, x_pix_max, y_pix_max = sub_tile["pixel_bbox"]

    # Compute resolutions
    pix_res = x_pix_max - x_pix_min  # width in pixel space
    cart_res = x_geo_max - x_geo_min

    x_min_pix = ((x_min_epsg - x_geo_min) / cart_res) * pix_res
    y_min_pix =  ((y_geo_max - y_min_epsg - h_epsg) / cart_res) * pix_res
    

    x = x_min_pix
    y = y_min_pix
    w = w_epsg / cart_res * pix_res
    h = h_epsg / cart_res * pix_res
    

    return x, y, w, h


def convert_segmentation(segmentation):
    segmentation = segmentation[0]
    points = np.array(segmentation, dtype=np.float32).reshape(-1, 2)

    points = (points - RESOLUTION_ORIGINAL//2) * RESOLUTION_NOUVELLE / RESOLUTION_ORIGINAL * METRE_CONVERSION + RESOLUTION_NOUVELLE // 2 # type: ignore

    segmentation = points.flatten().tolist()

    return [segmentation]
'''