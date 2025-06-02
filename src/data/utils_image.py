from typing import Literal, Union
import numpy as np
from PIL import Image

import warnings

import torch


def convert_image(
    image: Union[Image.Image, np.ndarray, str],
    format_in: str = 'pil_rgb',
    format_out: str = 'array_rgb_chw_int8',
) -> Union[Image.Image, np.ndarray]:
    """
    Convert images between PIL and NumPy representations, color spaces,
    array layouts, and dtypes.

    Args:
        image: Input image as a PIL.Image or NumPy array.
        format_in: Descriptor of input format.
        format_out: Descriptor of desired output format.

    Returns:
        Converted image in the requested format.
    """
    spaces = {'rgb', 'bgr', 'rgba', 'bgra'}
    layouts = {'chw', 'cwh', 'hwc', 'whc', '1chw', '1cwh', '1hwc', '1whc'}
    dtypes = {'int8', 'float32'}
    objects = {'pil', 'array', 'tensor', 'cv2', 'tiff'} # TODO cv2

    defaults = {
        'pil':   ('rgb',  'hwc', 'int8'),
        'cv2':   ('bgr',  'hwc', 'int8'),
        'array': ('rgb',  'hwc', 'int8'),
        'tensor':('rgb',  'chw', 'float32'),
        'tiff': ('rgb', 'chw', 'uint8')
    }

    # Parse formats
    def _split(fmt):
        space = layout = dtype = obj = None

        for tok in fmt.split('_'):
            if tok in spaces:
                space = tok
            elif tok in layouts:
                layout = tok
            elif tok in dtypes:
                dtype = tok
            elif tok in objects:
                obj = tok
            else:
                raise ValueError(f"Invalid format token: {tok!r}")

        obj = obj or 'array'

        def_space, def_layout, def_dtype = defaults[obj]
        space  = space  or def_space
        layout = layout or def_layout
        dtype  = dtype  or def_dtype

        return space, layout, dtype, obj

    in_space, in_layout, in_dtype, in_obj = _split(format_in)
    out_space, out_layout, out_dtype, out_obj = _split(format_out)
    
    # Warn on non-default loading settings
    def_space, def_layout, def_dtype = defaults[in_obj]
    if (in_space, in_layout, in_dtype) != (def_space, def_layout, def_dtype):
        warnings.warn(
            f"Non-default input format for '{in_obj}': "
            f"{in_space}_{in_layout}_{in_dtype} (default is "
            f"{def_space}_{def_layout}_{def_dtype})"
        )

    # --- 1) Bring input to a NumPy HWC uint8 RGB array ---
    if isinstance(image, str):
        image = Image.open(image)
    if isinstance(image, (Image.Image, torch.Tensor)):
        image = np.array(image)
        
    image = to_hwc(image, in_layout)

    # dtype
    if in_dtype != out_dtype:
        if in_dtype == 'float32':
            image = (image * 255.0).round().clip(0,255).astype(np.uint8)
        elif in_dtype == 'int8':
            image = (image.astype(np.float32) / 255.0)


    # color space
    if in_space != out_space:
        if 'a' in in_space:
            image = image[..., :3]
        if 'a' in out_space:
            h, w = image.shape[:2]
            alpha_val = 1 if out_dtype == 'float32' else 255
            alpha = np.full((h, w, 1), alpha_val, dtype=image.dtype)
            image = np.concatenate([image, alpha], axis=-1)

        if in_space[:3] != out_space[:3]:
            image[..., :3] = image[..., :3][..., ::-1]
        #raise NotImplementedError('color space convertion not suported')


    
    # Return PIL if requested
    image = from_hwc(image, out_layout)
    if out_obj == 'pil':

        if out_space == 'rgb' and image.shape[2] == 3:
            mode = 'RGB'
        elif out_space == 'rgba' and image.shape[2] == 4:
            mode = 'RGBA'
        else:
            raise AssertionError(
                "PIL mode output must be rgb or rgba"
            )
        
        if out_dtype == 'float32':
            raise AssertionError(
                "PIL multi-channel output must be int8; float32 not supported."
            )
        
        return Image.fromarray(image, mode=mode)
    elif out_obj == 'tensor':
        return torch.from_numpy(image)
    else:
        return image


def to_hwc(x: np.ndarray, layout: str) -> np.ndarray:
    # Remove 1 prefix dims
    shape = list(x.shape)
    if layout.startswith('1'):
        shape = shape[1:]
        x = x.reshape(shape)
        layout = layout[1:]
    # dimension order
    order = []
    for char in 'hwc':
        order.append(layout.index(char))
    x = np.transpose(x, tuple(order))
    return x

def from_hwc(x: np.ndarray, layout: str) -> np.ndarray:
    target = layout
    # Add leading 1 if needed
    add1 = layout.startswith('1')
    if add1:
        target = layout[1:]
    # determine axes
    axes = [None] * len(target)
    for i, char in enumerate(target):
        axes[i] = 'hwc'.index(char)
    x = np.transpose(x, tuple(axes))
    if add1:
        x = x[np.newaxis, ...]
    return x
